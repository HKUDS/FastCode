"""
Main FastCode Class - Orchestrate all components
"""

import os
import pickle
import logging
from typing import Optional, Dict, Any, List, Callable
import numpy as np
from rank_bm25 import BM25Okapi

from .utils import load_config, setup_logging, compute_file_hash, ensure_dir
from .loader import RepositoryLoader
from .parser import CodeParser
from .embedder import CodeEmbedder
from .indexer import CodeIndexer, CodeElement
from .vector_store import VectorStore
from .graph_builder import CodeGraphBuilder
from .symbol_resolver import SymbolResolver
from .module_resolver import ModuleResolver
from .global_index_builder import GlobalIndexBuilder
from .retriever import HybridRetriever
from .query_processor import QueryProcessor
from .answer_generator import AnswerGenerator
from .cache import CacheManager


class FastCode:
    """Main FastCode system for repository-level code understanding"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize FastCode system
        
        Args:
            config_path: Path to configuration file (default: config/config.yaml)
        """
        # Load configuration
        if config_path is None:
            # Try to find config in standard locations
            possible_paths = [
                "config/config.yaml",
                "../config/config.yaml",
                os.path.join(os.path.dirname(__file__), "../config/config.yaml"),
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    config_path = path
                    break
        
        if config_path and os.path.exists(config_path):
            self.config = load_config(config_path)
        else:
            # Use default configuration
            self.config = self._get_default_config()
        
        # Evaluation-specific overrides (keep core system decoupled)
        self.eval_config = self.config.get("evaluation", {})
        self.eval_mode = self.eval_config.get("enabled", False)
        self.in_memory_index = self.eval_config.get("in_memory_index", False)
        
        # Ensure in-memory mode disables disk-based caches/persistence
        if self.in_memory_index:
            self.config.setdefault("vector_store", {})["in_memory"] = True
            self.config.setdefault("cache", {})["enabled"] = False
        
        # Allow explicit cache disable via evaluation config
        if self.eval_config.get("disable_cache", False):
            self.config.setdefault("cache", {})["enabled"] = False
        
        # Setup logging
        self.logger = setup_logging(self.config)
        self.logger.info("Initializing FastCode system")

        # Initialize resolver attributes (will be set in index_repository)
        self.global_index_builder = None
        self.module_resolver = None
        self.symbol_resolver = None
        
        # Initialize components
        self.loader = RepositoryLoader(self.config)
        self.parser = CodeParser(self.config)
        self.embedder = CodeEmbedder(self.config)
        self.vector_store = VectorStore(self.config)
        self.indexer = CodeIndexer(self.config, self.loader, self.parser, self.embedder, self.vector_store)
        self.graph_builder = CodeGraphBuilder(self.config)
        
        # Get repo_root from config if available
        config_repo_root = self.config.get("repo_root")
        config_repo_root = os.path.abspath(config_repo_root)
        ensure_dir(config_repo_root)
        self.logger.info(f"Configured repo_root: {config_repo_root}")
        
        self.retriever = HybridRetriever(self.config, self.vector_store, 
                                         self.embedder, self.graph_builder,
                                         repo_root=config_repo_root)
        self.query_processor = QueryProcessor(self.config)
        self.answer_generator = AnswerGenerator(self.config)
        self.cache_manager = CacheManager(self.config)
        
        # State
        self.repo_loaded = False
        self.repo_indexed = False
        self.repo_info = {}
        
        # Multi-repository state
        self.multi_repo_mode = False
        self.loaded_repositories = {}  # {repo_name: repo_info}
    
    def load_repository(self, source: str, is_url: bool = True, is_zip: bool = False):
        """
        Load repository from URL, local path, or ZIP file
        
        Args:
            source: Repository URL, local path, or ZIP file path
            is_url: True if source is a URL, False if local path or ZIP
            is_zip: True if source is a ZIP file, False otherwise
        """
        self.logger.info(f"Loading repository: {source}")
        
        try:
            if is_zip:
                self.loader.load_from_zip(source)
            elif is_url:
                self.loader.load_from_url(source)
            else:
                self.loader.load_from_path(source)
            
            self.repo_loaded = True
            self.repo_info = self.loader.get_repository_info()
            
            # CRITICAL: Update config with the actual repo path.
            # This ensures path_utils can correctly normalize paths relative to the root.
            if self.loader.repo_path:
                self.config["repo_root"] = self.loader.repo_path
                self.logger.info(f"Set repo_root to: {self.loader.repo_path}")
                
                # Initialize retriever agents if agency mode is enabled
                self.retriever.set_repo_root(self.loader.repo_path)
            
            self.logger.info(f"Loaded repository: {self.repo_info.get('name')}")
            self.logger.info(f"Files: {self.repo_info.get('file_count')}, "
                           f"Size: {self.repo_info.get('total_size_mb', 0):.2f} MB")
            
        except Exception as e:
            self.logger.error(f"Failed to load repository: {e}")
            raise
    
    def index_repository(self, force: bool = False):
        """
        Index the loaded repository
        
        Args:
            force: Force re-indexing even if cache exists
        """
        # Evaluation can request forced re-indexing to respect commit checkouts
        force = force or self.eval_config.get("force_reindex", False)
        
        if not self.repo_loaded:
            raise RuntimeError("No repository loaded. Call load_repository() first.")
        
        self.logger.info("Indexing repository")
        
        repo_name = self.repo_info.get("name", "default")
        
        # Check cache
        if not force and self._should_use_cache():
            loaded = self._try_load_from_cache()
            if loaded:
                self.repo_indexed = True
                return
        
        try:
            # Get repository name for indexing
            repo_url = self.repo_info.get("url")
            
            # Index code elements with repository information
            elements = self.indexer.index_repository(repo_name=repo_name, repo_url=repo_url)
            
            # Initialize vector store if not already done
            if self.vector_store.dimension is None:
                self.vector_store.initialize(self.embedder.embedding_dim)
            
            # Add embeddings to vector store
            vectors = []
            metadata = []
            
            for elem in elements:
                embedding = elem.metadata.get("embedding")
                if embedding is not None:
                    vectors.append(embedding)
                    metadata.append(elem.to_dict())
            
            if vectors:
                vectors_array = np.array(vectors)
                self.vector_store.add_vectors(vectors_array, metadata)
            
            # Initialize resolvers for complete graph building
            # This fixes the "0 edges" issue by providing the necessary context for resolution
            try:
                self.logger.info("Initializing resolvers for precise graph building...")
                
                # Ensure repo_root is set
                repo_root = self.config.get("repo_root")
                if not repo_root and self.loader.repo_path:
                    repo_root = self.loader.repo_path
                    self.config["repo_root"] = repo_root

                # 1. Create GlobalIndexBuilder
                self.global_index_builder = GlobalIndexBuilder(self.config)

                # 2. Build global maps
                self.logger.info(f"Building global index maps (Repo Root: {repo_root})...")
                self.global_index_builder.build_maps(elements, repo_root or "")
                self.logger.info(f"  - Mapped {len(self.global_index_builder.file_map)} files")
                self.logger.info(f"  - Mapped {len(self.global_index_builder.module_map)} modules")

                # 3. Create ModuleResolver
                self.module_resolver = ModuleResolver(self.global_index_builder)

                # 4. Create SymbolResolver
                self.symbol_resolver = SymbolResolver(self.global_index_builder, self.module_resolver)
                
                self.logger.info("Resolvers initialized successfully")

            except Exception as e:
                self.logger.warning(f"Resolver initialization failed: {e}")
                self.logger.warning("Using fallback graph building (less accurate)")
                import traceback
                self.logger.error(traceback.format_exc())
                self.module_resolver = None
                self.symbol_resolver = None

            # Build code graphs with resolvers
            # This will now use the initialized resolvers to build precise graphs
            self.graph_builder.build_graphs(elements, self.module_resolver, self.symbol_resolver)
            
            # Index for BM25
            self.retriever.index_for_bm25(elements)
            
            # Build separate BM25 index for repository overviews
            self.retriever.build_repo_overview_bm25()
            
            # Save artifacts only when persistence is enabled
            if self._should_persist_indexes():
                # Save to cache with repository-specific name
                self._save_to_cache(cache_name=repo_name)

                # Save BM25 and graph data
                self.retriever.save_bm25(repo_name)
                self.graph_builder.save(repo_name)
            else:
                self.logger.info("Skipping on-disk persistence (ephemeral/evaluation mode)")

            self.repo_indexed = True
            self.logger.info(f"Repository indexing complete for {repo_name}")
            
            # Log statistics
            self._log_statistics()
            
        except Exception as e:
            self.logger.error(f"Failed to index repository: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            raise
    
    def query(self, question: str, filters: Optional[Dict[str, Any]] = None, 
              repo_filter: Optional[List[str]] = None,
              session_id: Optional[str] = None,
              enable_multi_turn: Optional[bool] = None,
              use_agency_mode: Optional[bool] = None,
              prompt_builder: Optional[Callable[[str, str, Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]], str]] = None) -> Dict[str, Any]:
        """
        Query the repository (or multiple repositories)
        
        Args:
            question: User question
            filters: Optional filters for retrieval
            repo_filter: Optional list of repository names to search in
            session_id: Optional session ID for multi-turn dialogue
            enable_multi_turn: Override config setting for multi-turn mode
            prompt_builder: Optional callable to build a custom LLM prompt using
                (question, prepared_context, query_info, dialogue_history)
        
        Returns:
            Dictionary with answer and metadata (including summary if multi-turn)
        """
        if not self.repo_indexed:
            raise RuntimeError("Repository not indexed. Call index_repository() first.")
        
        # Determine if multi-turn mode is enabled
        if enable_multi_turn is None:
            enable_multi_turn = self.config.get("generation", {}).get("enable_multi_turn", False)
        
        if repo_filter:
            self.logger.info(f"Processing query: {question} in repositories: {repo_filter}")
        else:
            self.logger.info(f"Processing query: {question}")
        
        # Get dialogue history if in multi-turn mode
        dialogue_history = []
        if enable_multi_turn and session_id:
            # Get recent summaries from cache (last 10 turns for iterative agent)
            history_summary_rounds = self.config.get("query", {}).get("history_summary_rounds", 10)
            dialogue_history = self.cache_manager.get_recent_summaries(session_id, history_summary_rounds)

            if dialogue_history:
                self.logger.info(f"Retrieved {len(dialogue_history)} previous dialogue summaries")
        
        # NOTE: Query result caching is disabled to ensure full iterative_agent flow
        # Original cache logic (disabled):
        # use_cache = (not enable_multi_turn or not session_id) and self._should_use_cache()
        # cached_result = None
        # cache_key = None
        # repo_hash = None
        # if use_cache:
        #     repo_hash = self._get_repo_hash()
        #     cache_key = f"{question}_{','.join(sorted(repo_filter)) if repo_filter else 'all'}"
        #     cached_result = self.cache_manager.get_query_result(cache_key, repo_hash)
        #     if cached_result:
        #         self.logger.info("Returning cached result")
        # result = cached_result

        result = None  # Always process through full flow
        processed_query = None
        retrieved: List[CodeElement] = []

        try:
            if result is None:
                # Determine if iterative enhancement should be used
                use_iterative_enhancement = (
                    self.retriever.enable_agency_mode
                    and self.retriever.iterative_agent is not None
                )

                # Process query: skip query_processor entirely in iterative mode
                if use_iterative_enhancement:
                    # Iterative agent will handle all query enhancement
                    # Create minimal ProcessedQuery object
                    from .query_processor import ProcessedQuery
                    processed_query = ProcessedQuery(
                        original=question,
                        expanded=question,
                        keywords=[],
                        intent="unknown",
                        subqueries=[],
                        filters=filters or {},
                        rewritten_query=None,
                        pseudocode_hints=None,
                        search_strategy=None
                    )
                    self.logger.info("Iterative mode: skipping query_processor, all enhancements handled by iterative_agent")
                else:
                    # Standard mode: use full query processing
                    processed_query = self.query_processor.process(
                        question,
                        dialogue_history,
                        use_llm_enhancement=True
                    )
                    self.logger.info(f"Query intent: {processed_query.intent}")
                    self.logger.info(f"Keywords: {processed_query.keywords}")

                # Retrieve relevant code (with repository filter and agency mode)
                # Pass ProcessedQuery object for enhanced retrieval
                # Pass dialogue_history for multi-turn context in iterative mode
                retrieved = self.retriever.retrieve(
                    processed_query,  # Pass full ProcessedQuery object for multi-repo support
                    filters=filters,
                    repo_filter=repo_filter,
                    use_agency_mode=use_agency_mode,
                    dialogue_history=dialogue_history if enable_multi_turn else None
                )
                
                # Generate answer (with dialogue history for multi-turn)
                result = self.answer_generator.generate(
                    question,
                    retrieved,
                    query_info=processed_query.to_dict(),
                    dialogue_history=self._get_full_dialogue_history(session_id, enable_multi_turn),
                    prompt_builder=prompt_builder
                )
            
            # Add repository information to result
            if repo_filter:
                result["searched_repositories"] = repo_filter
            
            # Persist dialogue for any session (even single-turn) so users keep history
            if session_id:
                turn_number = self._get_next_turn_number(session_id)
                summary = result.get("summary", "")
                # Use formatted sources from result instead of raw retrieved elements
                # This ensures proper display format when loading history
                sources = result.get("sources", [])
                # Ensure sources are fully JSON-serializable
                serializable_sources = self._ensure_jsonable_sources(sources)

                # Ensure metadata is JSON-serializable
                metadata = {
                    "intent": getattr(processed_query, "intent", None),
                    "keywords": getattr(processed_query, "keywords", None),
                    "repo_filter": repo_filter,
                    "multi_turn": enable_multi_turn,
                }
                serializable_metadata = self._ensure_jsonable_metadata(metadata)

                self.cache_manager.save_dialogue_turn(
                    session_id=session_id,
                    turn_number=turn_number,
                    query=question,
                    answer=result.get("answer", ""),
                    summary=summary,
                    retrieved_elements=serializable_sources,
                    metadata=serializable_metadata
                )

                self.logger.info(f"Saved dialogue turn {turn_number} for session {session_id}")
            
            # Cache result for stateless flows (including single-turn sessions)
            # NOTE: Query result caching is disabled to ensure full iterative_agent flow
            # if use_cache and result is not None and cache_key and repo_hash:
            #     self.cache_manager.set_query_result(cache_key, repo_hash, result)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Query failed: {e}")
            return {
                "answer": f"Error processing query: {str(e)}",
                "query": question,
                "error": str(e),
            }

    def query_stream(self, question: str, filters: Optional[Dict[str, Any]] = None,
                    repo_filter: Optional[List[str]] = None,
                    session_id: Optional[str] = None,
                    enable_multi_turn: Optional[bool] = None,
                    use_agency_mode: Optional[bool] = None,
                    prompt_builder: Optional[Callable[[str, str, Optional[Dict[str, Any]], Optional[List[Dict[str, Any]]]], str]] = None):
        """
        Query the repository with streaming response (yields answer chunks)

        Args:
            question: User question
            filters: Optional filters for retrieval
            repo_filter: Optional list of repository names to search in
            session_id: Optional session ID for multi-turn dialogue
            enable_multi_turn: Override config setting for multi-turn mode
            use_agency_mode: Override config setting for agency mode
            prompt_builder: Optional callable to build custom LLM prompt

        Yields:
            Tuples of (chunk_text or None, metadata_dict or None)
            - First yield: (None, {"status": "retrieving"})
            - After retrieval: (None, {"status": "generating", "sources": [...], ...})
            - During generation: (text_chunk, None)
            - Final yield: (None, {"status": "complete", "summary": ..., ...})
        """
        if not self.repo_indexed:
            yield None, {"error": "Repository not indexed. Call index_repository() first."}
            return

        # Determine if multi-turn mode is enabled
        if enable_multi_turn is None:
            enable_multi_turn = self.config.get("generation", {}).get("enable_multi_turn", False)

        if repo_filter:
            self.logger.info(f"Processing streaming query: {question} in repositories: {repo_filter}")
        else:
            self.logger.info(f"Processing streaming query: {question}")

        # Get dialogue history if in multi-turn mode
        dialogue_history = []
        if enable_multi_turn and session_id:
            history_summary_rounds = self.config.get("query", {}).get("history_summary_rounds", 10)
            dialogue_history = self.cache_manager.get_recent_summaries(session_id, history_summary_rounds)
            if dialogue_history:
                self.logger.info(f"Retrieved {len(dialogue_history)} previous dialogue summaries")

        try:
            # Notify start of retrieval
            yield None, {"status": "retrieving", "query": question}

            # Retrieval phase (same as query method)
            use_iterative_enhancement = (
                self.retriever.enable_agency_mode
                and self.retriever.iterative_agent is not None
            )

            if use_iterative_enhancement:
                from .query_processor import ProcessedQuery
                processed_query = ProcessedQuery(
                    original=question,
                    expanded=question,
                    keywords=[],
                    intent="unknown",
                    subqueries=[],
                    filters=filters or {},
                    rewritten_query=None,
                    pseudocode_hints=None,
                    search_strategy=None
                )
                self.logger.info("Iterative mode: skipping query_processor, all enhancements handled by iterative_agent")
            else:
                processed_query = self.query_processor.process(
                    question,
                    dialogue_history,
                    use_llm_enhancement=True
                )
                self.logger.info(f"Query intent: {processed_query.intent}")
                self.logger.info(f"Keywords: {processed_query.keywords}")

            # Retrieve relevant code
            retrieved = self.retriever.retrieve(
                processed_query,
                filters=filters,
                repo_filter=repo_filter,
                use_agency_mode=use_agency_mode,
                dialogue_history=dialogue_history if enable_multi_turn else None
            )

            # Notify start of generation
            yield None, {"status": "generating", "retrieved_count": len(retrieved)}

            # Stream answer generation
            full_answer_parts = []
            answer_metadata = {}

            for chunk, metadata in self.answer_generator.generate_stream(
                question,
                retrieved,
                query_info=processed_query.to_dict(),
                dialogue_history=self._get_full_dialogue_history(session_id, enable_multi_turn),
                prompt_builder=prompt_builder
            ):
                if chunk:
                    full_answer_parts.append(chunk)
                    yield chunk, None
                if metadata:
                    answer_metadata.update(metadata)

            # Build complete result
            full_answer = "".join(full_answer_parts)
            summary = answer_metadata.get("summary")

            result = {
                "status": "complete",
                "answer": full_answer,
                "query": question,
                "context_elements": len(retrieved),
                "sources": answer_metadata.get("sources", self._extract_sources_from_elements(retrieved)),
            }

            if summary:
                result["summary"] = summary

            if repo_filter:
                result["searched_repositories"] = repo_filter

            # Save dialogue turn if session_id provided
            if session_id:
                turn_number = self._get_next_turn_number(session_id)
                serializable_sources = self._ensure_jsonable_sources(result.get("sources", []))
                serializable_metadata = self._ensure_jsonable_metadata({
                    "intent": getattr(processed_query, "intent", None),
                    "keywords": getattr(processed_query, "keywords", None),
                    "repo_filter": repo_filter,
                    "multi_turn": enable_multi_turn,
                })

                self.cache_manager.save_dialogue_turn(
                    session_id=session_id,
                    turn_number=turn_number,
                    query=question,
                    answer=full_answer,
                    summary=summary or "",
                    retrieved_elements=serializable_sources,
                    metadata=serializable_metadata
                )

                self.logger.info(f"Saved dialogue turn {turn_number} for session {session_id}")

            # Final yield with complete result
            yield None, result

        except Exception as e:
            self.logger.error(f"Streaming query failed: {e}")
            import traceback
            error_trace = traceback.format_exc()
            self.logger.error(f"Full error traceback:\n{error_trace}")
            yield None, {
                "status": "error",
                "error": str(e),
                "query": question,
            }

    def _extract_sources_from_elements(self, elements: List) -> List[Dict[str, Any]]:
        """Extract source information from retrieved elements"""
        sources = []
        for elem_data in elements:
            elem = elem_data.get("element", {})
            source = {
                "file": elem.get("relative_path", ""),
                "repo": elem.get("repo_name", ""),
                "type": elem.get("type", ""),
                "name": elem.get("name", ""),
                "start_line": elem.get("start_line", 0),
                "end_line": elem.get("end_line", 0),
            }
            sources.append(source)
        return sources

    def get_repository_summary(self) -> str:
        """Get summary of the loaded repository"""
        if not self.repo_info:
            return "No repository loaded"
        
        summary_parts = [
            f"Repository: {self.repo_info.get('name', 'Unknown')}",
            f"Files: {self.repo_info.get('file_count', 0)}",
            f"Size: {self.repo_info.get('total_size_mb', 0):.2f} MB",
        ]
        
        if self.repo_indexed:
            summary_parts.append(f"Indexed elements: {self.vector_store.get_count()}")
        
        return "\n".join(summary_parts)
    
    def _try_load_from_cache(self) -> bool:
        """Try to load indexed data from cache for single repository"""
        if not self._should_use_cache():
            self.logger.info("Cache loading disabled (ephemeral/evaluation mode)")
            return False
        
        try:
            cache_name = self._get_cache_name()
            
            # Try to load vector store
            if self.vector_store.load(cache_name):
                self.logger.info(f"Loaded vector store from cache for {cache_name}")
                
                # Load BM25 index
                bm25_loaded = self.retriever.load_bm25(cache_name)
                if not bm25_loaded:
                    self.logger.warning("Failed to load BM25 index, will need to rebuild")
                
                # Build separate repo overview BM25 index
                self.retriever.build_repo_overview_bm25()
                
                # Load graph data
                graph_loaded = self.graph_builder.load(cache_name)
                if not graph_loaded:
                    self.logger.warning("Failed to load graph data, will need to rebuild")
                
                # If BM25 or graph failed to load, reconstruct from metadata
                if not bm25_loaded or not graph_loaded:
                    self.logger.info("Reconstructing missing components from metadata...")
                    elements = self._reconstruct_elements_from_metadata()
                    
                    if elements:
                        if not bm25_loaded:
                            self.retriever.index_for_bm25(elements)
                            self.logger.info(f"Rebuilt BM25 index with {len(elements)} elements")
                        
                        if not graph_loaded:
                            # Note: Rebuilding graph from metadata is a fallback.
                            # Precise linking might be limited if repo_root context is lost.
                            self.graph_builder.build_graphs(elements)
                            self.logger.info("Rebuilt code graph (fallback mode)")
                    else:
                        self.logger.warning("No elements reconstructed from metadata")
                
                self.logger.info("Cache loaded successfully")
                self._log_statistics()
                return True
            
            return False
            
        except Exception as e:
            self.logger.warning(f"Failed to load from cache: {e}")
            return False
    
    def _save_to_cache(self, cache_name: Optional[str] = None):
        """Save indexed data to cache"""
        if not self._should_persist_indexes():
            self.logger.info("Cache save disabled (ephemeral/evaluation mode)")
            return
        
        try:
            if cache_name is None:
                cache_name = self._get_cache_name()
            self.vector_store.save(cache_name)
            self.logger.info(f"Saved index to cache: {cache_name}")
        except Exception as e:
            self.logger.warning(f"Failed to save to cache: {e}")
    
    def _get_cache_name(self) -> str:
        """Get cache name for current repository"""
        return self.repo_info.get("name", "default")
    
    def _get_repo_hash(self) -> str:
        """Get hash of repository for cache key"""
        return self.repo_info.get("commit", self.repo_info.get("name", "default"))
    
    def _reconstruct_elements_from_metadata(self) -> List[CodeElement]:
        """
        Reconstruct CodeElement objects from vector store metadata
        Excludes repository_overview elements (they're in separate storage)
        
        Returns:
            List of CodeElement objects
        """
        elements = []
        for meta in self.vector_store.metadata:
            try:
                # Skip repository_overview elements
                if meta.get("type") == "repository_overview":
                    continue
                
                # Reconstruct CodeElement from metadata dictionary
                element = CodeElement(
                    id=meta.get("id", ""),
                    type=meta.get("type", ""),
                    name=meta.get("name", ""),
                    file_path=meta.get("file_path", ""),
                    relative_path=meta.get("relative_path", ""),
                    language=meta.get("language", ""),
                    start_line=meta.get("start_line", 0),
                    end_line=meta.get("end_line", 0),
                    code=meta.get("code", ""),
                    signature=meta.get("signature"),
                    docstring=meta.get("docstring"),
                    summary=meta.get("summary"),
                    metadata=meta.get("metadata", {}),
                    repo_name=meta.get("repo_name"),
                    repo_url=meta.get("repo_url"),
                )
                elements.append(element)
            except Exception as e:
                self.logger.warning(f"Failed to reconstruct element: {e}")
                continue
        
        self.logger.info(f"Reconstructed {len(elements)} elements from metadata (excluding repository_overview)")
        return elements
    
    def _log_statistics(self):
        """Log indexing statistics"""
        stats = {
            "vector_count": self.vector_store.get_count(),
            "graph_stats": self.graph_builder.get_graph_stats(),
        }
        
        self.logger.info(f"Statistics: {stats}")
    
    def _is_ephemeral_mode(self) -> bool:
        """Return True when running in evaluation/in-memory mode."""
        return self.in_memory_index or getattr(self.vector_store, "in_memory", False)

    def _should_use_cache(self) -> bool:
        """Determine whether cache/index reuse is allowed."""
        if self.eval_config.get("disable_cache", False):
            return False
        return not self._is_ephemeral_mode()

    def _should_persist_indexes(self) -> bool:
        """Determine whether indexes should be persisted to disk."""
        if self.eval_config.get("disable_persistence", False):
            return False
        return not self._is_ephemeral_mode()
    
    def _serialize_retrieved_elements(self, elements: List[Any]) -> List[Dict[str, Any]]:
        """Safely serialize retrieved elements for caching/session history"""
        serialized = []
        for elem in elements or []:
            try:
                if hasattr(elem, "to_dict"):
                    serialized.append(elem.to_dict())
                elif isinstance(elem, dict):
                    serialized.append(elem)
                else:
                    serialized.append({"repr": repr(elem)})
            except Exception as e:
                self.logger.warning(f"Failed to serialize element: {e}")
        return serialized
    
    def _ensure_jsonable_sources(self, sources: List[Any]) -> List[Dict[str, Any]]:
        """Ensure sources are JSON-serializable, converting any complex objects"""
        jsonable_sources = []
        for source in sources or []:
            try:
                if isinstance(source, dict):
                    # Ensure all values in dict are JSON-serializable
                    clean_source = {}
                    for key, value in source.items():
                        if isinstance(value, (str, int, float, bool, type(None))):
                            clean_source[key] = value
                        elif isinstance(value, (list, tuple)):
                            clean_source[key] = list(value)
                        elif isinstance(value, dict):
                            clean_source[key] = dict(value)
                        else:
                            clean_source[key] = str(value)
                    jsonable_sources.append(clean_source)
                else:
                    jsonable_sources.append({"repr": str(source)})
            except Exception as e:
                self.logger.warning(f"Failed to serialize source: {e}")
        return jsonable_sources

    def _ensure_jsonable_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure metadata is JSON-serializable, converting any complex objects"""
        jsonable_metadata = {}
        for key, value in (metadata or {}).items():
            try:
                if isinstance(value, (str, int, float, bool, type(None))):
                    jsonable_metadata[key] = value
                elif isinstance(value, (list, tuple)):
                    # Recursively ensure list items are serializable
                    jsonable_metadata[key] = [
                        v if isinstance(v, (str, int, float, bool, type(None))) else str(v)
                        for v in value
                    ]
                elif isinstance(value, dict):
                    # Recursively ensure dict values are serializable
                    jsonable_metadata[key] = self._ensure_jsonable_metadata(value)
                else:
                    # Convert complex objects to string
                    jsonable_metadata[key] = str(value)
            except Exception as e:
                self.logger.warning(f"Failed to serialize metadata key '{key}': {e}")
                jsonable_metadata[key] = str(value)
        return jsonable_metadata

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "repository": {
                "clone_depth": 1,
                "max_file_size_mb": 5,
                "ignore_patterns": ["*.pyc", "__pycache__", "node_modules", ".git"],
                "supported_extensions": [".py", ".js", ".ts", ".java", ".go"],
            },
            "parser": {
                "extract_docstrings": True,
                "extract_comments": True,
                "extract_imports": True,
            },
            "embedding": {
                "model": "sentence-transformers/all-MiniLM-L6-v2",
                "device": "cpu",
                "batch_size": 32,
            },
            "indexing": {
                "levels": ["file", "class", "function", "documentation"],
            },
            "vector_store": {
                "persist_directory": "./data/vector_store",
                "distance_metric": "cosine",
            },
            "retrieval": {
                "semantic_weight": 0.6,
                "keyword_weight": 0.3,
                "graph_weight": 0.1,
                "max_results": 5,
            },
            "generation": {
                "provider": "openai",
                "model": "gpt-4-turbo-preview",
                "temperature": 0.1,
                "max_tokens": 2000,
            },
            "evaluation": {
                "enabled": False,
                "in_memory_index": False,
                "disable_cache": False,
                "disable_persistence": False,
                "force_reindex": False,
            },
            "cache": {
                "enabled": True,
                "backend": "disk",
                "cache_directory": "./data/cache",
                "cache_queries": False,
            },
            "logging": {
                "level": "INFO",
                "console": True,
            },
        }
    
    def load_multiple_repositories(self, sources: List[Dict[str, Any]]):
        """
        Load and index multiple repositories (saves each repository separately)
        
        Args:
            sources: List of dictionaries with 'source', 'is_url', and optionally 'is_zip' keys
                    Example: [{'source': 'https://github.com/user/repo1', 'is_url': True},
                             {'source': '/path/to/repo2', 'is_url': False},
                             {'source': '/path/to/repo3.zip', 'is_url': False, 'is_zip': True}]
        """
        self.logger.info(f"Loading {len(sources)} repositories")
        self.multi_repo_mode = True
        
        successfully_indexed = []
        
        for i, source_info in enumerate(sources):
            source = source_info.get('source')
            is_url = source_info.get('is_url', True)
            is_zip = source_info.get('is_zip', False)
            
            try:
                self.logger.info(f"[{i+1}/{len(sources)}] Loading repository: {source}")
                
                # Load repository
                if is_zip:
                    self.loader.load_from_zip(source)
                elif is_url:
                    self.loader.load_from_url(source)
                else:
                    self.loader.load_from_path(source)
                
                repo_info = self.loader.get_repository_info()
                repo_name = repo_info.get('name')
                repo_url = repo_info.get('url', source)
                
                # Update config with repo_root for each repo (Critical for graph building)
                if self.loader.repo_path:
                    self.config["repo_root"] = self.loader.repo_path
                
                # Store repository info
                self.loaded_repositories[repo_name] = repo_info
                
                self.logger.info(f"Indexing repository: {repo_name}")
                
                # Create a fresh vector store for this repository
                temp_vector_store = VectorStore(self.config)
                temp_vector_store.initialize(self.embedder.embedding_dim)
                
                # Create a temporary indexer with the temp vector store for this repo
                temp_indexer = CodeIndexer(self.config, self.loader, self.parser, 
                                          self.embedder, temp_vector_store)
                
                # Index with repository information
                elements = temp_indexer.index_repository(repo_name=repo_name, repo_url=repo_url)
                
                # Add to temporary vector store
                vectors = []
                metadata = []
                
                for elem in elements:
                    embedding = elem.metadata.get("embedding")
                    if embedding is not None:
                        vectors.append(embedding)
                        metadata.append(elem.to_dict())
                
                if vectors:
                    vectors_array = np.array(vectors)
                    temp_vector_store.add_vectors(vectors_array, metadata)
                    
                    # Save this repository's vector index separately
                    temp_vector_store.save(repo_name)
                    
                    # Build and save BM25 index for this repository
                    temp_retriever = HybridRetriever(self.config, temp_vector_store, 
                                                     self.embedder, self.graph_builder,
                                                     repo_root=self.loader.repo_path)
                    temp_retriever.index_for_bm25(elements)
                    temp_retriever.save_bm25(repo_name)
                    self.logger.info(f"Saved BM25 index for {repo_name}")
                    
                    # Build separate BM25 index for repository overviews
                    temp_retriever.build_repo_overview_bm25()
                    self.logger.info(f"Built repo overview BM25 index")
                    
                    # Build and save graph for this repository (Using temporary graph builder)
                    # We need a fresh graph builder to avoid mixing graphs between repos during this loop
                    # unless we want to support cross-repo graphs immediately
                    temp_graph_builder = CodeGraphBuilder(self.config)
                    
                    # Initialize resolvers for precise graph building
                    repo_root = self.loader.repo_path
                    temp_module_resolver = None
                    temp_symbol_resolver = None
                    
                    try:
                        self.logger.info(f"Initializing resolvers for {repo_name}...")
                        temp_global_index = GlobalIndexBuilder(self.config)
                        temp_global_index.build_maps(elements, repo_root)
                        temp_module_resolver = ModuleResolver(temp_global_index)
                        temp_symbol_resolver = SymbolResolver(temp_global_index, temp_module_resolver)
                        self.logger.info(f"Resolvers initialized for {repo_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to initialize resolvers for {repo_name}: {e}")
                        temp_module_resolver = None
                        temp_symbol_resolver = None

                    temp_graph_builder.build_graphs(elements, temp_module_resolver, temp_symbol_resolver)
                    temp_graph_builder.save(repo_name)
                    self.logger.info(f"Saved graph data for {repo_name}")
                    
                    successfully_indexed.append(repo_name)
                    
                    self.logger.info(f"Successfully indexed and saved {repo_name}: {len(elements)} elements")
                else:
                    self.logger.warning(f"No vectors generated for {repo_name}")
                
            except Exception as e:
                self.logger.error(f"Failed to load repository {source}: {e}")
                import traceback
                self.logger.error(traceback.format_exc())
                # Continue with next repository
                continue
        
        if successfully_indexed:
            self.logger.info(f"Successfully indexed {len(successfully_indexed)} repositories:")
            for repo_name in successfully_indexed:
                self.logger.info(f"  - {repo_name}")
            
            # Merge all indexed repositories into the main vector store for statistics
            self.logger.info("Merging repositories into main vector store for statistics...")
            if self.vector_store.dimension is None:
                self.vector_store.initialize(self.embedder.embedding_dim)
            
            for repo_name in successfully_indexed:
                if self.vector_store.merge_from_index(repo_name):
                    self.logger.info(f"Merged {repo_name} into main store")
                else:
                    self.logger.warning(f"Failed to merge {repo_name}")
        else:
            self.logger.error("No repositories were successfully indexed")
        
        self.repo_indexed = len(successfully_indexed) > 0
        self.repo_loaded = len(successfully_indexed) > 0
        
        self.logger.info(f"Indexing complete. Each repository saved separately.")
    
    def list_repositories(self) -> List[Dict[str, Any]]:
        """
        List all indexed repositories
        
        Returns:
            List of repository information dictionaries
        """
        repo_names = self.vector_store.get_repository_names()
        repo_counts = self.vector_store.get_count_by_repository()
        
        repositories = []
        for repo_name in repo_names:
            repo_info = self.loaded_repositories.get(repo_name, {})
            repositories.append({
                'name': repo_name,
                'element_count': repo_counts.get(repo_name, 0),
                'file_count': repo_info.get('file_count', 0),
                'size_mb': repo_info.get('total_size_mb', 0),
                'url': repo_info.get('url', 'N/A'),
            })
        
        return repositories
    
    def get_repository_stats(self) -> Dict[str, Any]:
        """
        Get statistics about all indexed repositories
        
        Returns:
            Dictionary with repository statistics
        """
        repo_counts = self.vector_store.get_count_by_repository()
        repo_names = self.vector_store.get_repository_names()
        
        stats = {
            'total_repositories': len(repo_names),
            'total_elements': self.vector_store.get_count(),
            'repositories': []
        }
        
        for repo_name in repo_names:
            repo_info = self.loaded_repositories.get(repo_name, {})
            stats['repositories'].append({
                'name': repo_name,
                'elements': repo_counts.get(repo_name, 0),
                'files': repo_info.get('file_count', 0),
                'size_mb': repo_info.get('total_size_mb', 0),
            })
        
        return stats
    
    def _load_multi_repo_cache(self, repo_names: Optional[List[str]] = None) -> bool:
        """
        Load multi-repository index from cache by merging individual repository indices
        
        Args:
            repo_names: Optional list of specific repository names to load.
                       If None, loads all available repositories.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Discover available repository indexes
            persist_dir = self.vector_store.persist_dir
            available_repos = []
            
            if os.path.exists(persist_dir):
                for file in os.listdir(persist_dir):
                    if file.endswith('.faiss'):
                        repo_name = file.replace('.faiss', '')
                        metadata_file = os.path.join(persist_dir, f"{repo_name}_metadata.pkl")
                        if os.path.exists(metadata_file):
                            available_repos.append(repo_name)
            
            if not available_repos:
                self.logger.error("No repository indexes found")
                return False
            
            # Filter repositories if specific ones are requested
            if repo_names:
                repos_to_load = [r for r in available_repos if r in repo_names]
                if not repos_to_load:
                    self.logger.error(f"None of the requested repositories found: {repo_names}")
                    return False
            else:
                repos_to_load = available_repos
            
            self.logger.info(f"Found {len(repos_to_load)} repository indexes: {', '.join(repos_to_load)}")
            
            # Initialize vector store if needed
            if self.vector_store.dimension is None:
                self.vector_store.initialize(self.embedder.embedding_dim)
            
            # Load each repository index and merge them
            for repo_name in repos_to_load:
                self.logger.info(f"Loading index for {repo_name}...")
                try:
                    # Merge this repository's index into the main vector store
                    if self.vector_store.merge_from_index(repo_name):
                        self.logger.info(f"Successfully merged {repo_name}")
                    else:
                        self.logger.warning(f"Failed to merge index for {repo_name}")
                        
                except Exception as e:
                    self.logger.error(f"Error loading {repo_name}: {e}")
                    continue
            
            # Check if we successfully loaded any repositories
            if self.vector_store.get_count() == 0:
                self.logger.error("Failed to load any repository indexes")
                return False
            
            # Register loaded repositories
            # We know which repos were successfully loaded from repos_to_load
            for repo_name in repos_to_load:
                if repo_name not in self.loaded_repositories:
                    self.loaded_repositories[repo_name] = {
                        "name": repo_name,
                        "file_count": 0,  # Will be updated if needed
                        "total_size_mb": 0,
                    }
            
            # Try to load BM25 and graph data from saved files
            # For multi-repo, we merge BM25 data from all loaded repositories
            self.logger.info("Loading BM25 and graph data...")
            
            all_bm25_elements = []
            all_bm25_corpus = []
            graphs_loaded = False
            
            for repo_name in repos_to_load:
                # Try loading BM25 for each repo
                bm25_path = os.path.join(self.retriever.persist_dir, f"{repo_name}_bm25.pkl")
                if os.path.exists(bm25_path):
                    try:
                        with open(bm25_path, 'rb') as f:
                            data = pickle.load(f)
                            all_bm25_corpus.extend(data["bm25_corpus"])
                            
                            # Reconstruct CodeElement objects
                            for elem_dict in data["bm25_elements"]:
                                all_bm25_elements.append(CodeElement(**elem_dict))
                        
                        self.logger.info(f"Loaded BM25 data for {repo_name}")
                    except Exception as e:
                        self.logger.warning(f"Failed to load BM25 data for {repo_name}: {e}")
                
                # Load graph data (merge into main graph)
                if not graphs_loaded:
                    # Load the first repository's graph as base
                    if self.graph_builder.load(repo_name):
                        graphs_loaded = True
                        self.logger.info(f"Loaded graph data from {repo_name} as base")
                else:
                    # Merge additional repository graphs
                    if self.graph_builder.merge_from_file(repo_name):
                        self.logger.info(f"Merged graph data from {repo_name}")
                    else:
                        self.logger.warning(f"Failed to merge graph data from {repo_name}")
                    # TODO: Merge additional repository graphs if needed
            # Rebuild FULL BM25 index with merged data (for repository selection)
            if all_bm25_elements and all_bm25_corpus:
                self.retriever.full_bm25_elements = all_bm25_elements
                self.retriever.full_bm25_corpus = all_bm25_corpus
                self.retriever.full_bm25 = BM25Okapi(all_bm25_corpus)
                self.logger.info(f"Rebuilt full BM25 index with {len(all_bm25_elements)} merged elements")
            else:
                # Fallback: reconstruct from metadata
                self.logger.info("No BM25 data found, reconstructing from metadata...")
                elements = self._reconstruct_elements_from_metadata()
                
                if elements:
                    self.retriever.index_for_bm25(elements)
                    self.logger.info(f"Rebuilt BM25 index with {len(elements)} elements")
                    
                    if not graphs_loaded:
                        self.graph_builder.build_graphs(elements)
                        self.logger.info("Rebuilt code graph")
                else:
                    self.logger.warning("No elements reconstructed from metadata")
            
            # Build separate BM25 index for repository overviews
            self.retriever.build_repo_overview_bm25()
            self.logger.info("Built separate BM25 index for repository overviews")
            
            self.multi_repo_mode = True
            self.repo_indexed = True
            self.repo_loaded = True
            
            self.logger.info(f"Successfully loaded {len(repos_to_load)} repositories with {self.vector_store.get_count()} total vectors")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load multi-repo cache: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            return False
    
    def cleanup(self):
        """Cleanup resources"""
        self.loader.cleanup()
        self.logger.info("Cleanup complete")
    
    def _get_full_dialogue_history(self, session_id: Optional[str], enable_multi_turn: bool) -> Optional[List[Dict[str, Any]]]:
        """
        Get full dialogue history for answer generation
        
        Args:
            session_id: Session ID
            enable_multi_turn: Whether multi-turn is enabled
        
        Returns:
            List of dialogue turns or None
        """
        if not enable_multi_turn or not session_id:
            return None

        context_rounds = self.config.get("generation", {}).get("context_rounds", 10)
        history = self.cache_manager.get_dialogue_history(session_id, max_turns=context_rounds)

        return history if history else None
    
    def _get_next_turn_number(self, session_id: str) -> int:
        """
        Get the next turn number for a session
        
        Args:
            session_id: Session ID
        
        Returns:
            Next turn number (1-indexed)
        """
        session_index = self.cache_manager._get_session_index(session_id)
        if session_index:
            return session_index.get("total_turns", 0) + 1
        else:
            return 1
    
    def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """
        Get dialogue history for a session
        
        Args:
            session_id: Session ID
        
        Returns:
            List of dialogue turns
        """
        return self.cache_manager.get_dialogue_history(session_id)
    
    def delete_session(self, session_id: str) -> bool:
        """
        Delete a dialogue session
        
        Args:
            session_id: Session ID
        
        Returns:
            True if successful
        """
        return self.cache_manager.delete_session(session_id)
    
    def remove_repository(self, repo_name: str, delete_source: bool = True) -> Dict[str, Any]:
        """
        Fully remove a repository: vector index files, BM25, graphs,
        repo overview, and optionally the cloned source code.

        Args:
            repo_name: Name of the repository to remove
            delete_source: If True, also remove ./repos/<repo_name>

        Returns:
            Dict with deleted files and freed bytes
        """
        import shutil

        persist_dir = self.vector_store.persist_dir
        deleted_files = []
        freed_bytes = 0

        # Files to delete from vector_store directory
        file_patterns = [
            f"{repo_name}.faiss",
            f"{repo_name}_metadata.pkl",
            f"{repo_name}_bm25.pkl",
            f"{repo_name}_graphs.pkl",
        ]

        for fname in file_patterns:
            fpath = os.path.join(persist_dir, fname)
            if os.path.exists(fpath):
                size = os.path.getsize(fpath)
                os.remove(fpath)
                deleted_files.append(fname)
                freed_bytes += size
                self.logger.info(f"Deleted {fpath} ({size / (1024*1024):.2f} MB)")

        # Remove overview entry from repo_overviews.pkl
        if self.vector_store.delete_repo_overview(repo_name):
            deleted_files.append("repo_overviews.pkl (entry)")
            self.logger.info(f"Deleted overview entry for {repo_name}")

        # Remove cloned source code
        if delete_source:
            repo_root = self.config.get("repo_root", "./repos")
            repo_dir = os.path.join(repo_root, repo_name)
            if os.path.isdir(repo_dir):
                dir_size = sum(
                    os.path.getsize(os.path.join(dp, f))
                    for dp, _, fns in os.walk(repo_dir)
                    for f in fns
                )
                shutil.rmtree(repo_dir)
                deleted_files.append(f"repos/{repo_name}/")
                freed_bytes += dir_size
                self.logger.info(f"Deleted source directory {repo_dir}")

        # Invalidate scan cache
        self.vector_store.invalidate_scan_cache()

        return {
            "repo_name": repo_name,
            "deleted_files": deleted_files,
            "freed_bytes": freed_bytes,
            "freed_mb": round(freed_bytes / (1024 * 1024), 2),
        }

    def list_sessions(self) -> List[Dict[str, Any]]:
        """
        List all dialogue sessions with enriched metadata

        Returns:
            List of session metadata with first query as title
        """
        sessions = self.cache_manager.list_sessions()

        # Enrich each session with the first query as a title
        enriched_sessions = []
        for session in sessions:
            session_id = session.get("session_id", "")
            if session_id:
                # Get the first turn to use its query as title
                first_turn = self.cache_manager.get_dialogue_turn(session_id, 1)
                if first_turn:
                    first_query = first_turn.get("query", "")
                    # Truncate long queries
                    if len(first_query) > 80:
                        title = first_query[:77] + "..."
                    else:
                        title = first_query
                    session["title"] = title
                else:
                    session["title"] = f"Session {session_id}"
            else:
                session["title"] = "Unknown Session"

            enriched_sessions.append(session)

        return enriched_sessions
