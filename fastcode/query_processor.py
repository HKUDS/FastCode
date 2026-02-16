"""
Query Processor - Process and enhance user queries with LLM-based understanding
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import os
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv

from .llm_utils import openai_chat_completion


@dataclass
class ProcessedQuery:
    """Processed query with extracted information"""
    original: str
    expanded: str
    keywords: List[str]
    intent: str  # 'how', 'what', 'where', 'debug', 'explain', 'find', 'implement'
    subqueries: List[str]
    filters: Dict[str, Any]
    rewritten_query: Optional[str] = None  # LLM-rewritten query for semantic search
    # repo_matching_terms: Optional[List[str]] = None  # Terms specifically for matching repository overviews/summaries
    pseudocode_hints: Optional[str] = None  # Pseudocode for implementation queries
    search_strategy: Optional[str] = None  # Recommended search strategy
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "original": self.original,
            "expanded": self.expanded,
            "keywords": self.keywords,
            "intent": self.intent,
            "subqueries": self.subqueries,
            "filters": self.filters,
            "rewritten_query": self.rewritten_query,
            # "repo_matching_terms": self.repo_matching_terms,
            "pseudocode_hints": self.pseudocode_hints,
            "search_strategy": self.search_strategy,
        }


class QueryProcessor:
    """Process user queries to improve retrieval with LLM-based enhancement"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.query_config = config.get("query", {})
        self.gen_config = config.get("generation", {})
        self.logger = logging.getLogger(__name__)
        
        self.expand_query = self.query_config.get("expand_query", True)
        self.decompose_complex = self.query_config.get("decompose_complex", True)
        self.max_subqueries = self.query_config.get("max_subqueries", 3)
        self.extract_keywords = self.query_config.get("extract_keywords", True)
        self.detect_intent = self.query_config.get("detect_intent", True)
        
        # NEW: LLM-based enhancement settings
        self.use_llm_enhancement = self.query_config.get("use_llm_enhancement", True)
        self.llm_enhancement_mode = self.query_config.get("llm_enhancement_mode", "adaptive")  # adaptive, always, off

        # Multi-turn dialogue settings
        self.history_summary_rounds = self.query_config.get("history_summary_rounds", 10)
        self.max_summary_words = self.query_config.get("max_summary_words", 250)

        # LLM settings
        load_dotenv()
        self.provider = self.gen_config.get("provider", "openai")
        self.model = os.getenv("MODEL")
        self.temperature = 0.3  # Slightly higher for creative query expansion
        self.max_tokens = 2000  # Shorter responses for query processing
        
        # Initialize LLM client
        if self.use_llm_enhancement:
            self.api_key = os.getenv("OPENAI_API_KEY")
            self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
            self.base_url = os.getenv("BASE_URL")
            self.llm_client = self._initialize_llm_client()
        else:
            self.llm_client = None
        
        # Intent keywords
        self.intent_patterns = {
            "how": ["how", "implement", "create", "build", "make"],
            "what": ["what", "is", "are", "does", "define", "purpose"],
            "where": ["where", "locate", "find", "which file"],
            "debug": ["error", "bug", "issue", "problem", "fix", "why not", "doesn't work"],
            "explain": ["explain", "describe", "tell me about", "understand"],
            "find": ["find", "search", "locate", "show me", "list"],
            "implement": ["implement", "write", "code", "develop", "algorithm"],
        }
        
        # Code-related keywords
        self.code_keywords = {
            "function", "method", "class", "module", "variable", "parameter",
            "return", "import", "export", "api", "endpoint", "route",
            "database", "query", "model", "schema", "table",
            "authentication", "auth", "login", "user", "session",
            "test", "unittest", "spec", "testing",
        }
    
    def _initialize_llm_client(self):
        """Initialize LLM client for query enhancement"""
        try:
            if self.provider == "openai":
                api_key = self.api_key
                if not api_key:
                    self.logger.warning("OPENAI_API_KEY not set, LLM enhancement disabled")
                    return None
                return OpenAI(api_key=api_key, base_url=self.base_url)
            
            elif self.provider == "anthropic":
                api_key = self.anthropic_api_key
                if not api_key:
                    self.logger.warning("ANTHROPIC_API_KEY not set, LLM enhancement disabled")
                    return None
                return Anthropic(api_key=api_key, base_url=self.base_url)
            
            else:
                self.logger.warning(f"Unknown provider: {self.provider}, LLM enhancement disabled")
                return None
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM client: {e}, LLM enhancement disabled")
            return None
    
    def process(
        self,
        query: str,
        dialogue_history: Optional[List[Dict[str, Any]]] = None,
        use_llm_enhancement: Optional[bool] = None
    ) -> ProcessedQuery:
        """
        Process user query with LLM-based enhancement

        Args:
            query: User query string
            dialogue_history: Previous dialogue summaries for multi-turn mode
            use_llm_enhancement: Override LLM enhancement per call (None = default)
                                If False, iterative agent will handle all enhancements

        Returns:
            ProcessedQuery object with enhanced information
        """
        self.logger.info(f"Processing query: {query}")

        # Clean query
        query = query.strip()

        # Determine if we should use LLM enhancement
        should_use_llm = self.use_llm_enhancement if use_llm_enhancement is None else use_llm_enhancement

        # Multi-turn mode: reference resolution and query rewriting
        # IMPORTANT: Skip this if iterative agent is enabled (use_llm_enhancement=False)
        # because iterative agent will handle dialogue history integration
        if dialogue_history and len(dialogue_history) > 0 and should_use_llm:
            self.logger.info("Multi-turn mode: performing reference resolution and query rewriting")
            query = self._resolve_references_and_rewrite(query, dialogue_history)
            self.logger.info(f"Rewritten query: {query}")
        elif dialogue_history and len(dialogue_history) > 0 and not should_use_llm:
            self.logger.info("Multi-turn mode with iterative agent: skipping reference resolution (will be handled by iterative agent)")

        # Detect intent (rule-based first)
        intent = self._detect_intent(query) if self.detect_intent else "general"

        # Extract keywords
        keywords = self._extract_keywords(query) if self.extract_keywords else []

        # Extract filters
        filters = self._extract_filters(query)

        # Expand query (basic rule-based)
        expanded = self._expand_query(query) if self.expand_query else query

        # Decompose complex queries
        subqueries = self._decompose_query(query) if self.decompose_complex else []

        # NEW: LLM-based enhancement
        rewritten_query = None
        # repo_matching_terms = None
        pseudocode_hints = None
        search_strategy = None

        if should_use_llm and self._should_use_llm_enhancement(query, intent):
            try:
                llm_enhancements = self._enhance_with_llm(query, intent, keywords, filters)
                rewritten_query = llm_enhancements.get("rewritten_query")
                # repo_matching_terms = llm_enhancements.get("repo_matching_terms") if llm_enhancements.get("repo_matching_terms") else []
                pseudocode_hints = llm_enhancements.get("pseudocode_hints")
                search_strategy = llm_enhancements.get("search_strategy")

                self.logger.info(f"rewritten_query: {rewritten_query}")
                # self.logger.info(f"Repo matching terms: {repo_matching_terms}")
                self.logger.info(f"pseudocode_hints: {pseudocode_hints}")
                self.logger.info(f"search_strategy: {search_strategy}")
                self.logger.info(f"refined_intent: {llm_enhancements.get('refined_intent')}")
                self.logger.info(f"selected_keywords: {llm_enhancements.get('selected_keywords')}")
                self.logger.info(f"query: {query}")
                self.logger.info(f"keywords: {keywords}")
                
                # Refine intent if LLM provides better classification
                if llm_enhancements.get("refined_intent"):
                    intent = llm_enhancements["refined_intent"]
                
                # Enhance keywords with LLM-suggested terms
                selected_keywords = llm_enhancements["selected_keywords"]
                if selected_keywords:
                    if len(selected_keywords) > 2:
                        keywords = selected_keywords
                    else:
                        keywords.extend(selected_keywords)
                    # Remove duplicates while preserving order
                    seen = set()
                    keywords = [k for k in keywords if not (k in seen or seen.add(k))]
                
                self.logger.info(f"LLM enhancement applied for query: {query[:50]}...")
            except Exception as e:
                self.logger.warning(f"LLM enhancement failed, using rule-based only: {e}")
        
        return ProcessedQuery(
            original=query,
            expanded=expanded,
            keywords=keywords,
            intent=intent,
            subqueries=subqueries,
            filters=filters,
            rewritten_query=rewritten_query,
            # repo_matching_terms=repo_matching_terms,
            pseudocode_hints=pseudocode_hints,
            search_strategy=search_strategy,
        )
    
    def _detect_intent(self, query: str) -> str:
        """Detect query intent"""
        query_lower = query.lower()
        
        # Check for each intent
        intent_scores = {}
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if pattern in query_lower)
            intent_scores[intent] = score
        
        # Get intent with highest score
        if intent_scores:
            max_score = max(intent_scores.values())
            if max_score > 0:
                for intent, score in intent_scores.items():
                    if score == max_score:
                        self.logger.debug(f"Detected intent: {intent}")
                        return intent
        
        return "general"
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract important keywords from query"""
        # Remove common words
        stopwords = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
            "been", "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "this", "that",
            "these", "those", "i", "you", "he", "she", "it", "we", "they",
        }
        
        # Tokenize and filter
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [w for w in words if w not in stopwords and len(w) > 2]
        
        # Prioritize code-related keywords
        prioritized = [k for k in keywords if k in self.code_keywords]
        other = [k for k in keywords if k not in self.code_keywords]
        
        return prioritized + other
    
    def _extract_filters(self, query: str) -> Dict[str, Any]:
        """Extract filters from query (file type, language, etc.)"""
        filters = {}
        
        # Extract file types
        file_type_pattern = r'\.(py|js|ts|java|go|cpp|c|rs|rb|php|cs)\b'
        file_types = re.findall(file_type_pattern, query.lower())
        if file_types:
            filters["extension"] = f".{file_types[0]}"
        
        # Extract language mentions with STRICT context-aware patterns
        # Only match when language name appears in specific contexts to avoid false positives
        # This prevents matching repo names like "Django", "requests", "go-kit", etc.
        language_context_patterns = {
            "python": [
                r'\bin\s+python\b',  # "in Python"
                r'\bpython\s+(code|implementation|script|function|class|module)\b',
                r'\busing\s+python\b',
                r'\bwritten\s+in\s+python\b',
                r'\bpython\s+implement',  # "Python implements"
            ],
            "javascript": [
                r'\bin\s+javascript\b',
                r'\bjavascript\s+(code|implementation|function|class|module)\b',
                r'\busing\s+javascript\b',
                r'\bin\s+js\b',
                r'\bjs\s+(code|implementation)\b',
            ],
            "typescript": [
                r'\bin\s+typescript\b',
                r'\btypescript\s+(code|implementation|function|class|module)\b',
                r'\busing\s+typescript\b',
                r'\bts\s+(code|implementation)\b',  # "ts code"
            ],
            "java": [
                r'\bin\s+java\b',
                r'\bjava\s+(code|implementation|class|method)\b',
                r'\busing\s+java\b',
            ],
            "go": [
                r'\bin\s+go\b',
                r'\bgo\s+(code|implementation|function|package)\b',
                r'\busing\s+go\b',
                r'\bin\s+golang\b',
                r'\bgolang\s+(code|implementation)\b',
            ],
            "cpp": [
                r'\bin\s+c\+\+\b',
                r'\bc\+\+\s+(code|implementation|class|function)\b',
                r'\busing\s+c\+\+\b',
            ],
            "rust": [
                r'\bin\s+rust\b',
                r'\brust\s+(code|implementation|function|module)\b',
                r'\busing\s+rust\b',
            ],
        }
        
        query_lower = query.lower()
        for lang, patterns in language_context_patterns.items():
            if any(re.search(pattern, query_lower) for pattern in patterns):
                filters["language"] = lang
                break
        
        # Extract file path mentions
        # Look for quoted strings that might be paths
        path_pattern = r'["\']([a-zA-Z0-9_/\.-]+)["\']'
        paths = re.findall(path_pattern, query)
        if paths:
            filters["file_path"] = paths[0]
        
        return filters
    
    def _expand_query(self, query: str) -> str:
        """Expand query with synonyms and related terms"""
        # Simple expansion with common synonyms
        expansions = {
            "function": ["function", "method", "procedure", "routine"],
            "method": ["method", "function", "member function"],
            "class": ["class", "object", "type"],
            "error": ["error", "exception", "bug", "issue"],
            "api": ["api", "endpoint", "route", "service"],
            "database": ["database", "db", "storage", "persistence"],
            "auth": ["authentication", "authorization", "auth", "login"],
            "test": ["test", "unittest", "spec", "testing"],
        }
        
        expanded_terms = []
        words = query.lower().split()
        
        for word in words:
            if word in expansions:
                # Add original and expansions
                expanded_terms.extend(expansions[word])
            else:
                expanded_terms.append(word)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_terms = []
        for term in expanded_terms:
            if term not in seen:
                seen.add(term)
                unique_terms.append(term)
        
        expanded = " ".join(unique_terms)
        
        if expanded != query.lower():
            self.logger.debug(f"Expanded query: {expanded}")
        
        return expanded
    
    def _decompose_query(self, query: str) -> List[str]:
        """Decompose complex queries into sub-queries"""
        # Check if query is complex (contains multiple clauses)
        if len(query.split()) < 10:
            return []  # Too short to decompose
        
        subqueries = []
        
        # Split by common separators
        separators = [" and ", " or ", ", ", "; "]
        parts = [query]
        
        for sep in separators:
            new_parts = []
            for part in parts:
                new_parts.extend(part.split(sep))
            parts = new_parts
        
        # Filter and clean subqueries
        for part in parts:
            part = part.strip()
            if len(part) > 15 and part != query:  # Must be substantial
                subqueries.append(part)
        
        # Limit number of subqueries
        subqueries = subqueries[:self.max_subqueries]
        
        if subqueries:
            self.logger.debug(f"Decomposed into {len(subqueries)} sub-queries")
        
        return subqueries
    
    def is_code_query(self, query: str) -> bool:
        """Check if query is asking about code"""
        query_lower = query.lower()
        
        # Check for code-related keywords
        code_indicators = [
            "function", "class", "method", "variable", "code",
            "implementation", "how to", "algorithm", "logic",
            "file", "module", "import", "api", "endpoint",
        ]
        
        return any(indicator in query_lower for indicator in code_indicators)
    
    def extract_code_entity(self, query: str) -> Optional[Tuple[str, str]]:
        """
        Extract code entity mention from query
        
        Returns:
            Tuple of (entity_type, entity_name) or None
        """
        # Pattern: "function named X", "class X", "X function", etc.
        patterns = [
            (r'function\s+(?:named\s+)?[\"\']?(\w+)[\"\']?', 'function'),
            (r'class\s+(?:named\s+)?[\"\']?(\w+)[\"\']?', 'class'),
            (r'method\s+(?:named\s+)?[\"\']?(\w+)[\"\']?', 'function'),
            (r'[\"\'](\w+)[\"\']?\s+function', 'function'),
            (r'[\"\'](\w+)[\"\']?\s+class', 'class'),
        ]
        
        query_lower = query.lower()
        
        for pattern, entity_type in patterns:
            match = re.search(pattern, query_lower)
            if match:
                entity_name = match.group(1)
                return (entity_type, entity_name)
        
        return None
    
    def _should_use_llm_enhancement(self, query: str, intent: str) -> bool:
        """
        Determine if LLM enhancement should be used for this query
        
        Args:
            query: The user query
            intent: Detected intent
        
        Returns:
            True if LLM enhancement should be applied
        """
        if not self.use_llm_enhancement or self.llm_client is None:
            return False
        
        # Mode: always use LLM
        if self.llm_enhancement_mode == "always":
            return True
        
        # Mode: never use LLM
        if self.llm_enhancement_mode == "off":
            return False
        
        # Mode: adaptive - use LLM for complex or implementation queries
        # Use LLM for:
        # 1. Implementation/how-to queries (benefit from pseudocode)
        # 2. Complex queries (multiple clauses or technical depth)
        # 3. Ambiguous queries (short and vague)
        
        query_lower = query.lower()
        
        # Check for implementation intent
        implementation_indicators = [
            "implement", "how to", "write", "create", "build", "develop",
            "algorithm", "logic", "code for", "function that", "method that"
        ]
        is_implementation = any(ind in query_lower for ind in implementation_indicators)
        
        # Check for complexity (multiple technical terms)
        tech_term_count = sum(1 for keyword in self.code_keywords if keyword in query_lower)
        is_complex = tech_term_count >= 3 or len(query.split()) >= 15
        
        # Check for ambiguity (short and vague)
        is_ambiguous = len(query.split()) <= 5 and tech_term_count <= 1
        
        should_enhance = is_implementation or is_complex or is_ambiguous
        
        if should_enhance:
            self.logger.debug(f"LLM enhancement enabled: impl={is_implementation}, "
                            f"complex={is_complex}, ambiguous={is_ambiguous}")
        
        return should_enhance
    
    def _enhance_with_llm(self, query: str, intent: str, 
                         keywords: List[str], filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use LLM to enhance query understanding and expansion
        
        Args:
            query: Original user query
            intent: Detected intent
            keywords: Extracted keywords
            filters: Extracted filters
        
        Returns:
            Dictionary with enhancement information
        """
        prompt = self._build_enhancement_prompt(query, intent, keywords, filters)
        
        try:
            if self.provider == "openai":
                response = self._call_openai(prompt)
            elif self.provider == "anthropic":
                response = self._call_anthropic(prompt)
            else:
                return {}

            print(f"LLM response of _enhance_with_llm: {response}")
            
            # Parse LLM response
            enhancements = self._parse_llm_response(response, intent)
            return enhancements
            
        except Exception as e:
            self.logger.error(f"LLM enhancement error: {e}")
            return {}
    
    def _build_enhancement_prompt(self, query: str, intent: str,
                                  keywords: List[str], filters: Dict[str, Any]) -> str:
        """Build prompt for LLM query enhancement"""
        
        # Standard mode (backward compatible)
        prompt = f"""You are a code search query analyzer. Analyze this query to help retrieve relevant code. 

User Query: "{query}"
Extracted Keywords: {', '.join(keywords) if keywords else 'None'}
Filters: {filters if filters else 'None'}

CRITICAL INSTRUCTION: Regardless of the language used in the above content, **ALL your output fields below MUST be generated in ENGLISH.**

Please provide:
1. REFINED_INTENT: Task type classification from the following options:
   - Code QA: General questions about existing code
   - Document QA: Questions about documentation, README, or comments
   - API Usage: How to use a specific API or library
   - Bug Fixing: Debugging or fixing errors
   - Feature Addition: Adding new functionality or refactoring
   - Architecture: Repository-level architecture or design questions
   - Cross-repo: Cross-repository linking or library comparison

2. REWRITTEN_QUERY: Rewrite the query for semantic search to better match indexed repositories (focus on technical terms and code concepts)

3. SELECTED_KEYWORDS: Important code-related keywords for BM25 search (comma-separated, You can include additional important keywords)

4. PSEUDOCODE_HINTS: If implementation-related, provide pseudocode structure (3-7 lines, or "N/A")

Format your response EXACTLY as follows (strictly follow this format):
REFINED_INTENT: <intent>
REWRITTEN_QUERY: <rewritten query on a SINGLE line>
SELECTED_KEYWORDS: <keyword1, keyword2, keyword3>
PSEUDOCODE_HINTS: <pseudocode or "N/A">

IMPORTANT FORMATTING RULES:
- Each field must start on a new line
- REWRITTEN_QUERY must be on a SINGLE line (do not use line breaks within the query)
- SELECTED_KEYWORDS must be comma-separated on a SINGLE line
- Do not use markdown formatting (no **, *, or ` characters) in field values
- For PSEUDOCODE_HINTS, you can use multiple lines but do not wrap in code blocks (no ```)

Be concise and focus on improving code retrieval accuracy."""
        
        return prompt
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API for query enhancement"""
        response = openai_chat_completion(
            self.llm_client,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API for query enhancement"""
        response = self.llm_client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def _parse_llm_response(self, response: str, original_intent: str) -> Dict[str, Any]:
        """
        Parse LLM response to extract enhancements
        
        Args:
            response: Raw LLM response text
            original_intent: Original detected intent (fallback)
        
        Returns:
            Dictionary with parsed enhancements
        """
        enhancements = {}
        
        def clean_markdown(text: str) -> str:
            """Remove markdown formatting from text"""
            # Remove bold markers
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
            # Remove italic markers
            text = re.sub(r'\*([^*]+)\*', r'\1', text)
            # Remove inline code backticks (single backticks)
            text = re.sub(r'`([^`]+)`', r'\1', text)
            # Remove any remaining backticks
            text = text.replace('`', '')
            # Remove leading/trailing asterisks
            text = text.strip('*').strip()
            return text
        
        try:
            # Debug: log the response for troubleshooting
            self.logger.debug(f"Raw response (repr): {repr(response)}")
            
            # Parse REFINED_INTENT
            # Handle variations: **REFINED_INTENT:** **Code QA** or REFINED_INTENT: Code QA or **REFINED_INTENT:** Code QA
            refined_intent_match = re.search(r'\*{0,2}REFINED_INTENT\*{0,2}:\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
            if refined_intent_match:
                intent = refined_intent_match.group(1).strip()
                # Clean markdown formatting
                intent = clean_markdown(intent).lower()
                # Map to standardized intents
                intent_mapping = {
                    "code qa": "code_qa",
                    "document qa": "document_qa",
                    "api usage": "api_usage",
                    "bug fixing": "bug_fixing",
                    "feature addition": "feature_addition",
                    "architecture": "architecture",
                    "cross-repo": "cross_repo",
                }
                enhancements["refined_intent"] = intent_mapping.get(intent, intent.replace(" ", "_"))
            
            # Parse REWRITTEN_QUERY (note: handle both REWRITTEN and REWRITEN typo)
            # Match until next field (look ahead for next uppercase field) - use DOTALL for multi-line
            rewritten_match = re.search(r'\*{0,2}REWRIT(?:T|)EN_QUERY\*{0,2}:\s*(.+?)(?=\n\s*\*{0,2}[A-Z_]+\*{0,2}:|$)', response, re.IGNORECASE | re.DOTALL)
            if rewritten_match:
                rewritten = rewritten_match.group(1).strip()
                # Clean markdown formatting
                rewritten = clean_markdown(rewritten)
                # Remove quotes if present
                rewritten = re.sub(r'^["\']|["\']$', '', rewritten)
                # Join multi-line content into single line
                rewritten = ' '.join(rewritten.split())
                if rewritten:
                    enhancements["rewritten_query"] = rewritten
            else:
                self.logger.debug("Failed to match REWRITTEN_QUERY")
            
            # Parse SELECTED_KEYWORDS
            # Handle both single-line and potential multi-line keywords with backticks
            # Match from SELECTED_KEYWORDS to the next field or end
            keywords_match = re.search(r'\*{0,2}SELECTED_KEYWORDS\*{0,2}:\s*(.+?)(?=\n\s*\*{0,2}[A-Z_]+\*{0,2}:|$)', response, re.IGNORECASE | re.DOTALL)
            if keywords_match:
                keywords_str = keywords_match.group(1).strip()
                # Clean markdown formatting (including backticks)
                keywords_str = clean_markdown(keywords_str)
                # Remove newlines and extra spaces
                keywords_str = ' '.join(keywords_str.split())
                # Split by comma
                keywords = [k.strip() for k in keywords_str.split(',') if k.strip() and k.strip().lower() != 'none']
                # Additional cleaning: remove any remaining special characters
                keywords = [re.sub(r'[`*]', '', k) for k in keywords]
                keywords = [k.strip() for k in keywords if k.strip()]
                enhancements["selected_keywords"] = keywords[:10]  # Limit to 10 additional keywords
            else:
                self.logger.debug("Failed to match SELECTED_KEYWORDS")
            
            
            # Parse PSEUDOCODE_HINTS
            # Handle both single-line and multi-line code blocks with triple backticks
            pseudocode_match = re.search(r'\*{0,2}PSEUDOCODE_HINTS\*{0,2}:\s*(.+?)(?=\n\s*\*{0,2}[A-Z_]+\*{0,2}:|$)', response, re.IGNORECASE | re.DOTALL)
            if pseudocode_match:
                pseudocode = pseudocode_match.group(1).strip()
                # Remove code block markers if present (handle ```language or just ```)
                pseudocode = re.sub(r'^```[\w]*\s*\n', '', pseudocode, flags=re.MULTILINE)
                pseudocode = re.sub(r'\n\s*```\s*$', '', pseudocode, flags=re.MULTILINE)
                # Clean markdown formatting but preserve code structure
                pseudocode = pseudocode.strip('*').strip()
                # Check if it's a meaningful value
                if pseudocode and pseudocode.lower() not in ["n/a", "none", "not applicable"]:
                    # If it still starts with ```, it means the regex didn't work, try simpler approach
                    if pseudocode.startswith('```'):
                        # Extract content between triple backticks
                        code_block_match = re.search(r'```[\w]*\s*\n(.+?)\n```', pseudocode, re.DOTALL)
                        if code_block_match:
                            pseudocode = code_block_match.group(1).strip()
                    if pseudocode and not pseudocode.startswith('```'):
                        enhancements["pseudocode_hints"] = pseudocode
            else:
                self.logger.debug("Failed to match PSEUDOCODE_HINTS")
            
            self.logger.debug(f"Parsed LLM enhancements: {list(enhancements.keys())}")
            
        except Exception as e:
            self.logger.warning(f"Error parsing LLM response: {e}")
        
        return enhancements
    
    def _resolve_references_and_rewrite(self, query: str, dialogue_history: List[Dict[str, Any]]) -> str:
        """
        Resolve references and rewrite query based on dialogue history
        
        Args:
            query: Current user query
            dialogue_history: List of previous dialogue summaries
        
        Returns:
            Rewritten query with resolved references
        """
        if not self.llm_client:
            self.logger.warning("LLM client not available, skipping reference resolution")
            return query
        
        try:
            # Get recent summaries (limited by history_summary_rounds)
            recent_summaries = dialogue_history[-self.history_summary_rounds:] if len(dialogue_history) > self.history_summary_rounds else dialogue_history
            
            # Build prompt for reference resolution
            prompt = self._build_reference_resolution_prompt(query, recent_summaries)
            
            # Call LLM
            if self.provider == "openai":
                response = self._call_openai(prompt)
            elif self.provider == "anthropic":
                response = self._call_anthropic(prompt)
            else:
                return query
            
            # Parse rewritten query
            rewritten_query = self._parse_rewritten_query(response)
            
            return rewritten_query if rewritten_query else query
            
        except Exception as e:
            self.logger.error(f"Reference resolution failed: {e}")
            return query
    
    def _build_reference_resolution_prompt(self, query: str, recent_summaries: List[Dict[str, Any]]) -> str:
        """Build prompt for reference resolution and query rewriting"""
        
        prompt_parts = [
            "You are a query rewriting assistant for a code search system.",
            "Your task is to resolve references and rewrite the user's current query based on conversation history.",
            "",
            "**Conversation History (Recent Summaries):**"
        ]
        
        for summary_data in recent_summaries:
            turn_num = summary_data.get("turn_number", 0)
            prev_query = summary_data.get("query", "")
            summary = summary_data.get("summary", "")
            
            prompt_parts.append(f"\nTurn {turn_num}:")
            prompt_parts.append(f"Query: {prev_query}")
            if summary:
                prompt_parts.append(f"Summary: {summary}")
        
        prompt_parts.extend([
            "",
            f"**Current Query:** {query}",
            "",
            "**Instructions:**",
            "1. Identify any ambiguous references in the current query (e.g., 'that function', 'the class mentioned above', 'it')",
            "2. Resolve these references using information from the conversation history",
            "3. Rewrite the query to be self-contained and clear, without ambiguous references",
            "4. Keep the rewritten query concise but complete",
            "5. If there are no ambiguous references, return the original query",
            "6. IMPORTANT: Output ONLY the rewritten query text, nothing else",
            "",
            "Rewritten Query:"
        ])
        
        return "\n".join(prompt_parts)
    
    def _parse_rewritten_query(self, response: str) -> Optional[str]:
        """Parse LLM response to extract rewritten query"""
        # Clean up the response
        rewritten = response.strip()
        
        # Remove common prefixes if present
        prefixes_to_remove = [
            "rewritten query:",
            "rewritten:",
            "query:",
        ]
        
        for prefix in prefixes_to_remove:
            if rewritten.lower().startswith(prefix):
                rewritten = rewritten[len(prefix):].strip()
        
        # Remove quotes if present
        if rewritten.startswith('"') and rewritten.endswith('"'):
            rewritten = rewritten[1:-1]
        elif rewritten.startswith("'") and rewritten.endswith("'"):
            rewritten = rewritten[1:-1]
        
        return rewritten if rewritten else None

