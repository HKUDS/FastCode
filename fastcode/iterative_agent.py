"""
Iterative Agent - Multi-round retrieval with confidence-based iteration control
"""

import json
import logging
import os
import re
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
import numpy as np

from .agent_tools import AgentTools
from .path_utils import PathUtils


class IterativeAgent:
    """Agent for managing multi-round iterative retrieval with confidence and cost control"""

    def __init__(self, config: Dict[str, Any], retriever, repo_root: str, bm25_elements=None):
        """
        Initialize iterative agent

        Args:
            config: Configuration dictionary
            retriever: Reference to HybridRetriever instance
            repo_root: Root directory of the repository
            bm25_elements: Reference to indexed BM25 elements for retrieving file details
        """
        self.config = config
        self.retriever = retriever
        self.repo_root = repo_root
        self.bm25_elements = bm25_elements
        self.logger = logging.getLogger(__name__)

        # Initialize tools
        self.tools = AgentTools(repo_root)

        # Initialize path utilities
        self.path_utils = PathUtils(repo_root)
        
        # Agent settings
        self.agent_config = config.get("agent", {}).get("iterative", {})
        self.gen_config = config.get("generation", {})
        
        # Adaptive thresholds (will be set dynamically)
        self.base_max_iterations = self.agent_config.get("max_iterations", 4)
        self.base_confidence_threshold = self.agent_config.get("confidence_threshold", 95)
        self.min_confidence_gain = self.agent_config.get("min_confidence_gain", 5)
        self.max_total_lines = self.agent_config.get("max_total_lines", 12000) #12000

        self.temperature = self.agent_config.get("temperature_agent", 0.2)
        self.max_tokens = self.agent_config.get("max_tokens_agent", 6000)

        # Element limits
        self.max_elements = self.agent_config.get("max_elements", 100)
        self.max_candidates_display = self.agent_config.get("max_candidates_display", 100)
        
        # Dynamic thresholds (set per query)
        self.max_iterations = self.base_max_iterations
        self.confidence_threshold = self.base_confidence_threshold
        self.adaptive_line_budget = self.max_total_lines
        
        # LLM settings
        load_dotenv()
        self.provider = self.gen_config.get("provider", "openai")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        self.model = os.getenv("MODEL")

        
        # Initialize LLM client
        self.client = self._initialize_client()
        
        # Repo statistics (will be set later)
        self.repo_stats = None
        
        # Iteration history
        self.iteration_history = []
        self.tool_call_history = []
    
    def _initialize_client(self):
        """Initialize LLM client based on provider"""
        if self.provider == "openai":
            api_key = self.api_key
            if not api_key:
                self.logger.warning("OPENAI_API_KEY not set")
            return OpenAI(api_key=api_key, base_url=self.base_url)
        
        elif self.provider == "anthropic":
            api_key = self.anthropic_api_key
            if not api_key:
                self.logger.warning("ANTHROPIC_API_KEY not set")
            return Anthropic(api_key=api_key, base_url=self.base_url)
        
        else:
            self.logger.warning(f"Unknown provider: {self.provider}")
            return None
    
    def set_repo_stats(self, repo_stats: Dict[str, Any]):
        """Set repository statistics for cost calculation"""
        self.repo_stats = repo_stats
        self.logger.info(f"Set repo stats: {repo_stats}")
    
    def _initialize_adaptive_parameters(self, query_complexity: int):
        """
        Initialize adaptive parameters based on query and repo complexity
        
        Args:
            query_complexity: Query complexity score (0-100)
        """
        # Calculate repo complexity factor
        repo_factor = self._calculate_repo_factor()
        
        # Adaptive max iterations: scales with complexity
        # Simple query in simple repo: 2-3 iterations
        # Complex query in complex repo: 4-5 iterations
        complexity_score = (query_complexity / 100 + repo_factor) / 2
        self.max_iterations = max(2, min(6, int(self.base_max_iterations * (0.7 + complexity_score * 0.6))))
        
        # Adaptive confidence threshold: more complex queries can accept slightly lower confidence
        # to avoid over-iteration when perfect confidence is hard to achieve
        if query_complexity >= 80:
            # Very complex: accept 90% confidence
            self.confidence_threshold = max(90, self.base_confidence_threshold - 5)
        elif query_complexity >= 60:
            # Moderately complex: accept 92% confidence
            self.confidence_threshold = max(92, self.base_confidence_threshold - 3)
        else:
            # Simple: maintain high standard
            self.confidence_threshold = self.base_confidence_threshold
        
        # Adaptive line budget: scales with complexity
        # Simple queries: smaller budget to force conciseness
        # Complex queries: larger budget to allow thorough exploration
        if query_complexity <= 30:
            self.adaptive_line_budget = int(self.max_total_lines * 0.6)  # 6000 lines
        elif query_complexity <= 60:
            self.adaptive_line_budget = int(self.max_total_lines * 0.8)  # 8000 lines
        else:
            self.adaptive_line_budget = int(self.max_total_lines * 1.0 * repo_factor)  # Up to 20000 lines for very complex
        
        self.logger.info(
            f"Adaptive parameters set: max_iterations={self.max_iterations}, "
            f"confidence_threshold={self.confidence_threshold}, "
            f"line_budget={self.adaptive_line_budget}, "
            f"query_complexity={query_complexity}, repo_factor={repo_factor:.2f}"
        )
    
    def retrieve_with_iteration(self, query: str, processed_query, query_info: Dict[str, Any],
                                repo_filter: Optional[List[str]] = None,
                                dialogue_history: Optional[List[Dict[str, Any]]] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Main entry point for iterative retrieval

        Args:
            query: User query
            processed_query: ProcessedQuery object from query_processor
            query_info: Query information dict
            repo_filter: Optional list of repository names to filter
            dialogue_history: Previous dialogue summaries for multi-turn context

        Returns:
            Tuple of (final_results, iteration_metadata)
        """
        self.logger.info("Starting iterative retrieval")
        self.iteration_history = []
        self.tool_call_history = []

        # Store dialogue_history for use in prompts
        self.dialogue_history = dialogue_history

        # Round 1: Initial assessment and retrieval (with dialogue context)
        round1_result = self._round_one(query, processed_query, query_info, repo_filter, dialogue_history)
        self._record_tool_calls(1, round1_result.get("tool_calls", []), repo_filter)
        
        # Initialize adaptive parameters based on query complexity from round 1
        query_complexity = round1_result.get("query_complexity", 50)
        self._initialize_adaptive_parameters(query_complexity)
        
        self.logger.info(f"Round 1 result: confidence={round1_result.get('confidence')}, should_answer_directly={round1_result.get('should_answer_directly')}")
        
        if round1_result["should_answer_directly"]:
            # High confidence in round 1, but we force retrieval to ensure robustness (min 2 rounds)
            self.logger.info("High confidence in round 1, but forcing retrieval (min 2 rounds enforcement)")
            # We do NOT return here, effectively disabling the early exit.
            # return [], {
            #     "rounds": 1,
            #     "answered_directly": True,
            #     "initial_confidence": round1_result["confidence"],
            #     "query_complexity": round1_result.get("query_complexity", 0)
            # }
        
        # Perform retrieval based on round 1 decisions
        round1_elements = self._execute_round_one_retrieval(
            query, processed_query, query_info, round1_result, repo_filter
        )
        
        # Record round 1 results
        total_lines_r1 = self._calculate_total_lines(round1_elements)
        self.iteration_history.append({
            "round": 1,
            "confidence": round1_result["confidence"],
            "query_complexity": round1_result.get("query_complexity", 0),
            "elements_count": len(round1_elements),
            "total_lines": total_lines_r1,
            "confidence_gain": 0,
            "lines_added": total_lines_r1,
            "roi": 0.0,
            "budget_usage_pct": (total_lines_r1 / self.adaptive_line_budget) * 100
        })
        
        retained_elements = round1_elements
        pending_elements = []
        current_round = 2
        last_round_unfiltered_elements = None
        
        # Iterative rounds (2 to n)
        while current_round <= self.max_iterations:
            # Always include previous round elements plus newly found ones
            current_elements = self._merge_elements(retained_elements, pending_elements) if pending_elements else retained_elements
            self.logger.info(f"Starting round {current_round}")

            # Round n: Assessment and potential iteration (with dialogue_history)
            round_result = self._round_n(query, current_elements, query_info, current_round, dialogue_history)
            filtered_tool_calls = self._filter_redundant_tool_calls(
                current_round, round_result.get("tool_calls", []), repo_filter
            )
            round_result["tool_calls"] = filtered_tool_calls
            self._record_tool_calls(current_round, filtered_tool_calls, repo_filter)

            # Preserve unfiltered elements for last-round fallback
            last_round_unfiltered_elements = current_elements

            # Filter elements based on keep_files
            num_elements_before_filter = len(current_elements)
            if round_result.get("keep_files"):
                filtered_elements = self._filter_elements_by_keep_files(
                    current_elements, round_result["keep_files"]
                )
            else:
                filtered_elements = self._filter_elements_by_keep_files(current_elements, [])

            # Log element count change after filtering
            self.logger.info(
                f"Round {current_round} element filtering: {num_elements_before_filter} -> {len(filtered_elements)} elements"
            )

            # Allow the new round to explicitly drop previously retained items
            retained_elements = filtered_elements
            current_elements = retained_elements
            pending_elements = []
            
            confidence = round_result["confidence"]
            self.logger.info(f"Round {current_round} confidence: {confidence}")
            
            # Calculate metrics for this round
            total_lines = self._calculate_total_lines(current_elements)
            prev_confidence = self.iteration_history[-1]["confidence"]
            prev_lines = self.iteration_history[-1]["total_lines"]
            confidence_gain = confidence - prev_confidence
            lines_added = total_lines - prev_lines
            roi = (confidence_gain / lines_added * 1000) if lines_added > 0 else 0.0
            budget_usage_pct = (total_lines / self.adaptive_line_budget) * 100
            
            # Record round results with detailed metrics
            self.iteration_history.append({
                "round": current_round,
                "confidence": confidence,
                "elements_count": len(current_elements),
                "total_lines": total_lines,
                "confidence_gain": confidence_gain,
                "lines_added": lines_added,
                "roi": roi,
                "budget_usage_pct": budget_usage_pct
            })
            
            self.logger.info(
                f"Round {current_round} metrics: confidence={confidence:.1f} (+{confidence_gain:.1f}), "
                f"elements={len(current_elements)}, lines={total_lines} (+{lines_added}), "
                f"ROI={roi:.2f}, budget_usage={budget_usage_pct:.1f}%"
            )
            
            # Check if we should stop iteration
            if confidence >= self.confidence_threshold:
                self.logger.info(f"Confidence threshold reached: {confidence} >= {self.confidence_threshold}")
                break
            
            # Check if we should continue to next round
            if not round_result.get("tool_calls"):
                self.logger.info("No new tool calls requested, stopping iteration")
                break
            
            # Calculate cost and decide whether to continue
            should_continue = self._should_continue_iteration(
                current_round, confidence, current_elements, round1_result.get("query_complexity", 50)
            )
            
            if not should_continue:
                self.logger.info("Cost threshold exceeded or marginal gain too low, stopping iteration")
                break
            
            # Execute tool calls for next round
            new_elements = self._execute_tool_calls_round_n(
                query, round_result["tool_calls"], repo_filter, current_elements
            )
            
            if not new_elements:
                self.logger.info("No new elements found, stopping iteration")
                break
            
            # Defer merge to next round so selection considers previous + new together
            pending_elements = new_elements
            
            current_round += 1
        
        # Prepare iteration metadata with efficiency analysis
        final_elements = self._merge_elements(retained_elements, pending_elements) if pending_elements else retained_elements
        
        # Fallback: if last round filtered everything out, use pre-filter elements
        if not final_elements and last_round_unfiltered_elements:
            total_lines = self._calculate_total_lines(last_round_unfiltered_elements)
            if total_lines > self.adaptive_line_budget:
                self.logger.info(
                    f"Fallback to pre-filter elements with smart pruning: "
                    f"{total_lines} lines > {self.adaptive_line_budget} budget"
                )
                final_elements = self._smart_prune_elements(last_round_unfiltered_elements)
            else:
                self.logger.info(
                    "Fallback to pre-filter elements without pruning "
                    f"({total_lines} lines <= {self.adaptive_line_budget} budget)"
                )
                final_elements = last_round_unfiltered_elements
        iteration_metadata = self._generate_iteration_metadata(round1_result, final_elements)
        
        self.logger.info(f"Iterative retrieval completed: {iteration_metadata}")
        
        return final_elements, iteration_metadata
    
    def _generate_iteration_metadata(self, round1_result: Dict[str, Any], 
                                     final_elements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate comprehensive iteration metadata with efficiency analysis"""
        if not self.iteration_history:
            return {
                "rounds": 0,
                "answered_directly": False,
                "total_elements": 0,
                "total_lines": 0
            }
        
        # Basic stats
        initial_confidence = self.iteration_history[0]["confidence"]
        final_confidence = self.iteration_history[-1]["confidence"]
        total_confidence_gain = final_confidence - initial_confidence
        total_lines = self._calculate_total_lines(final_elements)
        
        # Efficiency metrics
        overall_roi = (total_confidence_gain / total_lines * 1000) if total_lines > 0 else 0.0
        
        # Per-round efficiency
        round_efficiencies = []
        for h in self.iteration_history[1:]:  # Skip round 1 for ROI
            round_efficiencies.append({
                "round": h["round"],
                "confidence_gain": h["confidence_gain"],
                "lines_added": h["lines_added"],
                "roi": h["roi"]
            })
        
        # Adaptive parameter effectiveness
        budget_used_pct = (total_lines / self.adaptive_line_budget) * 100
        iterations_used_pct = (len(self.iteration_history) / self.max_iterations) * 100
        
        # Stopping reason analysis
        stopping_reason = self._determine_stopping_reason(final_confidence)
        
        metadata = {
            # Basic info
            "rounds": len(self.iteration_history),
            "answered_directly": False,
            "query_complexity": round1_result.get("query_complexity", 0),
            
            # Confidence metrics
            "initial_confidence": initial_confidence,
            "final_confidence": final_confidence,
            "confidence_gain": total_confidence_gain,
            
            # Resource metrics
            "total_elements": len(final_elements),
            "total_lines": total_lines,
            "budget_used_pct": budget_used_pct,
            "iterations_used_pct": iterations_used_pct,
            
            # Efficiency metrics
            "overall_roi": overall_roi,
            "round_efficiencies": round_efficiencies,
            
            # Adaptive parameters used
            "adaptive_params": {
                "max_iterations": self.max_iterations,
                "confidence_threshold": self.confidence_threshold,
                "line_budget": self.adaptive_line_budget
            },
            
            # Analysis
            "stopping_reason": stopping_reason,
            "efficiency_rating": self._rate_efficiency(overall_roi, budget_used_pct),
            
            # Detailed history
            "history": self.iteration_history
        }
        
        return metadata
    
    def _determine_stopping_reason(self, final_confidence: int) -> str:
        """Determine why iteration stopped"""
        if final_confidence >= self.confidence_threshold:
            return "confidence_threshold_reached"
        elif len(self.iteration_history) >= self.max_iterations:
            return "max_iterations_reached"
        elif self.iteration_history and self.iteration_history[-1]["total_lines"] >= self.adaptive_line_budget:
            return "line_budget_exceeded"
        elif len(self.iteration_history) >= 3:
            recent_gains = [h["confidence_gain"] for h in self.iteration_history[-2:]]
            if all(g < self.min_confidence_gain for g in recent_gains):
                return "diminishing_returns"
        return "other"
    
    def _rate_efficiency(self, overall_roi: float, budget_used_pct: float) -> str:
        """Rate the efficiency of the iteration process"""
        # Good: High ROI and reasonable budget usage
        if overall_roi >= 5.0 and budget_used_pct < 70:
            return "excellent"
        elif overall_roi >= 3.0 and budget_used_pct < 85:
            return "good"
        elif overall_roi >= 1.5 or budget_used_pct < 90:
            return "acceptable"
        else:
            return "inefficient"
    
    def _round_one(self, query: str, processed_query, query_info: Dict[str, Any],
                   repo_filter: Optional[List[str]] = None,
                   dialogue_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Round 1: Initial assessment without any file content

        Returns:
            Dict with:
            - confidence: int (0-100)
            - query_complexity: int (0-100)
            - tool_calls: List[Dict] (if confidence < 95)
            - query_enhancement_params: Dict (if needed)
            - should_answer_directly: bool
        """
        self.logger.info("Round 1: Initial assessment")

        # Build prompt for round 1 (with dialogue history if available)
        prompt = self._build_round_one_prompt(query, processed_query, query_info, repo_filter, dialogue_history)

        # Call LLM
        response = self._call_llm(prompt)

        # Parse response
        result = self._parse_round_one_response(response)

        return result

    def _build_round_one_prompt(self, query: str, processed_query, query_info: Dict[str, Any],
                                repo_filter: Optional[List[str]] = None,
                                dialogue_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Build prompt for round 1 assessment with optional dialogue history"""

        # Get repository structure
        selected_repos = repo_filter or []
        repo_structure = self._generate_directory_tree(selected_repos)

        # print("repo_structure: ", repo_structure)

        # Confidence scoring rules
        confidence_rules = """
CONFIDENCE SCORING RULES (0-100):
- 95-100: You have complete knowledge to answer this question without needing any code files
- 80-94: You have good general knowledge but need to see specific implementation details
- 60-79: You understand the domain but need to examine the codebase structure and key files
- 40-59: The question requires detailed code inspection across multiple files
- 20-39: Complex cross-file analysis or deep implementation details needed
- 0-19: Highly specific question requiring comprehensive codebase examination

IMPORTANT: At this stage, you have NOT seen any code files yet. Base your confidence ONLY on:
1. Whether this is a general knowledge question vs specific implementation question
2. Whether the question asks about standard patterns vs custom implementation
3. Your general understanding of the technology/framework mentioned
"""

        # Build dialogue history context if available
        dialogue_context = ""
        if dialogue_history and len(dialogue_history) > 0:
            dialogue_context = "\n**Previous Conversation Context**:\n"
            for idx, turn in enumerate(dialogue_history[-10:], 1):  # Last 10 turns
                turn_query = turn.get("query", "")
                turn_summary = turn.get("summary", "")
                dialogue_context += f"\nTurn {idx}:\n"
                dialogue_context += f"  Query: {turn_query}\n"
                if turn_summary:
                    # Extract key information from summary
                    summary_preview = turn_summary[:]
                    dialogue_context += f"  Summary: {summary_preview}\n"
            dialogue_context += "\n**IMPORTANT**: Use this context to understand references in the current query (e.g., 'this function', 'that class'). The current query may refer to entities discussed in previous turns.\n"

        prompt = f"""You are a code analysis agent performing initial query assessment. You have NOT seen any code files yet.
{dialogue_context}
**Current User Query**: {query}

**Repository Structure**:
{repo_structure}

**Your Task**: Assess the query and decide on the retrieval strategy. If there is previous conversation context, use it to resolve any references (e.g., "this", "that", "the function") in the current query.

{confidence_rules}

**Output Format** (JSON only):

If confidence >= 95:
{{
  "confidence": <0-100>,
  "reasoning": "Brief explanation"
}}

If confidence < 95:
{{
  "confidence": <0-100>,
  "query_complexity": <0-100>,
  "reasoning": "Brief explanation",
  "query_enhancement": {{
    "needed": true/false,
    "refined_intent": "<intent>",
    "rewritten_query": "<optimized English query for semantic/BM25 retrieval, with key technical terms and concepts>",
    "selected_keywords": ["kw1", "kw2"],
    "pseudocode_hints": "<pseudocode or null>"
  }},
  "tool_calls": [
    {{"tool": "search_codebase", "parameters": {{"search_term": "...", "file_pattern": "*.py", "use_regex": false}}}},
    {{"tool": "list_directory", "parameters": {{"path": "src/core"}}}}
  ]
}}

**Query Complexity Scoring (0-100)**:
- 0-20: Simple lookup (find a function/class)
- 21-40: Single-file analysis (understand one component)
- 41-60: Multi-file analysis (trace logic across files)
- 61-80: Cross-module/architectural understanding
- 81-100: Complex debugging or system-wide refactoring questions

**Query Rewriting Guidelines**:
- Translate non-English queries to English for optimal retrieval accuracy
- Expand abbreviations and resolve references from dialogue context
- Include technical terms, class/function names, and domain-specific keywords
- Keep concise while preserving all essential meaning

**Tool Call Guidelines**:
- Use search_codebase for finding specific terms, classes, functions
  * search_term: literal text or regex pattern to find in file contents
  * file_pattern: SINGLE glob pattern per tool call to filter files (only one pattern allowed)
    * Format: "RepoName/actual_source_path/**/*.ext" (e.g., "django/django/**/*.py")
    * For all repos: omit repo prefix, use "**/*.ext"
  * use_regex: true if search_term is regex, false for literal (default: false)

- Use list_directory to explore directory structure
  * path: "RepoName/path/to/dir" format (e.g., "django/django/core")
  * For repo root: use "RepoName" (e.g., "django")

  **Note**: Repos often nest project folders (django/django/, flask/src/flask/). Always include the full path from repo root, not just the inner folder.

- Maximum 10 tool calls
- Be strategic: target likely locations based on query and repo structure
- Do not use the model's native tool_calls format. Instead, include tool call instructions in your text response content in a parseable format

**CRITICAL**:
- Respond with valid JSON only
- No markdown code blocks
- No comments in JSON
- If confidence >= 95, ONLY output confidence and reasoning"""
        
        return prompt
    
    def _parse_round_one_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response from round 1"""
        try:
            json_str = self._extract_json_from_response(response)
            data = self._robust_json_parse(json_str)
            
            confidence = data.get("confidence", 0)
            should_answer_directly = confidence >= self.confidence_threshold
            
            result = {
                "confidence": confidence,
                "reasoning": data.get("reasoning", ""),
                "should_answer_directly": should_answer_directly
            }
            
            if not should_answer_directly:
                result["query_complexity"] = data.get("query_complexity", 50)
                query_enhancement = self._normalize_query_enhancement(
                    data.get("query_enhancement", {})
                )
                fallback_enhancement = self._parse_query_enhancement_fallback(response)
                for key, value in fallback_enhancement.items():
                    if not query_enhancement.get(key):
                        query_enhancement[key] = value
                if "needed" not in query_enhancement:
                    query_enhancement["needed"] = any(
                        query_enhancement.get(k)
                        for k in ("refined_intent", "rewritten_query", "selected_keywords", "pseudocode_hints")
                    )
                result["query_enhancement"] = query_enhancement
                result["tool_calls"] = data.get("tool_calls", [])

                # DEBUG: Log tool calls to verify LLM prompt compliance
                if result["tool_calls"]:
                    self.logger.debug(f"[DEBUG] Round 1 LLM tool_calls ({len(result['tool_calls'])} calls):")
                    for idx, tc in enumerate(result["tool_calls"]):
                        tool_name = tc.get("tool", "unknown")
                        params = tc.get("parameters", {})
                        self.logger.info(f"  [{idx+1}] {tool_name}: {params}")
                        # Check prompt compliance for path format
                        if tool_name == "list_directory":
                            path = params.get("path", "")
                            self.logger.info(f"    -> list_directory path='{path}' (check if repo prefix included for multi-repo)")
                        elif tool_name == "search_codebase":
                            root_path = params.get("root_path", "not_specified")
                            search_term = params.get("search_term", "")
                            self.logger.info(f"    -> search_codebase root_path='{root_path}', search_term='{search_term}'")

            self.logger.info(f"Round 1 parsed: confidence={confidence}, tool_calls={len(result.get('tool_calls', []))}, should_answer_directly={should_answer_directly}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to parse round 1 response: {e}")
            self.logger.debug(f"Response content (first 500 chars): {response[:500]}")
            # Fallback: assume low confidence, no enhancement
            fallback_enhancement = self._parse_query_enhancement_fallback(response)
            if fallback_enhancement and "needed" not in fallback_enhancement:
                fallback_enhancement["needed"] = True
            return {
                "confidence": 50,
                "query_complexity": 50,
                "reasoning": "Parse error",
                "should_answer_directly": False,
                "query_enhancement": fallback_enhancement or {"needed": False},
                "tool_calls": []
            }

    def _normalize_query_enhancement(self, query_enhancement: Any) -> Dict[str, Any]:
        """Normalize query enhancement payload to a consistent dict format."""
        normalized = {}
        if isinstance(query_enhancement, dict):
            normalized = dict(query_enhancement)
        elif isinstance(query_enhancement, str):
            normalized = self._parse_query_enhancement_fallback(query_enhancement)

        refined_intent = normalized.get("refined_intent")
        if refined_intent is not None:
            intent_text = str(refined_intent).strip().lower()
            intent_mapping = {
                "code qa": "code_qa",
                "document qa": "document_qa",
                "api usage": "api_usage",
                "bug fixing": "bug_fixing",
                "feature addition": "feature_addition",
                "architecture": "architecture",
                "cross-repo": "cross_repo",
            }
            normalized["refined_intent"] = intent_mapping.get(
                intent_text, intent_text.replace(" ", "_")
            )

        rewritten = normalized.get("rewritten_query")
        if isinstance(rewritten, str):
            rewritten = rewritten.strip()
            rewritten = re.sub(r'^["\']|["\']$', '', rewritten)
            rewritten = " ".join(rewritten.split())
            if rewritten:
                normalized["rewritten_query"] = rewritten
            else:
                normalized.pop("rewritten_query", None)

        selected_keywords = normalized.get("selected_keywords")
        if isinstance(selected_keywords, str):
            keywords_str = " ".join(selected_keywords.split())
            keywords = [k.strip() for k in keywords_str.split(",") if k.strip()]
            normalized["selected_keywords"] = keywords[:10]
        elif isinstance(selected_keywords, list):
            keywords = [str(k).strip() for k in selected_keywords if str(k).strip()]
            normalized["selected_keywords"] = keywords[:10]

        pseudocode = normalized.get("pseudocode_hints")
        if isinstance(pseudocode, str):
            pseudocode = re.sub(r'^```[\w]*\s*\n', '', pseudocode, flags=re.MULTILINE)
            pseudocode = re.sub(r'\n\s*```\s*$', '', pseudocode, flags=re.MULTILINE)
            pseudocode = pseudocode.strip('*').strip()
            if pseudocode and pseudocode.lower() not in ["n/a", "none", "not applicable"]:
                normalized["pseudocode_hints"] = pseudocode
            else:
                normalized.pop("pseudocode_hints", None)

        return normalized

    def _parse_query_enhancement_fallback(self, response: str) -> Dict[str, Any]:
        """Fallback parsing for query enhancement fields in non-JSON outputs."""
        enhancements = {}

        if not response:
            return enhancements

        def clean_markdown(text: str) -> str:
            text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)
            text = re.sub(r'\*([^*]+)\*', r'\1', text)
            text = re.sub(r'`([^`]+)`', r'\1', text)
            text = text.replace('`', '')
            text = text.strip('*').strip()
            return text

        refined_intent_match = re.search(
            r'\*{0,2}REFINED_INTENT\*{0,2}:\s*(.+?)(?:\n|$)',
            response,
            re.IGNORECASE
        )
        if refined_intent_match:
            intent = clean_markdown(refined_intent_match.group(1).strip()).lower()
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

        rewritten_match = re.search(
            r'\*{0,2}REWRIT(?:T|)EN_QUERY\*{0,2}:\s*(.+?)(?=\n\s*\*{0,2}[A-Z_]+\*{0,2}:|$)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if rewritten_match:
            rewritten = clean_markdown(rewritten_match.group(1).strip())
            rewritten = re.sub(r'^["\']|["\']$', '', rewritten)
            rewritten = " ".join(rewritten.split())
            if rewritten:
                enhancements["rewritten_query"] = rewritten

        keywords_match = re.search(
            r'\*{0,2}SELECTED_KEYWORDS\*{0,2}:\s*(.+?)(?=\n\s*\*{0,2}[A-Z_]+\*{0,2}:|$)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if keywords_match:
            keywords_str = clean_markdown(keywords_match.group(1).strip())
            keywords_str = " ".join(keywords_str.split())
            keywords = [k.strip() for k in keywords_str.split(",") if k.strip() and k.strip().lower() != "none"]
            enhancements["selected_keywords"] = keywords[:10]

        pseudocode_match = re.search(
            r'\*{0,2}PSEUDOCODE_HINTS\*{0,2}:\s*(.+?)(?=\n\s*\*{0,2}[A-Z_]+\*{0,2}:|$)',
            response,
            re.IGNORECASE | re.DOTALL
        )
        if pseudocode_match:
            pseudocode = pseudocode_match.group(1).strip()
            pseudocode = re.sub(r'^```[\w]*\s*\n', '', pseudocode, flags=re.MULTILINE)
            pseudocode = re.sub(r'\n\s*```\s*$', '', pseudocode, flags=re.MULTILINE)
            pseudocode = pseudocode.strip('*').strip()
            if pseudocode and pseudocode.lower() not in ["n/a", "none", "not applicable"]:
                enhancements["pseudocode_hints"] = pseudocode

        return enhancements
    
    def _perform_standard_retrieval(self, processed_query, filters, repo_filter):
        """
        Perform standard retrieval without triggering iterative mode recursion
        Directly calls retriever's internal methods (_semantic_search, _keyword_search, etc.)
        instead of the top-level retrieve() method which would detect iterative mode
        and cause infinite recursion.
        
        IMPORTANT: Do NOT call self.retriever.retrieve() here - use internal methods only
        """
        self.logger.info("Performing standard retrieval (bypass iterative mode)")
        
        # Get query text
        if hasattr(processed_query, 'rewritten_query') and processed_query.rewritten_query:
            search_text = processed_query.rewritten_query
        else:
            search_text = processed_query.original if hasattr(processed_query, 'original') else str(processed_query)
        
        keywords = processed_query.keywords if hasattr(processed_query, 'keywords') else []
        pseudocode = processed_query.pseudocode_hints if hasattr(processed_query, 'pseudocode_hints') else None
        
        self.logger.debug(f"Search text: {search_text[:100]}...")
        
        # Stage 1: Semantic search
        semantic_results = self.retriever._semantic_search(search_text, top_k=20, repo_filter=repo_filter)
        
        # Stage 2: Pseudocode search (if available)
        pseudocode_results = []
        if pseudocode:
            pseudocode_results = self.retriever._semantic_search(pseudocode, top_k=10, repo_filter=repo_filter)
        
        # Stage 3: Keyword search
        keyword_query = " ".join(keywords) if keywords else search_text
        keyword_results = self.retriever._keyword_search(keyword_query, top_k=10, repo_filter=repo_filter)
        
        # Stage 4: Combine results
        combined_results = self.retriever._combine_results(
            semantic_results, keyword_results, pseudocode_results
        )
        
        # Stage 5: Re-rank
        final_results = self.retriever._rerank(search_text, combined_results)
        
        # # Stage 6: Apply filters
        # if filters:
        #     final_results = self.retriever._apply_filters(final_results, filters)
        
        # Stage 7: Diversification
        final_results = self.retriever._diversify(final_results)
        
        # Limit results
        max_results = self.retriever.max_results
        final_results = final_results[:max_results]
        
        return final_results
    
    def _execute_round_one_retrieval(self, query: str, processed_query, query_info: Dict[str, Any],
                                     round1_result: Dict[str, Any],
                                     repo_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Execute retrieval based on round 1 decisions
        
        Returns:
            List of retrieved elements
        """
        self.logger.info("Executing round 1 retrieval")
        
        # Step 1: Perform standard retrieval (semantic + keyword) - bypass iterative mode
        retrieval_results = self._perform_standard_retrieval(
            processed_query, query_info.get("filters"), repo_filter
        )
        
        self.logger.info(f"Standard retrieval found {len(retrieval_results)} elements")
        
        # Step 2: Execute tool calls if any
        tool_results = []
        if round1_result.get("tool_calls"):
            tool_results = self._execute_tool_calls_with_selection(
                query, round1_result["tool_calls"], repo_filter
            )
            self.logger.info(f"Tool calls found {len(tool_results)} additional elements")
            self.logger.debug(f"Round 1 tool calls elements list: {self._format_element_list(tool_results)}")

        # Step 3: Merge and deduplicate
        all_results = retrieval_results + tool_results
        all_results = self._remove_duplicates_with_containment(all_results)

        # Step 4: Graph expansion
        if len(all_results) > 0:
            expanded_results = self.retriever._expand_with_graph(all_results, max_hops=2)
            self.logger.info(f"Graph expansion resulted in {len(expanded_results)} elements")
            self.logger.debug(f"Round 1 graph expansion elements list: {self._format_element_list(expanded_results)}")
            # Remove duplicates again after graph expansion (handle containment)
            expanded_results = self._remove_duplicates_with_containment(expanded_results)
            self.logger.info(f"After containment-aware deduplication: {len(expanded_results)} elements")
            self.logger.debug(f"Round 1 after dedup elements list: {self._format_element_list(expanded_results)}")
            # Keep only the most relevant elements to avoid large graph expansions
            expanded_results = self._limit_elements_by_relevance(expanded_results, max_elements=self.max_elements)
            return expanded_results
        
        return all_results
    
    def _execute_tool_calls_with_selection(self, query: str, tool_calls: List[Dict[str, Any]],
                                           repo_filter: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Execute tool calls and let LLM select specific elements (file/class/function level)
        
        Returns:
            List of selected elements
        """
        all_candidates = []
        selected_repos = repo_filter or []
        
        # Execute tool calls to get candidates
        for tool_call in tool_calls[:10]:
            tool_name = tool_call.get("tool", "")
            parameters = tool_call.get("parameters", {})
            
            self.logger.debug(f"[In Iterative Agent] Executing tool: {tool_name} with params: {parameters}")
            
            if tool_name == "search_codebase":
                candidates = self._execute_search_codebase(parameters, selected_repos)
                self.logger.info(f"search_codebase returned {len(candidates)} candidates")
                if not candidates:
                    self.logger.warning(f"No candidates returned from search_codebase in iterative agent, params: {parameters}, selected_repos: {selected_repos}")
                self.logger.debug(f"Candidates sample: {[c.get('file_path') for c in candidates[:]]}")
                all_candidates.extend(candidates)
            
            elif tool_name == "list_directory":
                candidates = self._execute_list_directory(parameters, selected_repos)
                self.logger.info(f"list_directory returned {len(candidates)} candidates")
                if not candidates:
                    self.logger.warning(f"No candidates returned from list_directory in iterative agent, params: {parameters}, selected_repos: {selected_repos}")
                self.logger.debug(f"Candidates sample: {[c.get('file_path') for c in candidates[:]]}")
                all_candidates.extend(candidates)
        
        if not all_candidates:
            # Fallback: try repository overview file selection
            self.logger.warning("No candidates from tool calls, attempting fallback")
            if self.retriever:
                try:
                    target_repos = selected_repos
                    if not target_repos and hasattr(self.retriever, "vector_store"):
                        target_repos = self.retriever.vector_store.get_repository_names()
                    if target_repos:
                        fallback_results = self.retriever._enhance_with_file_selection(
                            query, [], target_repos
                        )
                        return fallback_results
                except Exception as e:
                    self.logger.error(f"Fallback selection failed: {e}")
            return []
        
        # Let LLM select specific elements with granularity (file/class/function)
        selected_elements = self._llm_select_elements_with_granularity(query, all_candidates)
        
        return selected_elements
    
    def _llm_select_elements_with_granularity(self, query: str, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Let LLM select specific elements at file/class/function granularity
        
        Returns:
            List of selected element results
        """
        if not candidates:
            return []
        
        prompt = self._build_element_selection_prompt(query, candidates)

        self.logger.debug(f"Element selection prompt: {prompt}")
        
        try:
            response = self._call_llm(prompt)
            selected_items = self._parse_element_selection_response(response)

            self.logger.info(f"LLM selected elements: {selected_items}")
            
            # Convert selected items to element results
            results = self._convert_selections_to_elements(selected_items, candidates)

            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in element selection: {e}")
            # Fallback: return file-level elements
            return self._fallback_file_selection(candidates)
    
    def _build_element_selection_prompt(self, query: str, candidates: List[Dict[str, Any]]) -> str:
        """Build prompt for element-level selection"""
        
        # Format candidates with indexed elements
        candidates_text = self._format_candidates_with_elements(candidates)

        # print("candidates_text: ", candidates_text)
        
        prompt = f"""You are a code analysis agent selecting specific code elements to answer the query.

**User Query**: {query}

**File Candidates with Elements**:
{candidates_text}

**Your Task**: Select the MOST RELEVANT elements to answer the query at the appropriate granularity.

**Selection Granularity**:
- **file**: Select entire file when you need comprehensive context
- **class**: Select specific class when the question focuses on that class
- **function**: Select specific function when the question is about that function

**Selection Strategy**:
- If query asks about a specific class/function, select at that level (don't select entire file)
- If query requires broader context, select file level
- Include all directly relevant elements and any necessary nearby context; avoid tangential or unrelated selections

**Response Format** (JSON only):
{{
  "selected_elements": [
    {{"file_path": "path/to/file.py", "type": "class", "name": "ClassName", "repo_name": "repo"}},
    {{"file_path": "path/to/file.py", "type": "function", "name": "function_name", "repo_name": "repo"}},
    {{"file_path": "path/to/file.py", "type": "file", "repo_name": "repo"}}
  ]
}}

**CRITICAL**:
- Respond with valid JSON only
- No markdown blocks
- No comments
- Each selection must have: file_path, type (file/class/function), repo_name
- For class/function selections, include name field"""
        
        return prompt
    
    def _format_candidates_with_elements(self, candidates: List[Dict[str, Any]]) -> str:
        """Format candidates showing indexed elements"""
        lines = []

        for i, candidate in enumerate(candidates[:self.max_candidates_display], 1):
            file_path = candidate.get("file_path", "")
            indexed_elements = candidate.get("indexed_elements", [])
            match_count = candidate.get("match_count", 0)
            repo_name = candidate.get("repo_name", "")

            lines.append(f"\n{i}. {file_path}")
            if repo_name:
                lines.append(f"   Repo: {repo_name}")
            if match_count > 0:
                lines.append(f"   Matches: {match_count}")
            
            if indexed_elements:
                classes = [e for e in indexed_elements if e.get("type") == "class"]
                functions = [e for e in indexed_elements if e.get("type") == "function" and not e.get("is_method")]
                
                if classes:
                    lines.append(f"   Classes ({len(classes)}):")
                    for elem in classes[:10]:
                        sig = elem.get("signature", "")
                        name = elem.get("name", "")
                        lines.append(f"     - {sig if sig else f'class {name}'}")
                
                if functions:
                    lines.append(f"   Functions ({len(functions)}):")
                    for elem in functions[:10]:
                        sig = elem.get("signature", "")
                        name = elem.get("name", "")
                        lines.append(f"     - {sig if sig else f'def {name}'}")
        
        return "\n".join(lines)
    
    def _parse_element_selection_response(self, response: str) -> List[Dict[str, Any]]:
        """Parse LLM response for element selections"""
        try:
            json_str = self._extract_json_from_response(response)
            data = self._robust_json_parse(json_str)
            return data.get("selected_elements", [])
        except Exception as e:
            self.logger.error(f"Failed to parse element selection: {e}")
            self.logger.debug(f"Response content (first 500 chars): {response[:500]}")
            return []

    def _convert_selections_to_elements(self, selections: List[Dict[str, Any]],
                                       candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Convert selections to retrieval element format"""
        results = []

        self.logger.debug(f"[SELECTION DEBUG] Converting {len(selections)} selections to elements")
        self.logger.debug(f"[SELECTION DEBUG] Have {len(candidates)} candidates to match against")

        # Build known repos set from candidates and bm25_elements
        known_repos = set()
        for candidate in candidates:
            repo = candidate.get("repo_name", "")
            if repo:
                known_repos.add(repo)
        if self.bm25_elements:
            for elem in self.bm25_elements[:100]:  # Sample first 100
                if elem.repo_name:
                    known_repos.add(elem.repo_name)

        self.logger.debug(f"[SELECTION DEBUG] Known repos: {known_repos}")

        for selection in selections:
            file_path = selection.get("file_path", "")
            elem_type = selection.get("type", "file")
            elem_name = selection.get("name", "")
            repo_name_from_llm = selection.get("repo_name", "")

            self.logger.debug(f"[SELECTION DEBUG] Processing selection: file_path='{file_path}', type='{elem_type}', name='{elem_name}', repo='{repo_name_from_llm}'")

            # First, detect the correct repo_name from the path
            detected_repo_name = self.path_utils.detect_repo_name_from_path(file_path, known_repos)
            self.logger.debug(f"[SELECTION DEBUG]   Detected repo from path: '{detected_repo_name}'")

            # Find matching candidate using multi-pass strategy:
            # Pass 1: exact/substring path match WITH repo match (most precise)
            # Pass 2: basename match WITH repo match
            # Pass 3: exact/substring path match without repo (fallback)
            # Pass 4: basename match without repo (least precise)
            matching_candidate = None
            actual_repo_name = detected_repo_name or repo_name_from_llm
            target_repo = actual_repo_name

            self.logger.debug(f"[SELECTION DEBUG]   Searching for match: llm_path='{file_path}', target_repo='{target_repo}'")
            for idx_c, candidate in enumerate(candidates):
                if idx_c < 5:
                    self.logger.debug(f"[SELECTION DEBUG]     Candidate[{idx_c}]: path='{candidate.get('file_path', '')}', repo='{candidate.get('repo_name', '')}'")

            # Pass 1: exact/substring path match with repo match
            for candidate in candidates:
                cand_path = candidate.get("file_path", "")
                cand_repo = candidate.get("repo_name", "")
                if target_repo and cand_repo and cand_repo != target_repo:
                    continue
                if file_path == cand_path or file_path in cand_path or cand_path in file_path:
                    matching_candidate = candidate
                    if cand_repo:
                        actual_repo_name = cand_repo
                    self.logger.debug(f"[SELECTION DEBUG]   ✓ Pass1 match (path+repo): llm_path='{file_path}' <-> candidate_path='{cand_path}', repo='{cand_repo}'")
                    break

            # Pass 2: basename match with repo match
            if not matching_candidate:
                for candidate in candidates:
                    cand_path = candidate.get("file_path", "")
                    cand_repo = candidate.get("repo_name", "")
                    if target_repo and cand_repo and cand_repo != target_repo:
                        continue
                    if os.path.basename(file_path) == os.path.basename(cand_path):
                        matching_candidate = candidate
                        if cand_repo:
                            actual_repo_name = cand_repo
                        self.logger.debug(f"[SELECTION DEBUG]   ✓ Pass2 match (basename+repo): llm_path='{file_path}' <-> candidate_path='{cand_path}', repo='{cand_repo}'")
                        break

            # Pass 3: exact/substring path match without repo constraint
            if not matching_candidate:
                for candidate in candidates:
                    cand_path = candidate.get("file_path", "")
                    cand_repo = candidate.get("repo_name", "")
                    if file_path == cand_path or file_path in cand_path or cand_path in file_path:
                        matching_candidate = candidate
                        if cand_repo:
                            actual_repo_name = cand_repo
                        self.logger.debug(f"[SELECTION DEBUG]   ✓ Pass3 match (path only): llm_path='{file_path}' <-> candidate_path='{cand_path}', repo='{cand_repo}'")
                        break

            # Pass 4: basename match without repo constraint (least precise)
            if not matching_candidate:
                for candidate in candidates:
                    cand_path = candidate.get("file_path", "")
                    cand_repo = candidate.get("repo_name", "")
                    if os.path.basename(file_path) == os.path.basename(cand_path):
                        matching_candidate = candidate
                        if cand_repo:
                            actual_repo_name = cand_repo
                        self.logger.debug(f"[SELECTION DEBUG]   ✓ Pass4 match (basename only): llm_path='{file_path}' <-> candidate_path='{cand_path}', repo='{cand_repo}'")
                        break

            if not matching_candidate:
                self.logger.debug(f"[SELECTION DEBUG] No matching candidate found for selection: {file_path} (detected_repo: {detected_repo_name})")
                continue

            # Now normalize with the correct repo_name using the CANDIDATE's path (not LLM's potentially hallucinated path)
            candidate_file_path = matching_candidate.get("file_path", "")
            normalized_path = self.path_utils.normalize_path_with_repo(candidate_file_path, actual_repo_name)

            self.logger.debug(f"[SELECTION DEBUG] Normalization: llm_path='{file_path}' -> candidate_path='{candidate_file_path}' -> normalized='{normalized_path}' (repo='{actual_repo_name}')")
            
            if elem_type == "file":
                # Get file-level element using normalized path
                file_elems = self._retrieve_indexed_elements_for_file(
                    actual_repo_name, normalized_path
                )
                
                if file_elems:
                    for elem in file_elems:
                        elem["selection_granularity"] = "file"
                    results.extend(file_elems)
                    self.logger.debug(f"[SELECTION DEBUG]   ✓ Retrieved {len(file_elems)} file-level elements for {normalized_path}")
                else:
                    self.logger.debug(f"[SELECTION DEBUG]   ✗ Could not retrieve file-level element for {actual_repo_name}/{normalized_path}")

            elif elem_type in ["class", "function"] and elem_name:
                # Search for specific class/function element in bm25_elements
                found = False
                if self.bm25_elements:
                    # Try to find matching element by name, type, and path
                    for bm25_elem in self.bm25_elements:
                        # First check: repo_name and type and name must match
                        if (bm25_elem.repo_name != actual_repo_name or 
                            bm25_elem.type != elem_type or 
                            bm25_elem.name != elem_name):
                            continue
                        
                        # Second check: path matching using normalized path
                        elem_path = bm25_elem.relative_path
                        
                        # Direct match with normalized path
                        if elem_path == normalized_path:
                            self.logger.debug(f"[SELECTION DEBUG]   ✓ Found exact match: {elem_type} '{elem_name}' in {elem_path}")
                            results.append({
                                "element": bm25_elem.to_dict(),
                                "semantic_score": 0.0,
                                "keyword_score": 0.0,
                                "pseudocode_score": 0.0,
                                "graph_score": 0.0,
                                "total_score": 0.75,
                                "agent_found": True,
                                "selection_granularity": elem_type
                            })
                            found = True
                            break

                if not found:
                    self.logger.warning(
                        f"[SELECTION DEBUG]   ✗ Could not find {elem_type} '{elem_name}' in {normalized_path}, falling back to file-level"
                    )
                    # Fallback: use file-level element
                    file_elems = self._retrieve_indexed_elements_for_file(
                        actual_repo_name, normalized_path
                    )

                    if file_elems:
                        for elem in file_elems:
                            elem["selection_granularity"] = "file"
                        results.extend(file_elems)
                        self.logger.debug(f"[SELECTION DEBUG]   ✓ Fallback: Retrieved {len(file_elems)} file-level elements")
                    else:
                        self.logger.debug(f"[SELECTION DEBUG]   ✗ Fallback also failed for {actual_repo_name}/{normalized_path}")

        # Final summary of conversion
        self.logger.debug(f"[SELECTION DEBUG] ===== CONVERSION COMPLETE =====")
        self.logger.debug(f"[SELECTION DEBUG] {len(candidates)} candidates with {len(selections)} selections → {len(results)} elements")
        self.logger.debug(f"[SELECTION DEBUG] ===== ALL CONVERTED ELEMENTS ({len(results)}) =====")
        for i, res_data in enumerate(results):
            elem = res_data.get("element", {})
            path = elem.get("relative_path", elem.get("file_path", "N/A"))
            elem_type = elem.get("type", "N/A")
            repo = elem.get("repo_name", "N/A")
            granularity = res_data.get("selection_granularity", "N/A")
            self.logger.debug(f"[SELECTION DEBUG]   [{i}] repo='{repo}' | path='{path}' | type={elem_type} | granularity={granularity}")
        self.logger.debug(f"[SELECTION DEBUG] ===== END CONVERTED ELEMENTS =====")

        return results
    
    def _fallback_file_selection(self, candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fallback: select files when element selection fails"""
        results = []
        
        for candidate in candidates[:10]:
            file_path = candidate.get("file_path", "")
            repo_name = candidate.get("repo_name", "")
            
            # Normalize path before retrieval
            normalized_path = self.path_utils.normalize_path_with_repo(file_path, repo_name)
            file_elems = self._retrieve_indexed_elements_for_file(
                repo_name, normalized_path
            )
            results.extend(file_elems)
        
        return results
    
    def _round_n(self, query: str, current_elements: List[Dict[str, Any]],
                 query_info: Dict[str, Any], round_num: int,
                 dialogue_history: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Round N (2+): Assessment based on current elements

        Returns:
            Dict with:
            - keep_files: List[str] (files to keep)
            - confidence: int (0-100)
            - tool_calls: List[Dict] (if confidence < 95)
        """
        self.logger.info(f"Round {round_num}: Assessment with current elements")

        try:
            # Build prompt for round N (with dialogue_history)
            prompt = self._build_round_n_prompt(query, current_elements, query_info, round_num, dialogue_history)

            # # Don't log full prompt if it's too long
            # if len(prompt) > 5000:
            #     self.logger.info(f"Round {round_num} prompt length: {len(prompt)} chars (truncated)")
            # else:
            #     self.logger.info(f"Round {round_num} prompt: {prompt}")

            # Call LLM
            response = self._call_llm(prompt)

            # Parse response
            result = self._parse_round_n_response(response)

            return result

        except Exception as e:
            self.logger.error(f"Error in round {round_num} LLM call: {e}")
            self.logger.warning(f"Falling back to default assessment (stopping iteration)")

            # Return a fallback result that will stop iteration
            return {
                "keep_files": [],
                "confidence": 85,  # High enough to suggest we have good results
                "reasoning": f"LLM call failed in round {round_num}, using current results",
                "tool_calls": []  # No more tool calls
            }

    def _build_round_n_prompt(self, query: str, current_elements: List[Dict[str, Any]],
                              query_info: Dict[str, Any], round_num: int,
                              dialogue_history: Optional[List[Dict[str, Any]]] = None) -> str:
        """Build prompt for round N assessment with cost awareness and dialogue context"""

        # Get repository structure
        selected_repos = query_info.get("selected_repos", [])
        repo_structure = self._generate_directory_tree(selected_repos) if selected_repos else "Not available"

        # Format current elements with metadata
        elements_text = self._format_elements_with_metadata(current_elements)
        tool_history_text = self._format_tool_call_history(round_num)
        
        # Calculate current resource usage
        total_lines = self._calculate_total_lines(current_elements)
        remaining_budget = self.adaptive_line_budget - total_lines
        remaining_iterations = self.max_iterations - round_num
        budget_usage_pct = (total_lines / self.adaptive_line_budget) * 100
        
        # Confidence rules for round N
        confidence_rules = f"""
CONFIDENCE SCORING RULES (0-100) for Round {round_num}:
- 95-100: Current files provide complete information to answer the query accurately
- 80-94: Files are mostly sufficient, minor details might be missing
- 60-79: Files provide good foundation but key implementations or connections are missing
- 40-59: Files are relevant but substantial information gaps exist
- 20-39: Files are only partially relevant, need significant additional context
- 0-19: Current files are insufficient or off-target

Base your confidence on:
1. Coverage of key concepts mentioned in the query
2. Presence of relevant signatures, classes, functions
3. Completeness of call chains or dependency relationships
4. Whether graph-related files fill important gaps

**IMPORTANT: Balance confidence with cost efficiency**
"""
        
        # Resource status
        resource_status = f"""
**Current Resource Usage**:
- Current code lines: {total_lines} / {self.adaptive_line_budget} ({budget_usage_pct:.1f}% used)
- Remaining budget: {remaining_budget} lines
- Current round: {round_num} / {self.max_iterations}
- Remaining iterations: {remaining_iterations}
"""
        
        # Cost-aware guidelines
        cost_guidelines = """
**Cost-Aware Decision Making**:
1. **File Selection**:
   - Only remove irrelevant, redundant, or not useful files
   - Prefer class/function-level selections over entire files when possible, but select the entire file if multiple classes or functions within it are useful

2. **Confidence vs Cost Trade-off**:
   - If budget usage > 70%: Be very selective, only keep essential files
   - If budget usage > 85%: Only keep files critical for answering the query
   - If remaining_budget < 2000 lines: Do NOT request more tool calls unless critical gaps exist

3. **Stopping Criteria** (when to set confidence >= 95):
   - You have enough information to answer the query reasonably well
   - Additional files would provide diminishing returns
   - Budget is running low and current files are sufficient
   - Marginal benefit of more code < cost of retrieving it

4. **Tool Call Efficiency** (when confidence < 95):
   - Only request tool calls if they will find CRITICAL missing information
   - Be very specific to minimize noise
   - Do NOT repeat previous tool calls; use new terms/paths only
   - Consider if the information gap is worth the cost
"""

        # Build dialogue history context if available
        dialogue_context = ""
        if dialogue_history and len(dialogue_history) > 0:
            dialogue_context = "\n**Previous Conversation Context**:\n"
            for idx, turn in enumerate(dialogue_history[-10:], 1):  # Last 10 turns
                turn_query = turn.get("query", "")
                turn_summary = turn.get("summary", "")
                dialogue_context += f"\nTurn {idx}:\n"
                dialogue_context += f"  Query: {turn_query}\n"
                if turn_summary:
                    # Extract key information from summary (truncate for Round N to save tokens)
                    summary_preview = turn_summary[:]
                    dialogue_context += f"  Summary: {summary_preview}\n"
            dialogue_context += "\n**NOTE**: The current query may reference entities from previous turns. Use this context to understand what the user is asking about.\n"

        prompt = f"""You are a cost-aware code analysis agent in round {round_num} of iterative retrieval.
{dialogue_context}
**Current User Query**: {query}

**Repository Structure**:
{repo_structure}

{resource_status}

**Current Retrieved Elements**:
{elements_text}

**Previous Tool Calls**:
{tool_history_text}

{confidence_rules}

{cost_guidelines}

**Your Task**:
1. **Filter**: Keep files that are relevant to answering the query. If all files are potentially useful, keep all.
2. **Assess confidence**: Based on the kept files, how confident are you in answering the query?
3. **Decide on next action**:
   - If confidence >= {self.confidence_threshold} OR budget is critical: STOP (set confidence >= 95)
   - If critical information is missing AND budget allows: Request targeted tool calls
   - Otherwise: STOP with current files

**Output Format** (JSON only):

If stopping (confidence >= {self.confidence_threshold} or budget critical):
{{
  "keep_files": ["file1.py", "file2.py"],
  "confidence": <0-100>,
  "reasoning": "Brief explanation of why these files are sufficient"
}}

If continuing (confidence < {self.confidence_threshold} and budget available):
{{
  "keep_files": ["file1.py", "file2.py"],
  "confidence": <0-100>,
  "reasoning": "Brief explanation of what's missing",
  "tool_calls": [
    {{"tool": "search_codebase", "parameters": {{"search_term": "...", "file_pattern": "*.py", "use_regex": false}}}},
    {{"tool": "list_directory", "parameters": {{"path": "src/core"}}}}
  ]
}}

**Keep Files Format**:
- Filename for file-level: "path/to/file.py"
- Class-level: "path/to/file.py:ClassName"
- Function-level: "path/to/file.py:function_name"

**Tool Call Guidelines**:
- Use search_codebase for finding specific terms, classes, functions
  * search_term: literal text or regex pattern to find in file contents
  * file_pattern: SINGLE glob pattern per tool call to filter files (only one pattern allowed)
    * Format: "RepoName/actual_source_path/**/*.ext" (e.g., "django/django/**/*.py")
    * For all repos: omit repo prefix, use "**/*.ext"
  * use_regex: true if search_term is regex, false for literal (default: false)

- Use list_directory to explore directory structure
  * path: "RepoName/path/to/dir" format (e.g., "django/django/core")
  * For repo root: use "RepoName" (e.g., "django")

  **Note**: Repos often nest project folders (django/django/, flask/src/flask/). Always include the full path from repo root, not just the inner folder.

- Do NOT use the model's native tool_calls format. Instead, include tool call instructions in your text response content in a parseable format

**CRITICAL**:
- Respond with valid JSON only
- No markdown blocks
- No comments in JSON
- Be cost-conscious: fewer, more relevant files are better than many marginally useful files"""
        
        return prompt

    def _record_tool_calls(self, round_num: int, tool_calls: List[Dict[str, Any]],
                           selected_repos: Optional[List[str]] = None) -> None:
        """Record tool calls for prompt context in later rounds.

        This method records the RESOLVED paths (not raw LLM output) to ensure
        accurate deduplication across rounds. The path resolution logic mirrors
        what happens in _execute_search_codebase and _execute_list_directory.

        Args:
            round_num: Current iteration round number
            tool_calls: List of tool calls from LLM
            selected_repos: List of selected repository names for path resolution
        """
        if not tool_calls:
            return
        for tool_call in tool_calls:
            tool_name = tool_call.get("tool", "")
            parameters = tool_call.get("parameters", {})

            # Resolve parameters to match what actually gets executed
            resolved_parameters = self._resolve_tool_call_parameters(
                tool_name, parameters, selected_repos
            )

            self.logger.debug(f"[TOOL CALL RECORD] Round {round_num}: Recording tool call: {tool_name} with resolved parameters: {resolved_parameters}, original parameters: {parameters}, selected_repos: {selected_repos}")

            # remove _resolved_repo from parameters before recording, it's only for internal use
            resolved_parameters.pop("_resolved_repo", None)
            
            tool_call_record = {
                "round": round_num,
                "tool": tool_name,
                "parameters": resolved_parameters
            }
            
            self.logger.debug(f"[TOOL CALL RECORD] Final tool call record to append: {tool_call_record}")
            
            self.tool_call_history.append(tool_call_record)

    def _resolve_tool_call_parameters(self, tool_name: str, parameters: Dict[str, Any],
                                      selected_repos: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Resolve tool call parameters to their actual executed values.

        This mirrors the path resolution logic in _execute_search_codebase and
        _execute_list_directory to ensure recorded parameters match what's executed.

        Args:
            tool_name: Name of the tool (search_codebase or list_directory)
            parameters: Raw parameters from LLM
            selected_repos: List of selected repository names

        Returns:
            Resolved parameters dict with actual paths that will be used during execution
        """
        self.logger.debug(f"[RESOLVE] _resolve_tool_call_parameters called: tool={tool_name}, params={parameters}, repos={selected_repos}")

        resolved = dict(parameters)  # Make a copy to avoid mutating original
        is_single_repo = selected_repos and len(selected_repos) == 1

        if tool_name == "search_codebase":
            file_pattern = parameters.get("file_pattern", "*")
            root_path_hint = parameters.get("root_path", None)
            self.logger.debug(f"[RESOLVE] search_codebase: file_pattern='{file_pattern}', root_path_hint='{root_path_hint}'")

            # Determine target repo (mirrors logic in _execute_search_codebase)
            target_repo = None

            # Try to detect target repo from root_path_hint
            if selected_repos and root_path_hint and root_path_hint != ".":
                norm_path = root_path_hint.replace("\\", "/")
                for repo in selected_repos:
                    if norm_path.lower() == repo.lower() or norm_path.lower().startswith(repo.lower() + "/"):
                        target_repo = repo
                        self.logger.debug(f"[RESOLVE] Detected repo from root_path_hint: '{root_path_hint}' -> repo='{repo}'")
                        break

            # Try to detect target repo from file_pattern if not yet determined
            if selected_repos and target_repo is None and file_pattern and file_pattern != "*":
                for repo in selected_repos:
                    result = self.path_utils.validate_and_normalize_file_pattern(file_pattern, repo)
                    self.logger.debug(f"[RESOLVE] validate_and_normalize_file_pattern('{file_pattern}', '{repo}') -> {result}")
                    if result:
                        targets_repo, normalized_pattern = result
                        if targets_repo:
                            target_repo = repo
                            self.logger.debug(f"[RESOLVE] Detected repo from file_pattern: '{file_pattern}' -> repo='{repo}'")
                            break

            # Apply single-repo fallback
            if is_single_repo:
                target_repo = selected_repos[0]
                self.logger.debug(f"[RESOLVE] Single-repo fallback: target_repo='{target_repo}'")

            # Resolve root_path if we have a target repo
            if target_repo and root_path_hint and root_path_hint != ".":
                resolved_root = self.path_utils.resolve_repo_target_path(target_repo, root_path_hint)
                resolved["root_path"] = resolved_root
                self.logger.debug(f"[RESOLVE] Resolved root_path: '{root_path_hint}' -> '{resolved_root}'")
            elif target_repo:
                resolved["root_path"] = target_repo
                self.logger.debug(f"[RESOLVE] Using repo as root_path: '{target_repo}'")

            # Normalize file_pattern to canonical form using resolve_repo_target_path
            # on the directory portion, so it always starts with the repo prefix.
            if target_repo:
                current_pattern = file_pattern  # Always use original, not validate_and_normalize output
                norm_pattern = current_pattern.replace("\\", "/")

                # Split into directory part and glob part
                glob_chars = ('*', '?', '[')
                first_glob_idx = len(norm_pattern)
                for gc in glob_chars:
                    idx = norm_pattern.find(gc)
                    if idx != -1 and idx < first_glob_idx:
                        first_glob_idx = idx

                if first_glob_idx < len(norm_pattern):
                    # Find the last '/' before the first glob char to split dir/glob
                    last_slash = norm_pattern[:first_glob_idx].rfind('/')
                    if last_slash >= 0:
                        dir_part = norm_pattern[:last_slash]
                        glob_part = norm_pattern[last_slash + 1:]
                    else:
                        dir_part = ""
                        glob_part = norm_pattern
                else:
                    # No glob chars — entire pattern is a path
                    dir_part = norm_pattern
                    glob_part = ""

                if dir_part:
                    resolved_dir = self.path_utils.resolve_repo_target_path(target_repo, dir_part)
                    resolved["file_pattern"] = resolved_dir + "/" + glob_part if glob_part else resolved_dir
                else:
                    # Pure glob (e.g. "*.py"), prepend repo
                    resolved["file_pattern"] = target_repo + "/" + glob_part
                self.logger.debug(f"[RESOLVE] Resolved file_pattern: '{file_pattern}' -> '{resolved['file_pattern']}'")

            # Record the resolved target repo for logging
            if target_repo:
                resolved["_resolved_repo"] = target_repo

        elif tool_name == "list_directory":
            raw_path = parameters.get("path", ".")
            self.logger.debug(f"[RESOLVE] list_directory: raw_path='{raw_path}'")

            # Determine target repo (mirrors logic in _execute_list_directory)
            target_repo = None

            # Try to detect target repo from path
            if selected_repos and raw_path and raw_path != ".":
                norm_path = raw_path.replace("\\", "/")
                for repo in selected_repos:
                    if norm_path.lower() == repo.lower() or norm_path.lower().startswith(repo.lower() + "/"):
                        target_repo = repo
                        self.logger.debug(f"[RESOLVE] Detected repo from path: '{raw_path}' -> repo='{repo}'")
                        break

            # Apply single-repo fallback
            if is_single_repo:
                target_repo = selected_repos[0]
                self.logger.debug(f"[RESOLVE] Single-repo fallback: target_repo='{target_repo}'")

            # Resolve path if we have a target repo
            if target_repo:
                resolved_path = self.path_utils.resolve_repo_target_path(target_repo, raw_path)
                resolved["path"] = resolved_path
                resolved["_resolved_repo"] = target_repo
                self.logger.debug(f"[RESOLVE] Resolved path: '{raw_path}' -> '{resolved_path}'")

        self.logger.debug(f"[RESOLVE] Final resolved parameters: {resolved}")
        return resolved

    def _normalize_tool_call(self, tool_call: Dict[str, Any]) -> str:
        """Normalize a tool call for deduplication."""
        tool_name = tool_call.get("tool", "")
        parameters = tool_call.get("parameters", {})
        try:
            params_text = json.dumps(parameters, ensure_ascii=True, sort_keys=True, separators=(",", ":"))
        except TypeError:
            params_text = json.dumps(str(parameters), ensure_ascii=True)
        return f"{tool_name}:{params_text}"

    def _filter_redundant_tool_calls(self, round_num: int,
                                     tool_calls: List[Dict[str, Any]],
                                     selected_repos: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Filter out tool calls already used in earlier rounds.

        Args:
            round_num: Current round number
            tool_calls: List of tool calls from LLM (with raw/unresolved paths)
            selected_repos: List of selected repository names for path resolution

        Returns:
            Filtered list of tool calls (non-duplicates)
        """
        if not tool_calls:
            return []

        prior_calls = {
            self._normalize_tool_call(entry)
            for entry in self.tool_call_history
            if entry.get("round", 0) < round_num
        }
        seen_current = set()
        filtered = []
        removed_prior = 0
        removed_current = 0

        for tool_call in tool_calls:
            # Resolve the tool call parameters before comparison to match recorded history
            tool_name = tool_call.get("tool", "")
            parameters = tool_call.get("parameters", {})
            resolved_parameters = self._resolve_tool_call_parameters(
                tool_name, parameters, selected_repos
            )
            resolved_tool_call = {"tool": tool_name, "parameters": resolved_parameters}

            key = self._normalize_tool_call(resolved_tool_call)
            if key in prior_calls:
                removed_prior += 1
                continue
            if key in seen_current:
                removed_current += 1
                continue
            seen_current.add(key)
            filtered.append(tool_call)  # Return original tool_call, not resolved

        if removed_prior:
            self.logger.info(
                f"Filtered {removed_prior} duplicate tool calls from earlier rounds"
            )
        if removed_current:
            self.logger.info(
                f"Filtered {removed_current} duplicate tool calls within round {round_num}"
            )

        return filtered

    def _format_tool_call_history(self, current_round: int) -> str:
        """Format tool call history up to the previous round."""
        if not self.tool_call_history:
            return "None"

        lines = []
        for entry in self.tool_call_history:
            if entry.get("round", 0) >= current_round:
                continue
            tool_name = entry.get("tool", "")
            params = entry.get("parameters", {})
            params_text = json.dumps(params, ensure_ascii=True, sort_keys=True)
            lines.append(f"- Round {entry['round']}: {tool_name} {params_text}")

        return "\n".join(lines) if lines else "None"
    
    def _format_elements_with_metadata(self, elements: List[Dict[str, Any]]) -> str:
        """Format elements with metadata for round N prompt"""
        lines = []

        # Group by (repo_name, relative_path) to avoid conflicts when multiple repos have same file names
        file_groups = {}
        for elem_data in elements:
            elem = elem_data.get("element", {})
            repo_name = elem.get("repo_name", "")
            relative_path = elem.get("relative_path", elem.get("file_path", ""))

            # Use (repo_name, relative_path) as key for unique grouping
            group_key = (repo_name, relative_path)

            if group_key not in file_groups:
                file_groups[group_key] = []
            file_groups[group_key].append(elem_data)

        for i, (group_key, elem_list) in enumerate(file_groups.items(), 1):
            repo_name, relative_path = group_key

            # Display as "repo_name/relative_path" for clarity
            display_path = f"{repo_name}/{relative_path}" if repo_name else relative_path
            lines.append(f"\n{i}. {display_path}")

            # Aggregate metadata
            sources = set()
            related_to = set()
            total_lines = 0

            for elem_data in elem_list:
                elem = elem_data.get("element", {})
                
                # Determine source
                if elem_data.get("agent_found"):
                    sources.add("Tool")
                elif elem_data.get("llm_file_selected"):
                    sources.add("LLM Selection")
                elif elem_data.get("related_to"):
                    sources.add("Graph")
                    related_to.add(elem_data.get("related_to", ""))
                else:
                    sources.add("Retrieval")
                
                # Calculate lines
                start = elem.get("start_line", 0)
                end = elem.get("end_line", 0)
                if end > start:
                    total_lines += (end - start + 1)
            
            if repo_name:
                lines.append(f"   Repo: {repo_name}")
            lines.append(f"   Type: {elem_list[0]['element'].get('type', 'unknown')}")
            lines.append(f"   Source: {', '.join(sources)}")
            if total_lines > 0:
                lines.append(f"   Lines: {total_lines}")
            if related_to:
                lines.append(f"   Related to: {', '.join(related_to)}")
            
            # Show signatures for class/function elements
            for elem_data in elem_list[:5]:
                elem = elem_data.get("element", {})
                if elem.get("signature"):
                    lines.append(f"   - {elem['signature']}")
        
        return "\n".join(lines)
    
    def _parse_round_n_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response from round N"""
        try:
            json_str = self._extract_json_from_response(response)
            data = self._robust_json_parse(json_str)

            result = {
                "keep_files": data.get("keep_files", []),
                "confidence": data.get("confidence", 50),
                "reasoning": data.get("reasoning", ""),
                "tool_calls": data.get("tool_calls", [])
            }

            # DEBUG: Log tool calls to verify LLM prompt compliance
            if result["tool_calls"]:
                self.logger.debug(f"[DEBUG] Round N LLM tool_calls ({len(result['tool_calls'])} calls):")
                for idx, tc in enumerate(result["tool_calls"]):
                    tool_name = tc.get("tool", "unknown")
                    params = tc.get("parameters", {})
                    self.logger.info(f"  [{idx+1}] {tool_name}: {params}")
                    # Check prompt compliance for path format
                    if tool_name == "list_directory":
                        path = params.get("path", "")
                        self.logger.info(f"    -> list_directory path='{path}' (check if repo prefix included for multi-repo)")
                    elif tool_name == "search_codebase":
                        root_path = params.get("root_path", "not_specified")
                        search_term = params.get("search_term", "")
                        self.logger.info(f"    -> search_codebase root_path='{root_path}', search_term='{search_term}'")

            # DEBUG: Log keep_files decision
            if result["keep_files"]:
                self.logger.debug(f"[DEBUG] Round N keep_files ({len(result['keep_files'])} files): {result['keep_files']}...")

            return result

        except Exception as e:
            self.logger.error(f"Failed to parse round N response: {e}")
            self.logger.debug(f"Response content (first 500 chars): {response[:500]}")
            # Fallback: keep all files, medium confidence
            return {
                "keep_files": [],
                "confidence": 50,
                "reasoning": "Parse error",
                "tool_calls": []
            }
    
    def _filter_elements_by_keep_files(self, elements: List[Dict[str, Any]],
                                       keep_files: List[str]) -> List[Dict[str, Any]]:
        """
        Filter elements based on keep_files list with intelligent prioritization

        Strategy:
        1. Filter based on LLM's keep_files decisions
        2. Calculate relevance score for each element
        3. If exceeding line budget, prioritize by relevance/cost ratio
        """
        self.logger.debug(f"[FILTER DEBUG] ========== STARTING FILTER PROCESS ==========")
        self.logger.debug(f"[FILTER DEBUG] Input: {len(elements)} elements, {len(keep_files)} keep_files")

        # Print ALL keep_files (no truncation)
        self.logger.debug(f"[FILTER DEBUG] ===== ALL KEEP_FILES ({len(keep_files)}) =====")
        for i, kf in enumerate(keep_files):
            self.logger.debug(f"[FILTER DEBUG]   [{i}] '{kf}'")
        self.logger.debug(f"[FILTER DEBUG] ===== END KEEP_FILES =====")

        # Print ALL elements (no truncation)
        self.logger.debug(f"[FILTER DEBUG] ===== ALL INPUT ELEMENTS ({len(elements)}) =====")
        for i, elem_data in enumerate(elements):
            elem = elem_data.get("element", {})
            file_path = elem.get("file_path", "N/A")
            relative_path = elem.get("relative_path", "N/A")
            repo_name = elem.get("repo_name", "N/A")
            elem_type = elem.get("type", "N/A")
            elem_name = elem.get("name", "N/A")
            self.logger.debug(f"[FILTER DEBUG]   [{i}] repo='{repo_name}' | file_path='{file_path}' | relative_path='{relative_path}' | type={elem_type} | name='{elem_name}'")
        self.logger.debug(f"[FILTER DEBUG] ===== END INPUT ELEMENTS =====")

        if not keep_files:
            # No filtering specified, apply smart pruning based on budget
            self.logger.debug(f"[FILTER DEBUG] No keep_files specified, applying smart pruning")
            return self._smart_prune_elements(elements)

        # Step 1: Basic filtering by keep_files
        self.logger.debug(f"[FILTER DEBUG] ===== STARTING MATCHING PROCESS =====")
        filtered = []
        not_matched_elements = []
        matched_pairs = []

        for idx, elem_data in enumerate(elements):
            elem = elem_data.get("element", {})
            elem_type = elem.get("type", "")
            elem_name = elem.get("name", "")
            
            repo_name = elem.get("repo_name", "")
            relative_path = elem.get("relative_path", elem.get("file_path", ""))
            
            # Construct full path with repo for matching
            file_path = f"{repo_name}/{relative_path}" if repo_name else relative_path

            self.logger.debug(f"[FILTER DEBUG] Checking element [{idx}]: path='{file_path}', type='{elem_type}', name='{elem_name}'")

            # Check if this element should be kept
            matched = False
            matched_with = None
            for keep_item in keep_files:
                # Simple filename match
                if keep_item in file_path:
                    self.logger.debug(f"[FILTER DEBUG]   ✓ MATCHED (filename): keep_item='{keep_item}' found in file_path='{file_path}'")
                    filtered.append(elem_data)
                    matched = True
                    matched_with = keep_item
                    break
                # Class-level match: "filename:ClassName"
                if ":" in keep_item:
                    keep_file, keep_name = keep_item.split(":", 1)
                    if keep_file in file_path and elem_name == keep_name:
                        self.logger.debug(f"[FILTER DEBUG]   ✓ MATCHED (class/function): keep_item='{keep_item}' matched file_path='{file_path}' and name='{elem_name}'")
                        filtered.append(elem_data)
                        matched = True
                        matched_with = keep_item
                        break

            if matched:
                matched_pairs.append((idx, file_path, matched_with))
            else:
                not_matched_elements.append((idx, file_path, elem_type, elem_name))
                self.logger.debug(f"[FILTER DEBUG]   ✗ NOT MATCHED: file_path='{file_path}', name='{elem_name}'")

        self.logger.debug(f"[FILTER DEBUG] ===== END MATCHING PROCESS =====")
        self.logger.debug(f"[FILTER DEBUG] Matched: {len(matched_pairs)}, Not matched: {len(not_matched_elements)}")

        # Print summary of matches
        self.logger.debug(f"[FILTER DEBUG] ===== MATCHED ELEMENTS ({len(matched_pairs)}) =====")
        for idx, path, keep_item in matched_pairs:
            self.logger.debug(f"[FILTER DEBUG]   [{idx}] '{path}' ← matched by keep_file '{keep_item}'")
        self.logger.debug(f"[FILTER DEBUG] ===== END MATCHED ELEMENTS =====")

        # Print summary of non-matches
        self.logger.debug(f"[FILTER DEBUG] ===== NOT MATCHED ELEMENTS ({len(not_matched_elements)}) =====")
        for idx, path, elem_type, elem_name in not_matched_elements:
            self.logger.debug(f"[FILTER DEBUG]   [{idx}] '{path}' (type={elem_type}, name='{elem_name}')")
        self.logger.debug(f"[FILTER DEBUG] ===== END NOT MATCHED ELEMENTS =====")

        self.logger.debug(f"[FILTER DEBUG] Initial filtering: {len(elements)} -> {len(filtered)} elements")

        if len(filtered) == 0 and len(elements) > 0:
            self.logger.error(f"[FILTER DEBUG] ========== ERROR: ALL ELEMENTS FILTERED OUT ==========")
            self.logger.error(f"[FILTER DEBUG] All {len(elements)} elements were filtered out!")
            self.logger.error(f"[FILTER DEBUG] This indicates a path mismatch between keep_files and element paths")
            self.logger.error(f"[FILTER DEBUG] ALL keep_files (sorted): {sorted(keep_files)}")
            self.logger.error(f"[FILTER DEBUG] ALL element paths (sorted): {sorted([e.get('element', {}).get('relative_path', e.get('element', {}).get('file_path', 'N/A')) for e in elements])}")
            self.logger.error(f"[FILTER DEBUG] ========== END ERROR ==========")


        # Step 2: Check if we need to further prune due to budget
        total_lines = self._calculate_total_lines(filtered)
        if total_lines > self.adaptive_line_budget:
            self.logger.debug(f"[FILTER DEBUG] Applying smart pruning: {total_lines} lines > {self.adaptive_line_budget} budget")
            filtered = self._smart_prune_elements(filtered)

        # Final summary with ALL filtered elements
        self.logger.debug(f"[FILTER DEBUG] ===== FINAL FILTERED ELEMENTS ({len(filtered)}) =====")
        for i, elem_data in enumerate(filtered):
            elem = elem_data.get("element", {})
            path = elem.get("relative_path", elem.get("file_path", "N/A"))
            elem_type = elem.get("type", "N/A")
            self.logger.debug(f"[FILTER DEBUG]   [{i}] '{path}' (type={elem_type})")
        self.logger.debug(f"[FILTER DEBUG] ===== END FINAL FILTERED ELEMENTS =====")

        self.logger.debug(f"[FILTER DEBUG] ========== FILTER PROCESS COMPLETE ==========")
        self.logger.debug(f"[FILTER DEBUG] Result: {len(elements)} input → {len(filtered)} output")

        return filtered
    
    def _smart_prune_elements(self, elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Intelligently prune elements when exceeding budget
        
        Prioritization factors:
        1. Relevance score (from retrieval)
        2. Source (tool > semantic > graph)
        3. Element type (class/function > file)
        4. Code size (prefer smaller, focused elements)
        """
        if not elements:
            return elements
        
        # Calculate priority score for each element
        scored_elements = []
        for elem_data in elements:
            score = self._calculate_element_priority_score(elem_data)
            scored_elements.append((score, elem_data))
        
        # Sort by score (descending)
        scored_elements.sort(key=lambda x: x[0], reverse=True)
        
        # Select elements until budget is reached
        selected = []
        total_lines = 0
        for score, elem_data in scored_elements:
            elem = elem_data.get("element", {})
            start = elem.get("start_line", 0)
            end = elem.get("end_line", 0)
            elem_lines = end - start + 1 if end > start else 0
            
            # Always keep at least the top element
            if len(selected) == 0 or total_lines + elem_lines <= self.adaptive_line_budget:
                selected.append(elem_data)
                total_lines += elem_lines
            elif len(selected) >= 5:  # Ensure minimum coverage
                break
        
        self.logger.info(
            f"Smart pruning: kept {len(selected)}/{len(elements)} elements, "
            f"{total_lines} lines (budget: {self.adaptive_line_budget})"
        )
        
        return selected
    
    def _calculate_element_priority_score(self, elem_data: Dict[str, Any]) -> float:
        """
        Calculate priority score for an element
        
        Higher score = higher priority
        """
        elem = elem_data.get("element", {})
        
        # Factor 1: Retrieval relevance score (0-1)
        relevance_score = elem_data.get("total_score", 0.5)
        
        # Factor 2: Source bonus
        source_bonus = 0.0
        if elem_data.get("agent_found"):
            source_bonus = 0.3  # Tool-found elements are high priority
        elif elem_data.get("llm_file_selected"):
            source_bonus = 0.2  # LLM selected files
        elif elem_data.get("semantic_score", 0) > 0.7:
            source_bonus = 0.15  # High semantic match
        
        # Factor 3: Element type bonus (focused > broad)
        elem_type = elem.get("type", "")
        if elem_type == "function":
            type_bonus = 0.2
        elif elem_type == "class":
            type_bonus = 0.15
        else:  # file
            type_bonus = 0.0
        
        # Factor 4: Size penalty (prefer smaller, focused elements for efficiency)
        start = elem.get("start_line", 0)
        end = elem.get("end_line", 0)
        elem_lines = end - start + 1 if end > start else 100
        
        # Normalize size (ideal: 50-200 lines)
        if elem_lines < 50:
            size_score = 0.8  # Too small might lack context
        elif elem_lines <= 200:
            size_score = 1.0  # Ideal size
        elif elem_lines <= 500:
            size_score = 0.7  # Acceptable
        else:
            size_score = 0.5  # Large files are less focused
        
        # Factor 5: Selection granularity bonus
        granularity = elem_data.get("selection_granularity", "file")
        if granularity in ["class", "function"]:
            granularity_bonus = 0.15  # Precise selections are valuable
        else:
            granularity_bonus = 0.0
        
        # Combine factors
        priority_score = (
            relevance_score * 0.4 +
            source_bonus +
            type_bonus +
            size_score * 0.2 +
            granularity_bonus
        )
        
        return priority_score
    
    def _execute_tool_calls_round_n(self, query: str, tool_calls: List[Dict[str, Any]],
                                    repo_filter: Optional[List[str]], 
                                    existing_elements: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Execute tool calls in round N and return new elements"""
        new_elements = self._execute_tool_calls_with_selection(query, tool_calls, repo_filter)
        self.logger.info(f"Tool calls found {len(new_elements)} additional elements")
        self.logger.info(f"Tool calls elements list: {self._format_element_list(new_elements)}")

        if not new_elements:
            return []

        # Only keep elements not already present to avoid redundant expansion
        new_elements = self._filter_new_elements(existing_elements, new_elements)
        self.logger.info(f"After filtering against existing ({len(existing_elements)}): {len(new_elements)} elements remain")
        self.logger.info(f"After filtering elements list: {self._format_element_list(new_elements)}")
        if not new_elements:
            self.logger.info("No new unique elements after filtering existing elements")
            return []

        # Expand with graph for new unique elements only
        new_elements = self.retriever._expand_with_graph(new_elements, max_hops=2)
        self.logger.info(f"Graph expansion resulted in {len(new_elements)} elements")
        self.logger.info(f"Graph expansion elements list: {self._format_element_list(new_elements)}")
        # Remove contained elements after graph expansion
        new_elements = self._remove_duplicates_with_containment(new_elements)
        # Final filter against existing elements (avoid duplicates after expansion)
        new_elements = self._filter_new_elements(existing_elements, new_elements)
        # Keep only the most relevant elements to avoid large graph expansions
        new_elements = self._limit_elements_by_relevance(new_elements, max_elements=self.max_elements)
        
        return new_elements

    def _limit_elements_by_relevance(self, elements: List[Dict[str, Any]], max_elements: int) -> List[Dict[str, Any]]:
        """Limit elements to top-N by relevance (total_score)."""
        if not elements or len(elements) <= max_elements:
            return elements
        
        sorted_elements = sorted(
            elements,
            key=lambda x: x.get("total_score", 0.0),
            reverse=True
        )
        self.logger.info(
            f"Limiting graph-expanded elements: {len(elements)} -> {max_elements}"
        )
        return sorted_elements[:max_elements]
    
    def _merge_elements(self, existing: List[Dict[str, Any]], 
                       new: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge and deduplicate element lists"""
        all_elements = existing + new
        return self._remove_duplicates_with_containment(all_elements)

    def _format_element_list(self, elements: List[Dict[str, Any]]) -> str:
        """Format elements as a concise list for logging."""
        if not elements:
            return "[]"
        items = []
        for elem_data in elements:
            elem = elem_data.get("element", {})
            file_path = elem.get("relative_path", elem.get("file_path", ""))
            elem_type = elem.get("type", "")
            elem_name = elem.get("name", "")
            start = elem.get("start_line", "")
            end = elem.get("end_line", "")
            elem_id = elem.get("id", "")
            line_range = f":{start}-{end}" if start and end else ""
            id_str = f" id={elem_id}" if elem_id else ""
            items.append(f"{file_path}({elem_type} {elem_name}{line_range}{id_str})")
        return "[" + ", ".join(items) + "]"

    def _element_identity(self, elem_data: Dict[str, Any]) -> Tuple[str, str, str, str, int, int]:
        """Build a stable identity for element deduplication."""
        elem = elem_data.get("element", {})
        elem_id = elem.get("id")
        if elem_id:
            return ("id", str(elem_id), "", "", 0, 0)
        repo_name = elem.get("repo_name", "")
        file_path = elem.get("relative_path", elem.get("file_path", ""))
        elem_type = elem.get("type", "")
        elem_name = elem.get("name", "")
        start = elem.get("start_line", 0)
        end = elem.get("end_line", 0)
        return (repo_name, file_path, elem_type, elem_name, start, end)

    def _filter_new_elements(self, existing: List[Dict[str, Any]],
                             candidates: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter candidates to only those not present in existing."""
        if not candidates:
            return []
        existing_keys = {self._element_identity(elem) for elem in existing}
        filtered = []
        for elem in candidates:
            key = self._element_identity(elem)
            if key in existing_keys:
                continue
            filtered.append(elem)
        return filtered
    
    def _remove_duplicates_with_containment(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicates considering containment relationships
        
        Strategy:
        1. First remove exact duplicates (same element)
        2. Then remove contained elements (e.g., a function inside a class)
        
        For example:
        - Class A (lines 116-2561) contains Function B (lines 150-200)
        - Keep Class A, remove Function B
        
        Priority order: file > class > function
        """
        # Step 1: Remove exact duplicates using accurate_agent's logic
        results = self._remove_duplicates(results)
        
        if len(results) <= 1:
            return results
        
        # Step 2: Group by (repo_name, file_path)
        file_groups = {}
        for result in results:
            elem = result.get("element", {})
            repo_name = elem.get("repo_name", "")
            file_path = elem.get("relative_path", elem.get("file_path", ""))
            
            key = (repo_name, file_path)
            if key not in file_groups:
                file_groups[key] = []
            file_groups[key].append(result)
        
        # Step 3: Within each file, remove contained elements
        final_results = []
        for key, group in file_groups.items():
            if len(group) == 1:
                final_results.extend(group)
                continue
            
            # Sort by coverage (descending): file > class > function, then by line range
            def get_priority_and_range(result):
                elem = result.get("element", {})
                elem_type = elem.get("type", "")
                start = elem.get("start_line", 0)
                end = elem.get("end_line", 0)
                
                # Type priority: file=3, class=2, function=1
                type_priority = {"file": 3, "class": 2, "function": 1}.get(elem_type, 0)
                
                # Range size (larger is better)
                range_size = end - start + 1 if end > start else 0
                
                return (-type_priority, -range_size, start)
            
            group.sort(key=get_priority_and_range)
            
            # Keep elements that are not contained by others
            kept = []
            for i, result in enumerate(group):
                elem = result.get("element", {})
                start = elem.get("start_line", 0)
                end = elem.get("end_line", 0)
                
                # Check if this element is contained by any previously kept element
                is_contained = False
                for kept_result in kept:
                    kept_elem = kept_result.get("element", {})
                    kept_start = kept_elem.get("start_line", 0)
                    kept_end = kept_elem.get("end_line", 0)
                    
                    # Check containment: kept element contains current element
                    if kept_start <= start and end <= kept_end and (kept_start < start or end < kept_end):
                        is_contained = True
                        self.logger.debug(
                            f"Removing contained element: {elem.get('type')} {elem.get('name')} "
                            f"(L{start}-{end}) contained in {kept_elem.get('type')} "
                            f"{kept_elem.get('name')} (L{kept_start}-{kept_end})"
                        )
                        break
                
                if not is_contained:
                    kept.append(result)
            
            final_results.extend(kept)
        
        removed_count = len(results) - len(final_results)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} contained elements (kept {len(final_results)} independent elements)")
        
        return final_results
    
    def _should_continue_iteration(self, current_round: int, confidence: int,
                                   current_elements: List[Dict[str, Any]],
                                   query_complexity: int) -> bool:
        """
        Decide whether to continue iteration based on intelligent cost-benefit analysis
        
        Considers:
        1. Confidence gap to threshold
        2. Code lines vs budget
        3. Historical confidence gains (ROI) - ADAPTIVE
        4. Stagnation (flatline)
        5. Hard limits
        
        Returns:
            True if should continue, False otherwise
        """
        # Check 1: Confidence already sufficient
        if confidence >= self.confidence_threshold:
            self.logger.info(f"Stopping: confidence {confidence} >= threshold {self.confidence_threshold}")
            return False
        
        # Check 2: Hard iteration limit
        if current_round >= self.max_iterations:
            self.logger.info(f"Stopping: reached max iterations {self.max_iterations}")
            return False
        
        # Check 3: Line budget check
        total_lines = self._calculate_total_lines(current_elements)
        if total_lines >= self.adaptive_line_budget:
            self.logger.info(f"Stopping: exceeded line budget ({total_lines} >= {self.adaptive_line_budget})")
            return False
        
        # Check 4: Adaptive Trend & ROI Analysis (Revised)
        # Logic: 
        # - Confidence Drop -> Continue (Useful info found), unless consecutive drops.
        # - Low ROI (Inefficient Rise) -> Continue, unless consecutive low efficiency.
        # - Stagnation (No change) -> Stop.
        
        if len(self.iteration_history) >= 2:
            current_metrics = self.iteration_history[-1]
            current_gain = current_metrics["confidence_gain"]
            current_roi = current_metrics["roi"]
            
            # 4a. Stagnation Check (Priority: High)
            # If fluctuations are very small (up or down)
            if abs(current_gain) < 1.0: 
                 if len(self.iteration_history) >= 3:
                      prev_gain = self.iteration_history[-2]["confidence_gain"]
                      if abs(prev_gain) < 1.0:
                          self.logger.info("Stopping: Stagnation detected (consecutive small fluctuations).")
                          return False
            
            # Helper to determine if a round was "Low Performance"
            # Low Performance = Significant Drop OR (Small Gain AND Low ROI)
            def is_low_performance(gain, roi, query_comp, curr_conf):
                min_roi = self._get_min_roi_threshold(query_comp, curr_conf)
                # Condition 1: Significant Drop (negative gain)
                # Note: User says drop is "useful info", so we don't treat single drop as "Stop condition".
                # But consecutive drops is bad.
                if gain < -1.0: 
                    return True
                
                # Condition 2: Low Efficiency (Small positive gain AND Low ROI)
                # If gain is small (but > -1) and ROI is low
                if -1.0 <= gain < self.min_confidence_gain and roi < min_roi:
                    return True
                
                return False

            # Check current performance
            current_is_low = is_low_performance(current_gain, current_roi, query_complexity, confidence)
            
            if current_is_low:
                # If current is low, check if previous was also low
                if len(self.iteration_history) >= 3:
                     prev_metrics = self.iteration_history[-2]
                     prev_gain = prev_metrics["confidence_gain"]
                     prev_roi = prev_metrics["roi"]
                     prev_conf = prev_metrics["confidence"]
                     
                     if is_low_performance(prev_gain, prev_roi, query_complexity, prev_conf):
                         self.logger.info(f"Stopping: Consecutive low performance rounds (Current Gain: {current_gain:.1f}, Prev Gain: {prev_gain:.1f}).")
                         return False
                     
                     # Not consecutive, so we continue to explore
                     self.logger.info(f"Continuing: Low performance this round ({current_gain:.1f}), but allowing adaptive exploration.")
                else:
                    # First iterative round resulted in low performance.
                    # We should probably allow one more chance as it's the start of iteration.
                    self.logger.info(f"Continuing: First iterative round low performance ({current_gain:.1f}), allowing exploration.")

        # Check 5: Strict Stagnation (last 3 rounds) - Keep as safeguard
        if len(self.iteration_history) >= 3:
            last_three_confidences = [h["confidence"] for h in self.iteration_history[-3:]]
            if max(last_three_confidences) - min(last_three_confidences) < 2:
                self.logger.info(f"Stopping: confidence stagnation detected ({last_three_confidences})")
                return False
        
        # Check 6: Cost-benefit threshold
        confidence_gap = self.confidence_threshold - confidence
        remaining_line_budget = self.adaptive_line_budget - total_lines
        
        # Estimate if we can close the confidence gap with remaining budget
        estimated_lines_needed = confidence_gap * 100  # Conservative estimate
        
        if estimated_lines_needed > remaining_line_budget * 1.5:
            # Relax budget check if we just had a drop (exploration mode)
            if len(self.iteration_history) >= 2 and self.iteration_history[-1]["confidence_gain"] < 0:
                 self.logger.info("Relaxing budget check due to confidence drop (exploration mode).")
            else:
                self.logger.info(
                    f"Stopping: unlikely to reach target confidence "
                    f"(need ~{estimated_lines_needed} lines, have {remaining_line_budget})"
                )
                return False
        
        # All checks passed, continue iteration
        self.logger.info(
            f"Continuing: confidence_gap={confidence_gap}, "
            f"lines={total_lines}/{self.adaptive_line_budget}, "
            f"round={current_round}/{self.max_iterations}"
        )
        return True
    
    def _calculate_recent_confidence_gain(self) -> float:
        """Calculate confidence gain in the most recent iteration"""
        if len(self.iteration_history) < 2:
            return 0.0
        return self.iteration_history[-1]["confidence"] - self.iteration_history[-2]["confidence"]
    
    def _calculate_recent_lines_added(self) -> int:
        """Calculate lines added in the most recent iteration"""
        if len(self.iteration_history) < 2:
            return 0
        return self.iteration_history[-1]["total_lines"] - self.iteration_history[-2]["total_lines"]
    
    def _get_min_roi_threshold(self, query_complexity: int, current_confidence: int) -> float:
        """
        Get minimum acceptable ROI threshold based on query complexity and current confidence
        
        Returns:
            Minimum ROI (confidence gain per 1000 lines)
        """
        # Base ROI threshold
        base_roi = 2.0  # At least 2% confidence gain per 1000 lines
        
        # Adjust for query complexity (complex queries: accept lower ROI)
        if query_complexity >= 80:
            complexity_factor = 0.5
        elif query_complexity >= 60:
            complexity_factor = 0.7
        else:
            complexity_factor = 1.0
        
        # Adjust for current confidence (higher confidence: demand higher ROI)
        if current_confidence >= 85:
            confidence_factor = 1.5  # Demand more efficiency when close to target
        elif current_confidence >= 70:
            confidence_factor = 1.0
        else:
            confidence_factor = 0.8  # More lenient when starting low
        
        return base_roi * complexity_factor * confidence_factor
    
    def _calculate_repo_factor(self) -> float:
        """
        Calculate repository complexity factor (0.5 - 2.0)
        """
        if not self.repo_stats:
            return 1.0
        
        total_files = self.repo_stats.get("total_files", 100)
        total_classes = self.repo_stats.get("total_classes", 50)
        total_functions = self.repo_stats.get("total_functions", 100)
        avg_file_lines = self.repo_stats.get("avg_file_lines", 200)
        max_depth = self.repo_stats.get("max_depth", 5)
        
        # File count factor (log scale)
        file_factor = np.log10(total_files + 1) / np.log10(1000)
        file_factor = np.clip(file_factor, 0.3, 1.5)
        
        # Complexity factor
        complexity_factor = avg_file_lines / 200
        complexity_factor = np.clip(complexity_factor, 0.5, 2.0)
        
        # Depth factor
        depth_factor = max_depth / 5
        depth_factor = np.clip(depth_factor, 0.7, 1.3)
        
        final_factor = (file_factor + complexity_factor + depth_factor) / 3
        
        return float(np.clip(final_factor, 0.5, 2.0))
    
    def _calculate_total_lines(self, elements: List[Dict[str, Any]]) -> int:
        """Calculate total lines of code in elements"""
        total = 0
        for elem_data in elements:
            elem = elem_data.get("element", {})
            start = elem.get("start_line", 0)
            end = elem.get("end_line", 0)
            if end > start:
                total += (end - start + 1)
        return total
    
    def _call_llm(self, prompt: str) -> str:
        """Call LLM with prompt"""
        self.logger.info(f"Calling LLM: prompt_len={len(prompt)}, max_tokens={self.max_tokens}")

        if self.provider == "openai":
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a precise code analysis agent. Respond in specified format only."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            if not response or not getattr(response, "choices", None):
                raise ValueError(f"Empty response: {response}")

            finish_reason = getattr(response.choices[0], 'finish_reason', 'unknown')
            content = response.choices[0].message.content
            self.logger.info(f"LLM response: content_len={len(content) if content else 0}, finish_reason={finish_reason}")

            if content is None or content == "":
                self.logger.error(f"Empty content: finish_reason={finish_reason}, prompt_len={len(prompt)}")
                raise ValueError("No content in response")

            return content

        elif self.provider == "anthropic":
            response = self.client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                system="You are a precise code analysis agent. Respond in specified format only.",
                messages=[{"role": "user", "content": prompt}]
            )

            if not response or not getattr(response, "content", None):
                raise ValueError(f"Empty response: {response}")

            stop_reason = getattr(response, 'stop_reason', 'unknown')
            text = response.content[0].text if response.content else None
            self.logger.info(f"LLM response: content_len={len(text) if text else 0}, stop_reason={stop_reason}")

            if text is None or text == "":
                self.logger.error(f"Empty content: stop_reason={stop_reason}, prompt_len={len(prompt)}")
                raise ValueError(f"No text in response: {response}")

            return text

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    # ==================== Methods moved from AccurateSearchAgent ====================
    
    def _generate_directory_tree(self, repo_paths: List[str]) -> str:
        """
        Generate a tree-like structure of directories for selected repos
        """
        tree_lines = []
        max_depth = 5
        # Common directories to ignore
        ignore_dirs = {
            '.git', '__pycache__', 'node_modules', 'venv', '.env', 
            'dist', 'build', 'coverage', '.idea', '.vscode', 'target',
            'bin', 'obj', 'out'
        }

        def _add_dir_to_tree(path: str, prefix: str = "", depth: int = 0):
            if depth >= max_depth:
                return

            try:
                # List items
                if not os.path.exists(path) or not os.path.isdir(path):
                    return
                    
                items = os.listdir(path)
                # Filter for directories only and remove ignored ones
                dirs = [d for d in items 
                       if os.path.isdir(os.path.join(path, d)) 
                       and d not in ignore_dirs 
                       and not d.startswith('.')]
                dirs.sort()
                
                for i, d in enumerate(dirs):
                    is_last = (i == len(dirs) - 1)
                    connector = "└── " if is_last else "├── "
                    
                    tree_lines.append(f"{prefix}{connector}{d}/")
                    
                    # Recurse
                    new_prefix = prefix + ("    " if is_last else "│   ")
                    _add_dir_to_tree(os.path.join(path, d), new_prefix, depth + 1)
                    
            except Exception as e:
                # Silently fail for individual directories to keep output clean
                pass

        # Handle case where no specific repos selected (use root)
        if not repo_paths:
            repo_paths = ["."]

        for repo in repo_paths:
            # Resolve path relative to repo_root
            full_repo_path = os.path.join(self.repo_root, repo) if hasattr(self, 'repo_root') else repo
            
            # Add repo header
            tree_lines.append(f"{repo}/")
            
            # Generate tree
            _add_dir_to_tree(full_repo_path)
            
            tree_lines.append("")  # Empty line between repos
        
        return "\n".join(tree_lines).strip()
    
    def _extract_json_from_response(self, response: str) -> str:
        """
        Extract JSON string from LLM response, handling markdown blocks and reasoning text.
        More robust for small models that may generate malformed JSON.
        """
        response = response.strip()
        
        # Remove any leading/trailing non-JSON text that small models sometimes add
        # e.g., "Here's the JSON:", "The response is:", etc.
        prefixes_to_remove = [
            "here's the json:", "here is the json:", "the json is:",
            "json:", "response:", "output:", "result:",
            "here's the response:", "here is the response:",
        ]
        response_lower = response.lower()
        for prefix in prefixes_to_remove:
            if response_lower.startswith(prefix):
                response = response[len(prefix):].strip()
                break
        
        # 1. Try to find markdown code blocks (non-greedy for nested braces)
        json_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # 2. Try to find raw JSON object
            # Find first '{' and matching '}'
            start = response.find("{")
            if start == -1:
                return response
            
            # Find matching closing brace
            brace_count = 0
            end = -1
            for i in range(start, len(response)):
                if response[i] == '{':
                    brace_count += 1
                elif response[i] == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i
                        break
            
            if end == -1:
                # No matching brace found, use rfind as fallback
                end = response.rfind("}")
            
            if start != -1 and end != -1 and end > start:
                json_str = response[start:end+1]
            else:
                return response
        
        # 3. Clean up common issues for small models
        # Replace unescaped newlines in strings with escaped ones
        json_str = self._sanitize_json_string(json_str)
        
        return json_str
    
    def _sanitize_json_string(self, json_str: str) -> str:
        """
        Sanitize JSON string to fix common issues from small models:
        - Remove/escape control characters in strings
        - Fix trailing commas
        - Handle incomplete JSON
        - Fix missing commas between elements
        """
        # Remove or escape control characters (except in already escaped sequences)
        # This is a simplified approach - replace literal newlines/tabs in strings
        cleaned = []
        in_string = False
        escape_next = False
        
        for i, char in enumerate(json_str):
            if escape_next:
                cleaned.append(char)
                escape_next = False
                continue
            
            if char == '\\' and not escape_next:
                # Check if next character forms valid escape sequence
                if i + 1 < len(json_str):
                    next_char = json_str[i + 1]
                    # Valid JSON escape sequences: \", \\, \/, \b, \f, \n, \r, \t, \uXXXX
                    # if next_char in '"\\\/bfnrtu':
                    if next_char in r'"\\/bfnrtu':
                        cleaned.append(char)
                        escape_next = True
                    else:
                        # Invalid escape, add the backslash as-is
                        cleaned.append(char)
                else:
                    # Backslash at end of string
                    cleaned.append(char)
                continue
            
            if char == '"':
                in_string = not in_string
                cleaned.append(char)
                continue
            
            # If we're inside a string and hit a control character, escape it
            if in_string:
                if char == '\n':
                    cleaned.append('\\n')
                elif char == '\r':
                    cleaned.append('\\r')
                elif char == '\t':
                    cleaned.append('\\t')
                elif ord(char) < 32:  # Other control characters
                    # Skip or replace with space
                    cleaned.append(' ')
                else:
                    cleaned.append(char)
            else:
                cleaned.append(char)
        
        result = ''.join(cleaned)
        
        # Remove inline comments (# or //) outside of strings
        result = self._remove_json_comments(result)
        
        # Fix trailing commas before closing braces/brackets
        result = re.sub(r',(\s*[}\]])', r'\1', result)
        
        # Fix missing commas between } and {, ] and [, etc.
        result = re.sub(r'\}(\s*)\{', r'},\1{', result)
        result = re.sub(r'\](\s*)\[', r'],\1[', result)
        result = re.sub(r'\}(\s*)\[', r'},\1[', result)
        result = re.sub(r'\](\s*)\{', r'],\1{', result)
        
        # Fix missing commas between JSON values
        # Only add comma between closing quote/bracket/brace and opening quote
        result = re.sub(r'(["}\]])(\s*)(")', r'\1,\2\3', result)
        # Fix missing comma after boolean/null followed by quote or opening brace/bracket
        result = re.sub(r'\b(true|false|null)(\s*)(["{[])', r'\1,\2\3', result)
        
        return result
    
    def _remove_json_comments(self, json_str: str) -> str:
        """Remove inline comments from JSON string (# or // style)"""
        lines = json_str.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Track if we're inside a string
            in_string = False
            escape_next = False
            cleaned_line = []
            
            i = 0
            while i < len(line):
                char = line[i]
                
                if escape_next:
                    cleaned_line.append(char)
                    escape_next = False
                    i += 1
                    continue
                
                if char == '\\':
                    cleaned_line.append(char)
                    escape_next = True
                    i += 1
                    continue
                
                if char == '"':
                    in_string = not in_string
                    cleaned_line.append(char)
                    i += 1
                    continue
                
                # If not in string, check for comments
                if not in_string:
                    # Check for # comment
                    if char == '#':
                        # Remove everything after # on this line
                        break
                    # Check for // comment
                    if char == '/' and i + 1 < len(line) and line[i + 1] == '/':
                        # Remove everything after // on this line
                        break
                
                cleaned_line.append(char)
                i += 1
            
            cleaned_lines.append(''.join(cleaned_line).rstrip())
        
        return '\n'.join(cleaned_lines)
    
    def _robust_json_parse(self, json_str: str) -> Any:
        """
        Robustly parse JSON with multiple fallback strategies for small model outputs
        
        Args:
            json_str: JSON string to parse
        
        Returns:
            Parsed JSON object
        
        Raises:
            json.JSONDecodeError: If all parsing strategies fail
        """
        import json
        
        # Strategy 1: Direct parsing
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e1:
            self.logger.debug(f"Direct JSON parse failed: {e1}")
        
        # Strategy 2: Parse with sanitization
        try:
            sanitized = self._sanitize_json_string(json_str)
            return json.loads(sanitized)
        except json.JSONDecodeError as e2:
            self.logger.debug(f"Sanitized JSON parse failed: {e2}")
        
        # Strategy 3: Try to fix common JSON errors with regex
        try:
            # Fix unquoted keys
            fixed = re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)(\s*:)', r'\1"\2"\3', json_str)
            return json.loads(fixed)
        except (json.JSONDecodeError, Exception) as e3:
            self.logger.debug(f"Fixed keys JSON parse failed: {e3}")
        
        # Strategy 4: Use ast.literal_eval as safer alternative (can handle Python-style dicts)
        try:
            import ast
            # ast.literal_eval is safer than eval - only evaluates literals
            result = ast.literal_eval(json_str)
            if isinstance(result, (dict, list)):
                return result
        except (ValueError, SyntaxError, Exception) as e4:
            self.logger.debug(f"AST literal_eval parse failed: {e4}")
        
        # Strategy 5: Try to extract and parse just the first complete object
        try:
            # Find first { and try to parse incrementally
            start = json_str.find('{')
            if start != -1:
                for end in range(len(json_str), start, -1):
                    try:
                        subset = json_str[start:end]
                        if subset.count('{') == subset.count('}'):
                            return json.loads(subset)
                    except:
                        continue
        except Exception as e5:
            self.logger.debug(f"Incremental parse failed: {e5}")
        
        # All strategies failed
        raise json.JSONDecodeError("All parsing strategies failed", json_str, 0)

    def _execute_search_codebase(self, parameters: Dict[str, Any],
                                  selected_repos: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Execute search_codebase tool and return file candidates with robust repo detection"""
        search_term = parameters.get("search_term", "")
        file_pattern = parameters.get("file_pattern", "*")
        use_regex = parameters.get("use_regex", False)
        case_sensitive = parameters.get("case_sensitive", False)
        root_path_hint = parameters.get("root_path", None)

        # Auto-detect regex patterns if not explicitly set
        if not use_regex and search_term:
            regex_indicators = [r'\.', r'\*', r'\+', r'\?', r'\[', r'\]', r'\^', r'\$', r'\|', r'\(', r'\)', r'\{', r'\}']
            if any(indicator in search_term for indicator in regex_indicators):
                self.logger.debug(f"[DEBUG] Auto-detected regex pattern in search_term: '{search_term}', setting use_regex=True")
                use_regex = True

        # DEBUG: Log input parameters
        self.logger.debug(f"[DEBUG] _execute_search_codebase called:")
        self.logger.debug(f"  search_term='{search_term}'")
        self.logger.debug(f"  file_pattern='{file_pattern}'")
        self.logger.debug(f"  root_path_hint='{root_path_hint}'")
        self.logger.debug(f"  use_regex={use_regex}")
        self.logger.debug(f"  selected_repos={selected_repos}")
        self.logger.debug(f"  repo_root='{self.repo_root}'")

        if not search_term:
            self.logger.warning("[DEBUG] Empty search_term, returning empty candidates")
            return []

        candidates = []
        is_single_repo = selected_repos and len(selected_repos) == 1

        # Step 1: Intelligent repo detection from root_path_hint and file_pattern
        target_repos = None  # None means "not yet determined"
        adjusted_file_pattern = file_pattern

        # Try to detect target repo from root_path_hint
        if selected_repos and root_path_hint and root_path_hint != ".":
            norm_path = root_path_hint.replace("\\", "/")
            self.logger.debug(f"[DEBUG] Checking if root_path_hint '{norm_path}' targets specific repo from {selected_repos}")
            for repo in selected_repos:
                if norm_path.lower() == repo.lower() or norm_path.lower().startswith(repo.lower() + "/"):
                    self.logger.debug(f"[DEBUG] root_path '{root_path_hint}' explicitly targets repo '{repo}'")
                    target_repos = [repo]
                    break

        # Try to detect target repo from file_pattern if not yet determined
        if selected_repos and target_repos is None and file_pattern and file_pattern != "*":
            norm_pattern = file_pattern.replace("\\", "/")
            self.logger.debug(f"[DEBUG] Checking if file_pattern '{norm_pattern}' targets specific repo from {selected_repos}")
            for repo in selected_repos:
                result = self.path_utils.validate_and_normalize_file_pattern(file_pattern, repo)
                if result:
                    targets_repo, normalized_pattern = result
                    if targets_repo:
                        self.logger.debug(f"[DEBUG] file_pattern '{file_pattern}' explicitly targets repo '{repo}'")
                        target_repos = [repo]
                        adjusted_file_pattern = normalized_pattern
                        if adjusted_file_pattern != norm_pattern:
                            self.logger.debug(f"[DEBUG] Normalized file_pattern: '{file_pattern}' -> '{adjusted_file_pattern}'")
                        break

        # Step 2: Apply single-repo fallback or multi-repo default
        if is_single_repo:
            # Single repo scenario: always use that repo regardless of detection
            target_repos = selected_repos
            self.logger.debug(f"[DEBUG] Single-repo scenario: forcing target_repos to {target_repos}")
            # Keep the normalized pattern from step 1 (if repo prefix was stripped, use stripped version)
            self.logger.debug(f"[DEBUG] Single-repo: using file_pattern '{adjusted_file_pattern}'")
        elif target_repos is None:
            # Multi-repo scenario: detection failed, apply to all selected repos
            target_repos = selected_repos
            self.logger.debug(f"[DEBUG] Multi-repo scenario: no specific repo detected, applying to all {target_repos}")

        # If specific repos are selected (or filtered above), search within each repo
        if target_repos:
            self.logger.debug(f"[DEBUG] Searching in target_repos: {target_repos}")
            for repo_name in target_repos:
                # Use root_path_hint if provided, otherwise default to repo root
                if root_path_hint and root_path_hint != ".":
                    root_path = self.path_utils.resolve_repo_target_path(repo_name, root_path_hint)
                    self.logger.debug(f"[DEBUG] Resolved root_path for repo '{repo_name}': '{root_path_hint}' -> '{root_path}'")
                else:
                    root_path = repo_name
                    self.logger.debug(f"[DEBUG] Using repo root as search path: '{root_path}'")

                # Validate that the path exists
                full_path = os.path.join(self.repo_root, root_path) if hasattr(self, 'repo_root') else root_path
                self.logger.debug(f"[DEBUG] Resolved full path for searching: {full_path}")
                if not os.path.exists(full_path):
                    self.logger.debug(f"[DEBUG] Path does not exist: {full_path}, skipping search in repo {repo_name}")
                    continue

                # Log the exact parameters being passed to search_codebase
                self.logger.debug(f"[DEBUG] Calling search_codebase with:")
                self.logger.info(f"  search_term: '{search_term}'")
                self.logger.info(f"  file_pattern: '{adjusted_file_pattern}'")
                self.logger.info(f"  root_path: '{root_path}'")
                self.logger.info(f"  use_regex: {use_regex}")

                search_result = self.tools.search_codebase(
                    search_term=search_term,
                    file_pattern=adjusted_file_pattern,
                    root_path=root_path,
                    max_results=30,
                    case_sensitive=case_sensitive,
                    use_regex=use_regex
                )

                # Log the raw search result
                self.logger.debug(f"[DEBUG] Raw search_result: success={search_result.get('success')}, "
                                f"num_results={len(search_result.get('results', []))}, "
                                f"error={search_result.get('error', 'None')}")

                if search_result.get("success"):
                    result_count = len(search_result.get("results", []))
                    self.logger.debug(f"[DEBUG] search_codebase returned {result_count} results for repo '{repo_name}'")
                    for match in search_result.get("results", []):
                        file_path = match.get("file", "")
                        match_count = match.get("match_count", 0)

                        # Get file structure summary
                        structure = self.tools.get_file_structure_summary(file_path)

                        # Normalize path using robust normalization
                        normalized_path = self.path_utils.normalize_path_with_repo(file_path, repo_name)
                        self.logger.debug(f"[DEBUG] Normalized path: '{file_path}' -> '{normalized_path}'")

                        # Get indexed class and function elements
                        indexed_elements = self._get_indexed_class_function_elements(repo_name, normalized_path)

                        candidates.append({
                            "file_path": file_path,
                            "match_count": match_count,
                            "structure": structure if structure.get("success") else {},
                            "indexed_elements": indexed_elements,
                            "source": "search_codebase",
                            "repo_name": repo_name
                        })
                else:
                    self.logger.debug(f"[DEBUG] search_codebase failed for repo '{repo_name}': {search_result.get('error', 'unknown')}")
        else:
            # No specific repos selected, search all (use root_path_hint if provided)
            root_path = root_path_hint if root_path_hint and root_path_hint != "." else "."
            
            search_result = self.tools.search_codebase(
                search_term=search_term,
                file_pattern=file_pattern,
                root_path=root_path,
                max_results=30,
                case_sensitive=case_sensitive,
                use_regex=use_regex
            )
            
            if search_result.get("success"):
                for match in search_result.get("results", []):
                    file_path = match.get("file", "")
                    match_count = match.get("match_count", 0)
                    
                    # Get file structure summary
                    structure = self.tools.get_file_structure_summary(file_path)
                    
                    # Get indexed class and function elements (try to extract repo_name from file_path)
                    # If no repo_name in context, assume file_path might contain it or use empty string
                    repo_name_for_lookup = ""
                    normalized_path = file_path
                    # Try to detect repo_name from file_path if it contains "/" 
                    # In multi-repo scenarios, file_path might be "repo_name/path/to/file"
                    if "/" in file_path and self.bm25_elements:
                        # Try to match with any indexed repo_name (case-insensitive)
                        first_part = file_path.split("/")[0]
                        first_lower = first_part.lower()
                        for elem in self.bm25_elements:
                            if elem.repo_name == first_part or elem.repo_name.lower() == first_lower:
                                repo_name_for_lookup = elem.repo_name
                                # Use robust normalization
                                normalized_path = self.path_utils.normalize_path_with_repo(file_path, elem.repo_name)
                                break
                    
                    indexed_elements = self._get_indexed_class_function_elements(repo_name_for_lookup, normalized_path)
                    
                    candidates.append({
                        "file_path": file_path,
                        "match_count": match_count,
                        "structure": structure if structure.get("success") else {},
                        "indexed_elements": indexed_elements,
                        "source": "search_codebase"
                    })
        
        return candidates
    
    def _execute_list_directory(self, parameters: Dict[str, Any],
                                 selected_repos: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Execute list_directory tool and return file candidates with robust repo detection"""
        raw_path = parameters.get("path", ".")

        # DEBUG: Log input parameters
        self.logger.debug(f"[DEBUG] _execute_list_directory called:")
        self.logger.debug(f"  raw_path='{raw_path}'")
        self.logger.debug(f"  selected_repos={selected_repos}")
        self.logger.debug(f"  repo_root='{self.repo_root}'")

        candidates = []
        is_single_repo = selected_repos and len(selected_repos) == 1

        # Step 1: Intelligent repo detection from raw_path
        target_repos = None  # None means "not yet determined"

        # Try to detect target repo from path
        if selected_repos and raw_path and raw_path != ".":
            norm_path = raw_path.replace("\\", "/")
            self.logger.debug(f"[DEBUG] Checking if path '{norm_path}' targets specific repo from {selected_repos}")

            # Check if path explicitly targets a specific repo
            for repo in selected_repos:
                if norm_path.lower() == repo.lower() or norm_path.lower().startswith(repo.lower() + "/"):
                    self.logger.debug(f"[DEBUG] Path '{raw_path}' explicitly targets repo '{repo}'")
                    target_repos = [repo]
                    break

        # Step 2: Apply single-repo fallback or multi-repo default
        if is_single_repo:
            # Single repo scenario: always use that repo regardless of detection
            target_repos = selected_repos
            self.logger.debug(f"[DEBUG] Single-repo scenario: forcing target_repos to {target_repos}")
        elif target_repos is None:
            # Multi-repo scenario: detection failed, apply to all selected repos
            target_repos = selected_repos
            self.logger.debug(f"[DEBUG] Multi-repo scenario: no specific repo detected, applying to all {target_repos}")

        # If specific repos are selected (or filtered above), list within each repo
        if target_repos:
            self.logger.debug(f"[DEBUG] Listing directories in target_repos: {target_repos}")
            for repo_name in target_repos:
                # Combine repo name with path
                self.logger.debug(f"[DEBUG] Processing repo: {repo_name} with path: {raw_path}")

                full_path = self.path_utils.resolve_repo_target_path(repo_name, raw_path)
                self.logger.debug(f"[DEBUG] Resolved full path for listing: '{raw_path}' -> '{full_path}'")

                # Validate that the path exists
                abs_path = os.path.join(self.repo_root, full_path) if hasattr(self, 'repo_root') else full_path
                self.logger.debug(f"[DEBUG] Absolute path to check: {abs_path}")
                if not os.path.exists(abs_path):
                    self.logger.debug(f"[DEBUG] Directory does not exist: {abs_path}, skipping listing for repo {repo_name}")
                    continue

                dir_result = self.tools.list_directory(
                    path=full_path,
                    include_hidden=False
                )

                if dir_result.get("success"):
                    contents = dir_result.get("contents", [])
                    file_count = sum(1 for item in contents if item.get("type") == "file")
                    dir_count = sum(1 for item in contents if item.get("type") == "directory")
                    self.logger.debug(f"[DEBUG] list_directory returned {file_count} files, {dir_count} dirs for repo '{repo_name}'")

                    for item in contents:
                        if item.get("type") == "file":
                            file_path = item.get("path", "")

                            # Get file structure summary
                            structure = self.tools.get_file_structure_summary(file_path)

                            # Normalize path using robust normalization
                            normalized_path = self.path_utils.normalize_path_with_repo(file_path, repo_name)
                            self.logger.debug(f"[DEBUG] Normalized path: '{file_path}' -> '{normalized_path}'")

                            # Get indexed class and function elements
                            indexed_elements = self._get_indexed_class_function_elements(repo_name, normalized_path)

                            candidates.append({
                                "file_path": file_path,
                                "structure": structure if structure.get("success") else {},
                                "indexed_elements": indexed_elements,
                                "source": "list_directory",
                                "repo_name": repo_name
                            })
                else:
                    self.logger.debug(f"[DEBUG] Error listing directory {full_path}: {dir_result.get('error')}")
        else:
            # No specific repos selected, list from path
            dir_result = self.tools.list_directory(
                path=raw_path,
                include_hidden=False
            )
            
            if dir_result.get("success"):
                for item in dir_result.get("contents", []):
                    if item.get("type") == "file":
                        file_path = item.get("path", "")
                        
                        # Get file structure summary
                        structure = self.tools.get_file_structure_summary(file_path)
                        
                        parts = file_path.split("/")
                        detected_repo_name = parts[0] if len(parts) > 1 else ""
                        
                        if detected_repo_name:
                            normalized_path = self.path_utils.normalize_path_with_repo(file_path, detected_repo_name)
                            
                            indexed_elements = self._get_indexed_class_function_elements(detected_repo_name, normalized_path)
                            
                            candidates.append({
                                "file_path": file_path,
                                "structure": structure if structure.get("success") else {},
                                "indexed_elements": indexed_elements,
                                "source": "list_directory",
                                "repo_name": detected_repo_name
                            })
                        else:
                            pass
            else:
                self.logger.warning(f"Error listing directory {raw_path}: {dir_result.get('error')}")
        
        return candidates
    
    def _retrieve_indexed_elements_for_file(self, repo_name: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Retrieve indexed file-level element for a specific file
        Similar to retriever._retrieve_elements_from_files

        Args:
            repo_name: Repository name
            file_path: Normalized file path (without repo prefix)

        Returns:
            List containing only the file-level indexed element
        """
        self.logger.debug(f"[RETRIEVE DEBUG] _retrieve_indexed_elements_for_file called: repo='{repo_name}', path='{file_path}'")

        if not self.bm25_elements:
            self.logger.debug(f"[RETRIEVE DEBUG] No bm25_elements available")
            return []

        results = []

        # Find only file-level element from this file (use exact match to avoid duplicates)
        matches_found = 0
        for elem in self.bm25_elements:
            # if (elem.repo_name == repo_name and
            #     (elem.relative_path == file_path or file_path in elem.relative_path) and
            #     elem.type == "file"):
            if (elem.repo_name == repo_name and
                elem.relative_path == file_path and
                elem.type == "file"):

                self.logger.debug(f"[RETRIEVE DEBUG] MATCH FOUND: elem.relative_path='{elem.relative_path}', elem.type='{elem.type}'")
                matches_found += 1

                results.append({
                    "element": elem.to_dict(),
                    "semantic_score": 0.0,
                    "keyword_score": 0.0,
                    "pseudocode_score": 0.0,
                    "graph_score": 0.0,
                    "total_score": 0.8,  # Agent-found score
                    "agent_found": True
                })

        if matches_found == 0:
            self.logger.debug(f"[RETRIEVE DEBUG] No file-level element found for repo='{repo_name}', path='{file_path}'")
            # Log a few sample paths from bm25_elements for debugging
            if self.bm25_elements:
                sample_paths = [(e.repo_name, e.relative_path, e.type) for e in self.bm25_elements[:10] if e.repo_name == repo_name]
                if sample_paths:
                    self.logger.debug(f"[RETRIEVE DEBUG] Sample paths from bm25_elements for repo '{repo_name}': {sample_paths[:5]}")
                else:
                    self.logger.debug(f"[RETRIEVE DEBUG] No elements found in bm25_elements for repo '{repo_name}' at all")
        else:
            self.logger.debug(f"[RETRIEVE DEBUG] Found {matches_found} file-level elements")

        return results
    
    def _get_indexed_class_function_elements(self, repo_name: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Retrieve indexed class and function elements for a specific file
        
        Args:
            repo_name: Repository name
            file_path: Normalized file path (without repo prefix)
        
        Returns:
            List of class and function elements with signature, docstring, start_line, end_line
        """
        if not self.bm25_elements:
            return []
        
        results = []
        
        # Find class and function elements from this file
        for elem in self.bm25_elements:
            if (elem.repo_name == repo_name and 
                elem.relative_path == file_path and
                elem.type in ["class", "function"]):
                
                # Extract relevant information
                element_info = {
                    "type": elem.type,
                    "name": elem.name,
                    "signature": elem.signature,
                    "docstring": elem.docstring,
                    "start_line": elem.start_line,
                    "end_line": elem.end_line,
                }
                
                # Add type-specific metadata
                if elem.type == "function":
                    element_info["is_method"] = elem.metadata.get("is_method", False)
                    element_info["class_name"] = elem.metadata.get("class_name")
                elif elem.type == "class":
                    element_info["methods"] = elem.metadata.get("methods", [])
                    element_info["bases"] = elem.metadata.get("bases", [])
                
                results.append(element_info)
        
        return results
    
    def _remove_duplicates(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Remove duplicate files from the same repository
        
        Rules:
        1. If file-level result exists for a file, remove class/function-level results from same file
           (since file-level includes everything)
        2. For same type duplicates, keep the one with highest score
        
        Args:
            results: List of results
        
        Returns:
            Deduplicated results
        """
        # First, group by (repo_name, file_path)
        file_groups = {}
        
        for result in results:
            elem = result.get("element", {})
            repo_name = elem.get("repo_name", "")
            file_path = elem.get("relative_path", elem.get("file_path", ""))
            elem_type = elem.get("type", "")
            
            key = (repo_name, file_path)
            
            if key not in file_groups:
                file_groups[key] = []
            file_groups[key].append(result)
        
        # Process each file group
        deduplicated = []
        
        for key, group in file_groups.items():
            # Check if there's a file-level result
            has_file_level = any(r.get("element", {}).get("type") == "file" for r in group)
            
            if has_file_level:
                # If file-level exists, only keep file-level results (highest score if multiple)
                file_level_results = [r for r in group if r.get("element", {}).get("type") == "file"]
                if file_level_results:
                    # Keep the file-level result with highest score
                    best_file = max(file_level_results, key=lambda x: x.get("total_score", 0))
                    deduplicated.append(best_file)
            else:
                # No file-level result, keep all class and function results
                # Only remove exact duplicates (same element id or same type/name/location)
                elem_id_seen = {}
                
                for result in group:
                    elem = result.get("element", {})
                    elem_id = elem.get("id", "")
                    elem_type = elem.get("type", "")
                    elem_name = elem.get("name", "")
                    start_line = elem.get("start_line", 0)
                    
                    # Use element ID as primary key, fallback to (type, name, start_line) for uniqueness
                    if elem_id:
                        unique_key = elem_id
                    else:
                        unique_key = (elem_type, elem_name, start_line)
                    
                    # Keep the one with highest score for exact duplicates only
                    if unique_key not in elem_id_seen or result.get("total_score", 0) > elem_id_seen[unique_key].get("total_score", 0):
                        elem_id_seen[unique_key] = result
                
                deduplicated.extend(elem_id_seen.values())
        
        # Sort by score
        deduplicated.sort(key=lambda x: x.get("total_score", 0), reverse=True)
        
        self.logger.info(f"Removed {len(results) - len(deduplicated)} duplicate files")
        
        return deduplicated
