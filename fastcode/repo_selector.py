"""
Repository Selector - LLM-based repository and file selection for multi-repo scenarios
"""

import logging
from typing import List, Dict, Any, Optional
import os
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv
import re

from .llm_utils import openai_chat_completion


class RepositorySelector:
    """Use LLM to select relevant repositories and files based on user query"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.gen_config = config.get("generation", {})
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        load_dotenv()
        
        # LLM settings
        self.provider = self.gen_config.get("provider", "openai")
        self.model = os.getenv("MODEL")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        
        self.temperature = 0.2  # Low temperature for precise selection
        self.max_tokens = 2000
        
        # Initialize LLM client
        self.llm_client = self._initialize_client()
    
    def _initialize_client(self):
        """Initialize LLM client"""
        try:
            if self.provider == "openai":
                if not self.api_key:
                    self.logger.warning("OPENAI_API_KEY not set")
                    return None
                return OpenAI(api_key=self.api_key, base_url=self.base_url)
            
            elif self.provider == "anthropic":
                if not self.anthropic_api_key:
                    self.logger.warning("ANTHROPIC_API_KEY not set")
                    return None
                return Anthropic(api_key=self.anthropic_api_key, base_url=self.base_url)
            
            else:
                self.logger.warning(f"Unknown provider: {self.provider}")
                return None
        except Exception as e:
            self.logger.warning(f"Failed to initialize LLM client: {e}")
            return None
    
    def select_relevant_files(
        self,
        query: str,
        repo_overviews: List[Dict[str, Any]],
        max_files: int = 10,
        scenario_mode: str = "multi",
    ) -> List[Dict[str, str]]:
        """
        Select most relevant files from repositories based on user query
        
        Args:
            query: User query
            repo_overviews: List of repository overview dictionaries
            max_files: Maximum number of files to select
            scenario_mode: "single" when only one repository is in scope,
                           otherwise "multi"
        
        Returns:
            List of selected files with repo_name and file_path
        """
        if not self.llm_client:
            self.logger.warning("LLM client not available, cannot select files")
            return []
        
        self.logger.info(f"Selecting relevant files for query: {query[:50]}...")
        
        # Build prompt with repository information
        prompt = self._build_file_selection_prompt(
            query, repo_overviews, max_files, scenario_mode
        )
        
        try:
            # Call LLM
            if self.provider == "openai":
                response = self._call_openai(prompt)
            elif self.provider == "anthropic":
                response = self._call_anthropic(prompt)
            else:
                return []
            
            self.logger.info(f"fallback to LLM response of select_relevant_files: {response}")
            
            # Parse response to extract selected files
            selected_files = self._parse_file_selection_response(response, repo_overviews)
            
            self.logger.info(f"Selected {len(selected_files)} relevant files")
            return selected_files
            
        except Exception as e:
            self.logger.error(f"File selection failed: {e}")
            return []
    
    def _build_file_selection_prompt(
        self,
        query: str,
        repo_overviews: List[Dict[str, Any]],
        max_files: int,
        scenario_mode: str,
    ) -> str:
        """Build prompt for file selection"""

        is_single_repo = scenario_mode == "single"
        scope_line = (
            "You are a code navigation assistant. "
            "Select only files from this repository that best address the query.\n"
            if is_single_repo
            else "You are a code navigation assistant. Multiple repositories are in scope and some may be irrelevant. "
            "Identify the relevant repositories first, then pick the most relevant files only from those repositories.\n"
        )

        prompt_parts = [
            scope_line,
            f"\nUser Query: \"{query}\"\n",
            "\nRepository Information:\n",
        ]
        
        # Add repository overviews
        for i, overview in enumerate(repo_overviews, 1):
            repo_name = overview.get("repo_name", "Unknown")
            summary = overview.get("summary", "No summary available")
            structure_text = overview.get("structure_text", "")
            
            prompt_parts.append(f"\n{'='*60}")
            prompt_parts.append(f"\nRepository #{i}: {repo_name}")
            prompt_parts.append(f"\nSummary: {summary}")
            prompt_parts.append(f"\nFile Structure:")
            prompt_parts.append(structure_text[:])  # Limit structure text
            prompt_parts.append("\n")
        
        prompt_parts.append(f"\n{'='*60}\n")

        # task_prefix = (
        #     "Task: Analyze the query and repository information. Select up to "
        #     f"{max_files} most relevant files from this repository that would contain "
        #     "the answer or relevant code.\n"
        #     if is_single_repo
        #     else "Task: Analyze the query and repository information. Select up to "
        #     f"{max_files} most relevant files from the relevant repositories that would contain "
        #     "the answer or relevant code.\n"
        # )

        task_prefix = (
            "Task: Analyze the query and repository information. Select the fewest files needed "
            "to solve the problem (up to "
            f"{max_files} most relevant files) from this repository that would contain "
            "the answer or relevant code. For example, if one file is sufficient, pick only that file.\n"
            if is_single_repo
            else "Task: Analyze the query and repository information. Select the fewest files needed "
            "to solve the problem (up to "
            f"{max_files} most relevant files) from the relevant repositories that would contain "
            "the answer or relevant code. For example, if one file is sufficient, pick only that file.\n"
        )

        prompt_parts.append(f"\n{task_prefix}")
        
        prompt_parts.append("\nFor each selected file, provide:")
        prompt_parts.append("\n1. Repository name")
        prompt_parts.append("\n2. File path")
        
        prompt_parts.append("\nFormat your response EXACTLY as:")
        prompt_parts.append("\nFILE: <repo_name>::<file_path>")
        prompt_parts.append("\nREASON: <brief reason>")

        # print(f"Prompt parts: {''.join(prompt_parts)}")
        
        return "".join(prompt_parts)
    
    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API"""
        response = openai_chat_completion(
            self.llm_client,
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        return response.choices[0].message.content
    
    def _call_anthropic(self, prompt: str) -> str:
        """Call Anthropic API"""
        response = self.llm_client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.content[0].text
    
    def _parse_file_selection_response(self, response: str,
                                      repo_overviews: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Parse LLM response to extract selected files
        
        Args:
            response: LLM response text
            repo_overviews: Repository overviews for validation
        
        Returns:
            List of dictionaries with repo_name, file_path, and reason
        """
        selected_files = []
        
        # Get valid repo names for validation
        valid_repos = {ov.get("repo_name") for ov in repo_overviews}
        
        # Parse FILE: and REASON: pairs with flexible markdown support
        # Handles: FILE:, **FILE:**, and optional repo prefix
        file_pattern = r'\*{0,2}FILE:\*{0,2}\s*(?:(.+?)::)?(.+?)(?:\n|$)'
        reason_pattern = r'\*{0,2}REASON:\*{0,2}\s*(.+?)(?:\n|$)'
        
        file_matches = re.findall(file_pattern, response, re.MULTILINE)
        reason_matches = re.findall(reason_pattern, response, re.MULTILINE)
        
        # Match files with reasons
        for i, (repo_name, file_path) in enumerate(file_matches):
            repo_name = (repo_name or "").strip()
            file_path = file_path.strip()
            
            # Clean markdown formatting from both repo_name and file_path
            # Remove backticks: `filename`
            repo_name = re.sub(r'^`+|`+$', '', repo_name)
            file_path = re.sub(r'^`+|`+$', '', file_path)
            # Remove bold markers: **filename**
            repo_name = re.sub(r'^\*+|\*+$', '', repo_name)
            file_path = re.sub(r'^\*+|\*+$', '', file_path)
            # Strip again after removing markdown
            repo_name = repo_name.strip()
            file_path = file_path.strip()
            
            # Infer repo name when missing or invalid
            if repo_name not in valid_repos:
                inferred_repo = ""
                if file_path:
                    first_segment = file_path.split("/", 1)[0]
                    for candidate_repo in valid_repos:
                        if candidate_repo and candidate_repo.lower() == first_segment.lower():
                            inferred_repo = candidate_repo
                            # Strip repo prefix from file_path
                            remainder = file_path.split("/", 1)
                            file_path = remainder[1] if len(remainder) > 1 else ""
                            break
                if not inferred_repo and len(valid_repos) == 1:
                    inferred_repo = next(iter(valid_repos))
                if not inferred_repo:
                    self.logger.debug(f"Skipping invalid repo: {repo_name or '<missing>'}")
                    continue
                repo_name = inferred_repo
            
            reason = reason_matches[i].strip() if i < len(reason_matches) else "No reason provided"
            
            selected_files.append({
                "repo_name": repo_name,
                "file_path": file_path,
                "reason": reason
            })
        
        return selected_files
    
    def select_relevant_repos(
        self,
        query: str,
        repo_overviews: Dict[str, Dict[str, Any]],
        max_repos: int = 5,
    ) -> List[str]:
        """
        Use LLM to select relevant repositories from all available overviews.

        Args:
            query: User query
            repo_overviews: Dict mapping repo_name -> overview data
                            (as returned by vector_store.load_repo_overviews())
            max_repos: Maximum number of repositories to return

        Returns:
            List of selected repository names (robust-matched against actual names)
        """
        if not self.llm_client:
            self.logger.warning("LLM client not available, cannot select repos")
            return []

        available_names = list(repo_overviews.keys())
        if not available_names:
            return []

        self.logger.info(
            f"LLM repo selection: query='{query[:100]}', "
            f"candidates={available_names}, max_repos={max_repos}"
        )

        # --- build prompt ---
        prompt_parts = [
            "You are a code repository selection assistant.\n",
            f'User Query: "{query}"\n\n',
            "Below are the available repositories with their summaries.\n",
            "Select the repositories that are most relevant to the query.\n",
            f"Return at most {max_repos} repository names.\n\n",
        ]

        for idx, (repo_name, data) in enumerate(repo_overviews.items(), 1):
            metadata = data.get("metadata", {})
            summary = metadata.get("summary", data.get("content", "No summary available"))
            prompt_parts.append(f"Repository #{idx}: {repo_name}\n")
            prompt_parts.append(f"Summary: {summary[:1000]}\n\n")

        prompt_parts.append(
            "Respond with ONLY the selected repository names, one per line, "
            "using the format:\nREPO: <repository_name>\n\n"
            "If you cannot determine which repositories are relevant, "
            "or if the query is too general to narrow down, "
            "return ALL repository names listed above.\n"
            "Do NOT add any other text."
        )

        prompt = "".join(prompt_parts)
        self.logger.debug(f"LLM repo selection prompt ({len(prompt)} chars):\n{prompt}")

        # --- call LLM ---
        try:
            if self.provider == "openai":
                response = self._call_openai(prompt)
            elif self.provider == "anthropic":
                response = self._call_anthropic(prompt)
            else:
                return []

            self.logger.info(f"LLM repo selection response:\n{response}")
        except Exception as e:
            self.logger.error(f"LLM repo selection failed: {e}")
            return []

        # --- parse & robust-match ---
        return self._parse_repo_selection_response(response, available_names)

    # ------------------------------------------------------------------
    # Robust matching helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _normalize(name: str) -> str:
        """Lower-case, strip whitespace / backticks / quotes / asterisks."""
        name = name.strip().strip("`").strip("*").strip("'").strip('"')
        return name.lower()

    def _fuzzy_match_repo(self, candidate: str, available: List[str]) -> Optional[str]:
        """
        Try to match *candidate* (the string the LLM returned) to one of the
        *available* repository names using several heuristics, from strict to
        loose:

        1. Exact match (case-insensitive)
        2. Substring containment – candidate inside a real name or vice-versa
        3. Simple token-overlap ratio (Jaccard on alphanumeric tokens)

        Returns the best matching name, or None.
        """
        norm_candidate = self._normalize(candidate)
        if not norm_candidate:
            self.logger.info(f"Fuzzy match: candidate '{candidate}' is empty after normalization, skipping")
            return None

        self.logger.info(f"Fuzzy match: trying to match '{candidate}' (normalized: '{norm_candidate}')")

        # 1. exact (case-insensitive)
        for name in available:
            if self._normalize(name) == norm_candidate:
                self.logger.info(f"Fuzzy match: exact match '{candidate}' -> '{name}'")
                return name

        # 2. substring containment
        for name in available:
            norm_name = self._normalize(name)
            if norm_candidate in norm_name or norm_name in norm_candidate:
                self.logger.info(f"Fuzzy match: substring match '{candidate}' -> '{name}'")
                return name

        # 3. token-overlap ratio (Jaccard)
        def _tokens(s: str):
            return set(re.split(r'[\W_]+', s.lower())) - {''}

        cand_tokens = _tokens(norm_candidate)
        best_score, best_name = 0.0, None
        for name in available:
            name_tokens = _tokens(name)
            if not cand_tokens or not name_tokens:
                continue
            score = len(cand_tokens & name_tokens) / len(cand_tokens | name_tokens)
            self.logger.info(f"Fuzzy match: Jaccard('{candidate}', '{name}') = {score:.3f}")
            if score > best_score:
                best_score = score
                best_name = name

        if best_score >= 0.5:
            self.logger.info(f"Fuzzy match: Jaccard match '{candidate}' -> '{best_name}' (score={best_score:.3f})")
            return best_name

        self.logger.info(f"Fuzzy match: no match found for '{candidate}' (best Jaccard={best_score:.3f})")
        return None

    def _parse_repo_selection_response(
        self, response: str, available_names: List[str]
    ) -> List[str]:
        """
        Parse the LLM response and robustly match each name to available repos.
        """
        self.logger.info(f"Parsing repo selection response ({len(response)} chars), "
                         f"available repos: {available_names}")
        selected: List[str] = []
        seen: set = set()

        for line in response.splitlines():
            line = line.strip()
            # accept lines like "REPO: name" or "- name" or just "name"
            match = re.match(r'(?:\*{0,2}REPO:\*{0,2}\s*|[-•]\s*)(.*)', line, re.IGNORECASE)
            if match:
                raw_name = match.group(1).strip()
            elif line and not line.startswith("#"):
                raw_name = line
            else:
                continue

            matched = self._fuzzy_match_repo(raw_name, available_names)
            if matched and matched not in seen:
                selected.append(matched)
                seen.add(matched)
                self.logger.info(f"Matched LLM output '{raw_name}' -> '{matched}'")
            elif matched and matched in seen:
                self.logger.info(f"Skipping duplicate LLM output '{raw_name}' (already matched to '{matched}')")
            elif not matched:
                self.logger.warning(
                    f"Could not match LLM output '{raw_name}' to any available repo"
                )

        self.logger.info(f"Repo selection result: {selected} ({len(selected)} repos)")
        return selected

    def enhance_query_with_file_hints(self, query: str,
                                     selected_files: List[Dict[str, str]]) -> str:
        """
        Enhance query with file hints for better retrieval
        
        Args:
            query: Original query
            selected_files: Selected files from LLM
        
        Returns:
            Enhanced query string
        """
        if not selected_files:
            return query
        
        # Add file paths as context to query
        file_hints = []
        for sf in selected_files[:5]:  # Limit to top 5
            file_hints.append(f"{sf['repo_name']}/{sf['file_path']}")
        
        enhanced = f"{query} [Relevant files: {', '.join(file_hints)}]"
        return enhanced



