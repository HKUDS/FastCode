"""
Path utilities for converting file system paths to Python module paths
and centralized path handling for code exploration agents
"""

import os
from typing import Optional, Set
import logging


def file_path_to_module_path(file_path: str, repo_root: str) -> Optional[str]:
    """
    Convert a file system path to a logical dotted module path for RAG indexing

    Args:
        file_path: Absolute or relative file path (e.g., '/repo/app/services/auth.py')
        repo_root: Repository root directory (e.g., '/repo')

    Returns:
        Dotted module path (e.g., 'app.services.auth') or None if conversion fails

    Examples:
        >>> file_path_to_module_path('/repo/app/services/auth.py', '/repo')
        'app.services.auth'
        >>> file_path_to_module_path('/repo/app/__init__.py', '/repo')
        'app'
        >>> file_path_to_module_path('/repo/run-server.py', '/repo')
        'run-server'
        >>> file_path_to_module_path('/repo/db/01_init.py', '/repo')
        'db.01_init'

    Note:
        This function is designed for RAG systems, not Python imports.
        It generates logical identifiers even for files that cannot be imported
        (like scripts with hyphens or migrations starting with numbers).
    """
    logger = logging.getLogger(__name__)

    try:
        # Normalize paths to absolute paths
        abs_file_path = os.path.abspath(file_path)
        abs_repo_root = os.path.abspath(repo_root)

        # Normalize case for case-insensitive filesystems (Windows, macOS)
        abs_file_path = os.path.normcase(abs_file_path)
        abs_repo_root = os.path.normcase(abs_repo_root)

        # IMPROVEMENT: Use os.path.commonpath for more robust containment checking
        # This prevents path traversal attacks and handles edge cases like root directories
        try:
            common_path = os.path.commonpath([abs_file_path, abs_repo_root])
            if common_path != abs_repo_root.rstrip(os.path.sep):
                logger.warning(f"File {abs_file_path} is outside repo root {abs_repo_root}")
                return None
        except ValueError:
            # ValueError occurs when paths are on different drives (Windows) or completely unrelated
            logger.warning(f"File {abs_file_path} and repo root {abs_repo_root} are not comparable")
            return None

        # Get the relative path from repo root
        relative_path = os.path.relpath(abs_file_path.rstrip(os.path.sep), abs_repo_root.rstrip(os.path.sep))

        # Remove the file extension (.py)
        if not relative_path.endswith('.py'):
            # logger.warning(f"File {relative_path} is not a Python file")
            return None

        relative_path = relative_path[:-3]  # Remove '.py'

        # Convert path separators to dots
        module_path = relative_path.replace(os.path.sep, '.')

        # BUG FIX: Handle both endswith and exact match for __init__.py
        # This fixes cases like '/repo/__init__.py' -> '__init__' -> '' (should return '')
        if module_path == '__init__' or module_path.endswith('.__init__'):
            module_path = module_path[:-9] if module_path.endswith('.__init__') else ''

        # Remove any leading dots that might result from relative paths
        module_path = module_path.lstrip('.')

        # Validate the module path format
        if not module_path:
            # ROOT __init__.py CASE: When file_path is '/repo/__init__', module_path becomes empty
            # This is expected behavior - root __init__.py doesn't have a module name
            # Returning None is appropriate since there's no importable module
            logger.debug(f"Empty module path generated for {file_path} (likely root __init__.py)")
            return None

        # RAG-friendly validation: Only check for system-level problematic characters
        # We allow hyphens, numbers at start, and keywords since RAG needs to index all logical files
        if any(char in module_path for char in ['<', '>', ':', '"', '|', '?', '*']):
            logger.warning(f"Invalid characters in module path: {module_path}")
            return None

        # RAG NOTE: We intentionally DO NOT check for Python identifier compliance or keywords
        # Files like 'run-server.py', '01_init.py', or 'import.py' are valuable for RAG indexing
        # even though they cannot be imported as Python modules

        return module_path

    except (ValueError, OSError) as e:
        # More specific exception handling instead of catching all Exceptions
        logger.error(f"Path conversion error for {file_path}: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error converting {file_path} to module path: {e}")
        return None


def is_valid_python_file(file_path: str) -> bool:
    """
    Check if a file is a valid Python file

    Args:
        file_path: Path to check

    Returns:
        True if it's a valid Python file, False otherwise
    """
    # Check file extension
    if not file_path.endswith('.py'):
        return False

    # Check if it's a file (not directory)
    if not os.path.isfile(file_path):
        return False

    # Additional checks can be added here (e.g., file size, readable, etc.)
    return True


def normalize_repo_root(repo_root: str) -> str:
    """
    Normalize the repository root path

    Args:
        repo_root: Repository root path

    Returns:
        Normalized absolute path
    """
    return os.path.abspath(repo_root)


class PathUtils:
    """Centralized path handling utilities for agent-based code exploration"""

    def __init__(self, repo_root: str):
        """
        Initialize path utilities

        Args:
            repo_root: Root directory of the repository (security boundary)
        """
        self.repo_root = os.path.abspath(repo_root)
        self.logger = logging.getLogger(__name__)

        # Security: ensure repo_root exists and is a directory
        if not os.path.isdir(self.repo_root):
            raise ValueError(f"Repository root does not exist or is not a directory: {self.repo_root}")

    def resolve_path(self, path: str) -> Optional[str]:
        """
        Intelligently resolve path, specifically handling the case where
        repo_root ends with the same folder that path starts with.

        Example case:
            repo_root: /User/project/A/B/C
            path:      C/D/E

            Strategy A (Direct): /User/project/A/B/C/C/D/E  (Usually wrong in your case)
            Strategy B (Overlap): /User/project/A/B/C/D/E    (Correct, removes 'C' overlap)

        Args:
            path: Path to resolve (relative or absolute)

        Returns:
            Resolved absolute path, or None if path does not exist
        """
        path = path.strip()
        if not path or path == '.':
            return self.repo_root

        # 1. Preprocess path: normalize separators, handle ../ etc.
        norm_root = os.path.normpath(self.repo_root)
        norm_path = os.path.normpath(path)

        # 2. Prepare two candidate paths

        # --- Candidate A: Direct join ---
        # Result: .../A/B/C/C/D/E
        path_a = os.path.abspath(os.path.join(self.repo_root, path))

        # --- Candidate B: Smart dedup join ---
        path_b = None

        # Split path segments
        root_parts = norm_root.split(os.sep)
        input_parts = norm_path.split(os.sep)

        # Clean empty segments (e.g., from leading/trailing slashes)
        root_parts = [p for p in root_parts if p]
        input_parts = [p for p in input_parts if p]

        overlap_len = 0
        # Find maximum overlap in reverse order
        # e.g.: root=[..., 'B', 'C'], input=['C', 'D', 'E']
        # Loop will find root's ending ['C'] equals input's beginning ['C']
        min_len = min(len(root_parts), len(input_parts))
        for i in range(min_len, 0, -1):
            if root_parts[-i:] == input_parts[:i]:
                overlap_len = i
                break

        if overlap_len > 0:
            # input_parts[overlap_len:] is the remaining part after removing overlap
            # e.g., ['C', 'D', 'E'] minus 'C' becomes ['D', 'E']
            remaining_parts = input_parts[overlap_len:]
            if remaining_parts:
                path_b = os.path.abspath(os.path.join(self.repo_root, *remaining_parts))
            else:
                path_b = self.repo_root

        # 3. Decision logic

        exists_a = os.path.exists(path_a)
        # If path_b was not computed (no overlap), or equals path_a, treat B as non-existent
        exists_b = os.path.exists(path_b) if path_b and path_b != path_a else False

        # Priority logic
        if exists_a and exists_b:
            self.logger.warning(
                f"Path ambiguity: Both '{path_a}' and '{path_b}' exist. "
                f"Using direct join: '{path_a}'"
            )
            return path_a
        elif exists_a:
            return path_a
        elif exists_b:
            # This is the expected case: A doesn't exist, B exists, return B
            return path_b
        else:
            # Neither exists
            return None

    def is_safe_path(self, path: str) -> bool:
        """
        Check if path is within repository root (security check)

        Args:
            path: Path to check

        Returns:
            True if path is safe, False otherwise
        """
        try:
            resolved = self.resolve_path(path)
            if resolved is None:
                # Also check if the joined path would be safe (even if doesn't exist yet)
                abs_path = os.path.abspath(os.path.join(self.repo_root, path))
                return abs_path.startswith(self.repo_root)
            return resolved.startswith(self.repo_root)
        except Exception as e:
            self.logger.warning(f"Path security check failed for {path}: {e}")
            return False

    def detect_repo_name_from_path(self, file_path: str, known_repos: Set[str]) -> str:
        """
        Detect the correct repo_name from file_path by checking against known repos

        Args:
            file_path: File path that may contain repo name
            known_repos: Set of known repository names

        Returns:
            Detected repo_name or empty string
        """
        if not file_path:
            return ""

        path_parts = file_path.split('/')

        # Check each part of the path against known repo names
        for part in path_parts:
            # Exact match
            if part in known_repos:
                return part
            # Case-insensitive match
            part_lower = part.lower()
            for repo in known_repos:
                if repo.lower() == part_lower:
                    return repo

        # Fallback: return first known repo or empty string
        if known_repos:
            return next(iter(known_repos))

        return ""

    def normalize_path_with_repo(self, file_path: str, repo_name: str) -> str:
        """
        Normalize file path by intelligently removing repo prefix if present

        Handles various cases:
        1. repos/LoCoBench/locobench/evaluation/evaluator.py -> locobench/evaluation/evaluator.py
        2. astropy/astropy/wcs/wcs.py -> astropy/wcs/wcs.py
        3. LoCoBench/locobench/evaluation/evaluator.py -> locobench/evaluation/evaluator.py
        4. astropy/wcs/wcs.py -> wcs/wcs.py
        5. wcs/wcs.py -> wcs/wcs.py

        Args:
            file_path: Path that may or may not contain repo prefix
            repo_name: Repository name

        Returns:
            Normalized path without repo prefix
        """
        if not file_path or not repo_name:
            return file_path

        # Split path into parts
        path_parts = file_path.split('/')
        if not path_parts:
            return file_path

        repo_lower = repo_name.lower()

        # Find the position of repo_name in the path (case-insensitive)
        repo_index = -1
        for i, part in enumerate(path_parts):
            if part == repo_name or part.lower() == repo_lower:
                repo_index = i
                break

        self.logger.debug(f"[DEBUG] normalize_path_with_repo: file_path='{file_path}', repo_name='{repo_name}', repo_index={repo_index}")

        # If repo_name found in path
        if repo_index >= 0:
            # Check if there's a duplicate repo name after this one
            if repo_index + 1 < len(path_parts):
                next_part = path_parts[repo_index + 1]
                next_lower = next_part.lower()

                # If next part is also the repo name (same or different case), skip the first occurrence
                if next_part == repo_name or next_lower == repo_lower:
                    # Return from the second occurrence onwards
                    result = '/'.join(path_parts[repo_index + 1:])
                    self.logger.debug(f"[DEBUG] normalize_path_with_repo: duplicate repo name detected -> '{result}'")
                    return result

            # No duplicate, return from after repo_name
            result = '/'.join(path_parts[repo_index + 1:])
            self.logger.debug(f"[DEBUG] normalize_path_with_repo: removing repo prefix -> '{result}'")
            return result

        # repo_name not found in path, return as-is
        self.logger.debug(f"[DEBUG] normalize_path_with_repo: no repo prefix found -> returning as-is '{file_path}'")
        return file_path

    def resolve_repo_target_path(self, repo_name: str, user_path: str) -> str:
        """
        Helper: Resolve the actual path relative to repo_root based on LLM input.

        Logic:
        1. If user_path is empty or '.', target is the repo_name directory.
        2. If user_path starts with repo_name (e.g. "django/core"), use it as is.
        3. Otherwise (e.g. "core"), prepend repo_name (-> "django/core").
        4. **Important**: When the first path component matches repo_name case-insensitively,
           we validate against the filesystem to avoid incorrectly stripping subdirectory
           names that happen to match the repo name (e.g., FutureShow/futureshow/agent).

        Args:
            repo_name: Repository name
            user_path: User-provided path (may or may not include repo prefix)

        Returns:
            Resolved path with proper repo prefix
        """
        clean_path = user_path.strip().rstrip('/')
        self.logger.debug(f"[DEBUG] resolve_repo_target_path: repo_name='{repo_name}', user_path='{user_path}', clean_path='{clean_path}'")

        if not clean_path or clean_path == ".":
            self.logger.debug(f"[DEBUG] resolve_repo_target_path: empty/. path -> returning repo_name '{repo_name}'")
            return repo_name

        # Split path to check the first component
        # Use normpath to handle slash consistency
        parts = os.path.normpath(clean_path).split(os.sep)
        self.logger.debug(f"[DEBUG] resolve_repo_target_path: path parts = {parts}")

        # Check if the first part matches repo_name (case-insensitive for robustness)
        if parts[0].lower() == repo_name.lower():
            # The first component matches repo_name, but we need to validate
            # against the filesystem to handle the case where a subdirectory
            # has the same name as the repo (e.g., FutureShow/futureshow/agent)

            # Option A: Treat first part as repo prefix (strip it)
            # Result: repo_name/remaining_parts (e.g., FutureShow/agent)
            parts_stripped = parts[1:] if len(parts) > 1 else []
            if parts_stripped:
                path_stripped = os.path.join(repo_name, *parts_stripped)
            else:
                path_stripped = repo_name

            # Option B: Treat first part as subdirectory (keep it)
            # Result: repo_name/clean_path (e.g., FutureShow/futureshow/agent)
            path_unstripped = os.path.join(repo_name, clean_path)

            # Validate which path actually exists on filesystem
            full_path_stripped = os.path.join(self.repo_root, path_stripped)
            full_path_unstripped = os.path.join(self.repo_root, path_unstripped)

            stripped_exists = os.path.exists(full_path_stripped)
            unstripped_exists = os.path.exists(full_path_unstripped)

            self.logger.debug(f"[DEBUG] resolve_repo_target_path: path_stripped='{path_stripped}' exists={stripped_exists}")
            self.logger.debug(f"[DEBUG] resolve_repo_target_path: path_unstripped='{path_unstripped}' exists={unstripped_exists}")

            if unstripped_exists and not stripped_exists:
                # Unstripped path exists, stripped doesn't -> it's a subdirectory, not repo prefix
                self.logger.debug(f"[DEBUG] resolve_repo_target_path: unstripped path exists, treating as subdirectory -> returning '{path_unstripped}'")
                return path_unstripped
            elif stripped_exists:
                # Stripped path exists -> treat first part as repo prefix
                self.logger.debug(f"[DEBUG] resolve_repo_target_path: stripped path exists, treating as repo prefix -> returning '{path_stripped}'")
                return path_stripped
            else:
                # Neither exists - default to stripped (original behavior for new paths)
                self.logger.debug(f"[DEBUG] resolve_repo_target_path: neither path exists, defaulting to stripped -> returning '{path_stripped}'")
                return path_stripped
        else:
            # The model output is relative to the sub-repo
            result = os.path.join(repo_name, clean_path)
            self.logger.debug(f"[DEBUG] resolve_repo_target_path: prepending repo name -> returning '{result}'")
            return result

    def validate_and_normalize_file_pattern(self, file_pattern: str, repo_name: str) -> Optional[tuple[bool, str]]:
        """
        Validate if a file pattern actually targets a specific repo and return normalized pattern.

        This method handles the case where a subdirectory name matches the repo name.
        For example:
        - Repo: 'FutureShow'
        - Pattern: 'futureshow/agent/*.py'
        - Actual path: FutureShow/futureshow/agent/*.py

        The method checks if stripping the repo prefix results in a valid path before doing so.

        Args:
            file_pattern: File pattern that may contain repo prefix or glob patterns
            repo_name: Repository name to check against

        Returns:
            None if pattern doesn't target this repo
            (True, normalized_pattern) if pattern targets this repo and should be normalized

        The normalized_pattern will have the repo prefix stripped only if the resulting
        path actually exists in the filesystem.
        """
        if not file_pattern or not repo_name:
            return None

        norm_pattern = file_pattern.replace("\\", "/")

        # Check if pattern starts with repo name (case-insensitive)
        if not (norm_pattern.lower() == repo_name.lower() or
                norm_pattern.lower().startswith(repo_name.lower() + "/")):
            return None

        # Pattern potentially targets this repo, now validate by checking path existence
        # Get the proposed stripped pattern
        proposed_pattern = self.normalize_path_with_repo(norm_pattern, repo_name)

        # Extract the directory part (before any glob characters like *, ?, [)
        # Remove glob characters to get the base path for validation
        glob_chars = ['*', '?', '[']
        test_path = proposed_pattern
        for glob_char in glob_chars:
            if glob_char in test_path:
                test_path = test_path.split(glob_char)[0]
                break

        # Remove trailing slash
        test_path = test_path.rstrip('/')

        # If test_path is empty (pattern was just '*' or similar), accept the stripping
        if not test_path:
            self.logger.debug(f"[DEBUG] validate_and_normalize_file_pattern: pattern is pure glob, accepting stripped pattern '{proposed_pattern}'")
            return (True, proposed_pattern)

        # Build full path to check: repo_root/repo_name/test_path
        full_test_path_stripped = os.path.join(self.repo_root, repo_name, test_path)
        self.logger.debug(f"[DEBUG] validate_and_normalize_file_pattern: checking stripped path '{full_test_path_stripped}'")

        if os.path.exists(full_test_path_stripped):
            # Path exists with stripped prefix - the prefix was indeed the repo name
            self.logger.debug(f"[DEBUG] validate_and_normalize_file_pattern: stripped path exists, returning '{proposed_pattern}'")
            return (True, proposed_pattern)

        # Stripped path doesn't exist, check if unstripped path exists
        # This handles the case where pattern is actually a subdirectory matching repo name
        unstripped_test_path = norm_pattern
        for glob_char in glob_chars:
            if glob_char in unstripped_test_path:
                unstripped_test_path = unstripped_test_path.split(glob_char)[0]
                break
        unstripped_test_path = unstripped_test_path.rstrip('/')

        full_test_path_unstripped = os.path.join(self.repo_root, repo_name, unstripped_test_path)
        self.logger.debug(f"[DEBUG] validate_and_normalize_file_pattern: checking unstripped path '{full_test_path_unstripped}'")

        if os.path.exists(full_test_path_unstripped):
            # Unstripped path exists - this means the pattern is a subdirectory, not repo prefix
            self.logger.debug(f"[DEBUG] validate_and_normalize_file_pattern: unstripped path exists, pattern is subdirectory, returning '{norm_pattern}'")
            return (True, norm_pattern)

        # Neither exists - pattern might be for a file that will be created, or it's invalid
        # In this case, we default to the stripped version (original behavior)
        self.logger.debug(f"[DEBUG] validate_and_normalize_file_pattern: neither stripped nor unstripped path exists, defaulting to stripped '{proposed_pattern}'")
        return (True, proposed_pattern)