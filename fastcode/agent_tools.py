"""
Agent Tools - Read-only tools for code exploration by agents
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import fnmatch

from .path_utils import PathUtils


class AgentTools:
    """Read-only tools for agent-based code exploration"""

    def __init__(self, repo_root: str):
        """
        Initialize agent tools

        Args:
            repo_root: Root directory of the repository (security boundary)
        """
        self.repo_root = os.path.abspath(repo_root)
        self.logger = logging.getLogger(__name__)

        # Initialize path utilities
        self.path_utils = PathUtils(repo_root)

    def _resolve_path(self, path: str) -> Optional[str]:
        """
        Intelligently resolve path, specifically handling the case where
        repo_root ends with the same folder that path starts with.

        Delegates to PathUtils.resolve_path()
        """
        return self.path_utils.resolve_path(path)

    def _is_safe_path(self, path: str) -> bool:
        """
        Check if path is within repository root (security check)

        Delegates to PathUtils.is_safe_path()
        """
        return self.path_utils.is_safe_path(path)
    
    def list_directory(self, path: str = ".", 
                      include_hidden: bool = False) -> Dict[str, Any]:
        """
        List directory contents (read-only, secure)
        
        Args:
            path: Relative path from repo root (default: ".")

            include_hidden: Include hidden files/dirs (default: False)
        
        Returns:
            Dictionary with directory structure
        """
        # Security check
        if not self._is_safe_path(path):
            return {"success": False, "error": "Access denied: path outside repository root", "path": path}
        
        # Resolve path intelligently
        full_path = self._resolve_path(path)
        
        if full_path is None:
            return {"success": False, "error": f"Path does not exist: {path}", "path": path}
        
        if not os.path.isdir(full_path):
            return {"success": False, "error": f"Path is not a directory: {path}", "path": path}

        try:
            result = {"success": True, "path": path, "contents": []}

            # List directory contents
            for item in sorted(os.listdir(full_path)):
                # Skip hidden files if not included
                if not include_hidden and item.startswith('.'):
                    continue

                item_path = os.path.join(full_path, item)
                rel_path = os.path.relpath(item_path, self.repo_root)
                is_dir = os.path.isdir(item_path)
                
                item_info = {
                    "name": item,
                    "path": rel_path,
                    "type": "directory" if is_dir else "file"
                }
                
                # Add file size for files
                if not is_dir:
                    try:
                        item_info["size"] = os.path.getsize(item_path)
                    except:
                        item_info["size"] = 0
                
                result["contents"].append(item_info)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error listing directory {path}: {e}")
            return {"success": False, "error": str(e), "path": path}

    def search_codebase(self, search_term: str, file_pattern: str = "*", 
                       root_path: str = ".", max_results: int = 30,
                       case_sensitive: bool = False,
                       use_regex: bool = False) -> Dict[str, Any]:
        """
        Search for files containing specific strings or patterns
        
        Args:
            search_term: String or regex pattern to search for
            file_pattern: File pattern to match (e.g., "*.py", "*.js")
            root_path: Root path to start search (relative to repo root)
            max_results: Maximum number of matches to return
            case_sensitive: Whether search is case-sensitive
            use_regex: Whether to treat search_term as regex
        
        Returns:
            Dictionary with search results
        """
        # Security check
        self.logger.info(f"Searching codebase for term '{search_term}' "
                         f"in files matching '{file_pattern}' "
                         f"under '{root_path}' "
                         f"(case_sensitive={case_sensitive}, use_regex={use_regex}),"
                         f"working base path is {self.repo_root}")
        if not self._is_safe_path(root_path):
            return {
                "success": False,
                "error": f"Access denied: path outside repository root",
                "search_term": search_term
            }
        
        # Resolve path intelligently
        search_root = self._resolve_path(root_path)
        
        if search_root is None:
            return {
                "success": False,
                "error": f"Path does not exist: {root_path}",
                "search_term": search_term
            }
        try:
            # 1. Prepare search Pattern (moved out of loop)
            # -------------------------------------------------
            if use_regex:
                flags = 0 if case_sensitive else re.IGNORECASE
                try:
                    content_pattern = re.compile(search_term, flags)
                except re.error as e:
                    return {"success": False, "error": f"Invalid regex: {e}", "search_term": search_term}
            else:
                # Intelligently handle OR operation
                if '|' in search_term:
                    terms = [re.escape(t.strip()) for t in search_term.split('|')]
                    pattern_str = '|'.join(terms)
                else:
                    pattern_str = re.escape(search_term)
                flags = 0 if case_sensitive else re.IGNORECASE
                content_pattern = re.compile(pattern_str, flags)

            # 2. Prepare file path matching regex (moved out of loop and optimized)
            # -------------------------------------------------
            file_matcher = None
            normalized_file_pattern = file_pattern.lstrip('./').replace('\\', '/')

            # Define conversion function (Helper)
            def _compile_glob(pattern):
                # Convert glob (**/*.py) to regex
                parts = pattern.split('**')
                regex_parts = []
                for i, part in enumerate(parts):
                    if part:
                        # Use fnmatch to convert each part
                        part_regex = fnmatch.translate(part)
                        # Clean up trailing anchor (?ms) or \Z added by fnmatch
                        # Note: fnmatch output varies slightly across Python versions, universal cleanup method:
                        part_regex = re.sub(r'(\\Z(?:\(\?ms\))?|\\Z)$', '', part_regex)
                        regex_parts.append(part_regex)
                    if i < len(parts) - 1:
                        regex_parts.append('.*') # ** replaced with .*
                return re.compile('(?ms)' + ''.join(regex_parts) + r'\Z')

            if file_pattern != "*":
                try:
                    file_matcher = _compile_glob(normalized_file_pattern)
                except Exception as e:
                    self.logger.warning(f"Failed to compile file pattern '{file_pattern}', falling back to filename match. Error: {e}")
                    # file_matcher remains None, will be handled with fallback in loop

            results = []
            files_searched = 0

            # 3. Traverse files
            # -------------------------------------------------
            for root, dirs, files in os.walk(search_root):
                # Filter directories
                dirs[:] = [d for d in dirs if not d.startswith('.') and 
                        d not in ['__pycache__', 'node_modules', '.git', 'dist', 'build', 'venv']]
                
                for file in files:
                    if file.startswith('.'):
                        continue

                    file_path = os.path.join(root, file)

                    # [Key fix]: Calculate path relative to search_root for Pattern matching
                    # This way if user searches for "*.py" under "src", it can correctly match "main.py" in "src/main.py"
                    rel_to_search = os.path.relpath(file_path, search_root).replace(os.sep, '/')

                    # [For display]: Calculate path relative to Repo Root for returning results
                    rel_path_display = os.path.relpath(file_path, self.repo_root).replace(os.sep, '/')

                    # 4. File path matching logic
                    # -------------------------------------------------
                    match_file = False
                    if file_pattern == "*":
                        match_file = True
                    elif file_matcher:
                        # Try Regex matching first (match full path)
                        if file_matcher.match(rel_to_search):
                            match_file = True
                        # [Key fix]: If full path doesn't match, try matching just the filename
                        # This solves the case where user inputs "*.py" but rel_path is "subdir/file.py" (fnmatch usually handles this, but may be lost after regex conversion)
                        elif file_matcher.match(file):
                            match_file = True

                    # If regex conversion failed or no match, and no matcher, try original fnmatch as fallback
                    if not match_file and not file_matcher and file_pattern != "*":
                        if fnmatch.fnmatch(file, file_pattern) or fnmatch.fnmatch(rel_to_search, file_pattern):
                            match_file = True

                    if not match_file:
                        continue

                    files_searched += 1

                    # 5a. Filename/path match check
                    # -------------------------------------------------
                    filename_match = False
                    if use_regex:
                        if content_pattern.search(file) or content_pattern.search(rel_to_search):
                            filename_match = True
                    else:
                        if case_sensitive:
                            if search_term in file or search_term in rel_to_search:
                                filename_match = True
                        else:
                            search_lower = search_term.lower()
                            if search_lower in file.lower() or search_lower in rel_to_search.lower():
                                filename_match = True

                    # 5b. Content search
                    # -------------------------------------------------
                    try:
                        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                            content = f.read()

                        file_matches = []
                        lines = content.split('\n')

                        for i, line in enumerate(lines, 1):
                            if content_pattern.search(line):
                                file_matches.append({
                                    "line_number": i,
                                    "line_content": line.strip()[:200]
                                })
                                # Limit matches per file to prevent result explosion
                                if len(file_matches) >= 20:
                                    break

                        if file_matches or filename_match:
                            results.append({
                                "file": rel_path_display, # Return path relative to Repo to user
                                "match_count": len(file_matches),
                                "matches": file_matches,
                                "match_type": "both" if (file_matches and filename_match) else ("filename" if filename_match else "content")
                            })
                            if len(results) >= max_results:
                                break # Outer loop break requires logic control

                    except Exception:
                        # When content reading fails, still return if filename matches
                        if filename_match:
                            results.append({
                                "file": rel_path_display,
                                "match_count": 0,
                                "matches": [],
                                "match_type": "filename"
                            })
                            if len(results) >= max_results:
                                break
                        continue
                
                if len(results) >= max_results:
                    break
                
                # Auto-retry with recursive pattern if no results found
                # and pattern looks like it should be recursive (e.g., "dir/*.py" -> "dir/**/*.py")
            if len(results) == 0 and file_pattern != "*":
                # Check if pattern is non-recursive but could be made recursive
                if '**' not in file_pattern and '/' in file_pattern and '*' in file_pattern:
                    # Pattern like "src/_pytest/*.py" or "dir/subdir/*.ext"
                    # Try converting to recursive: "src/_pytest/**/*.py"
                    parts = file_pattern.rsplit('/', 1)
                    if len(parts) == 2:
                        dir_part, file_part = parts
                        recursive_pattern = f"{dir_part}/**/{file_part}"
                        
                        self.logger.info(f"No results with pattern '{file_pattern}', "
                                    f"auto-retrying with recursive pattern '{recursive_pattern}'")
                        
                        # Recursively call with the new pattern (only once, to avoid infinite loop)
                        return self.search_codebase(
                            search_term=search_term,
                            file_pattern=recursive_pattern,
                            root_path=root_path,
                            max_results=max_results,
                            case_sensitive=case_sensitive,
                            use_regex=use_regex
                        )
                
            # Add debug info if still no results found
            debug_info = {}
            if len(results) == 0:
                debug_info["files_searched"] = files_searched
                debug_info["search_root"] = rel_path_root = os.path.relpath(search_root, self.repo_root)
                if files_searched == 0:
                    debug_info["hint"] = "No files matched the file_pattern even after auto-retry with recursive pattern"
                else:
                    debug_info["hint"] = "Files were searched but no content matches found. Check search_term."
            
            return {
                "success": True,
                "search_term": search_term,
                "file_pattern": file_pattern,
                "files_searched": files_searched,
                "matches_found": len(results),
                "results": results,
                **debug_info
            }
            
        except Exception as e:
            self.logger.error(f"Error searching codebase: {e}")
            return {
                "success": False,
                "error": str(e),
                "search_term": search_term
            }
    
    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get basic file information without reading full content
        
        Args:
            file_path: Relative path to file
        
        Returns:
            Dictionary with file information
        """
        # Security check
        if not self._is_safe_path(file_path):
            return {
                "success": False,
                "error": f"Access denied: path outside repository root"
            }
        
        # Resolve path intelligently
        full_path = self._resolve_path(file_path)
        
        if full_path is None:
            return {
                "success": False,
                "error": f"File does not exist: {file_path}"
            }
        
        try:
            stat = os.stat(full_path)
            
            return {
                "success": True,
                "path": file_path,
                "size": stat.st_size,
                "is_file": os.path.isfile(full_path),
                "is_directory": os.path.isdir(full_path),
                "extension": os.path.splitext(file_path)[1]
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_file_structure_summary(self, file_path: str, max_lines: int = 100) -> Dict[str, Any]:
        """
        Get file structure summary (classes, functions) without full source code
        
        Args:
            file_path: Relative path to file
            max_lines: Maximum number of lines to read
        
        Returns:
            Dictionary with file structure
        """
        # Security check
        if not self._is_safe_path(file_path):
            return {
                "success": False,
                "error": f"Access denied: path outside repository root"
            }
        
        # Resolve path intelligently
        full_path = self._resolve_path(file_path)
        
        if full_path is None:
            return {
                "success": False,
                "error": f"File does not exist: {file_path}"
            }
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = []
                for i, line in enumerate(f):
                    if i >= max_lines:
                        break
                    lines.append(line.rstrip())
            
            # Extract structure (simple pattern matching)
            structure = {
                "success": True,
                "path": file_path,
                "total_lines_scanned": len(lines),
                "classes": [],
                "functions": [],
                "imports": []
            }
            
            for i, line in enumerate(lines, 1):
                stripped = line.strip()
                
                # Detect classes (Python, Java, etc.)
                if stripped.startswith('class '):
                    # Extract class name
                    match = re.match(r'class\s+(\w+)', stripped)
                    if match:
                        structure["classes"].append({
                            "name": match.group(1),
                            "line": i
                        })
                
                # Detect functions (Python, JavaScript, etc.)
                elif stripped.startswith('def ') or stripped.startswith('function '):
                    # Extract function name
                    match = re.match(r'(?:def|function)\s+(\w+)', stripped)
                    if match:
                        structure["functions"].append({
                            "name": match.group(1),
                            "line": i
                        })
                
                # Detect async functions
                elif 'async' in stripped and ('def ' in stripped or 'function ' in stripped):
                    match = re.search(r'(?:def|function)\s+(\w+)', stripped)
                    if match:
                        structure["functions"].append({
                            "name": match.group(1),
                            "line": i,
                            "is_async": True
                        })
                
                # Detect imports
                elif stripped.startswith('import ') or stripped.startswith('from '):
                    structure["imports"].append({
                        "line": i,
                        "statement": stripped[:100]  # Truncate long imports
                    })
            
            return structure
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def read_file_content(self, file_path: str, max_chars: int = 50000) -> Dict[str, Any]:
        """
        Read full file content (for including in retrieval results)
        
        Args:
            file_path: Relative path to file
            max_chars: Maximum number of characters to read
        
        Returns:
            Dictionary with file content
        """
        # Security check
        if not self._is_safe_path(file_path):
            return {
                "success": False,
                "error": f"Access denied: path outside repository root",
                "content": ""
            }
        
        # Resolve path intelligently
        full_path = self._resolve_path(file_path)
        
        if full_path is None:
            return {
                "success": False,
                "error": f"File does not exist: {file_path}",
                "content": ""
            }
        
        try:
            with open(full_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read(max_chars)
            
            # Count lines
            lines = content.split('\n')
            
            return {
                "success": True,
                "path": file_path,
                "content": content,
                "total_lines": len(lines),
                "truncated": len(content) >= max_chars
            }
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "content": ""
            }

