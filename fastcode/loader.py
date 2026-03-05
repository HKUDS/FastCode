"""
Repository Loader - Handle git cloning, local repository loading, and ZIP file extraction
"""

import os
import shutil
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from git import Repo, GitCommandError

from .utils import (
    is_supported_file,
    should_ignore_path,
    get_repo_name_from_url,
    normalize_path,
    ensure_dir,
)


class RepositoryLoader:
    """Load repositories from URLs or local paths"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.repo_config = config.get("repository", {})
        self.logger = logging.getLogger(__name__)
        
        self.clone_depth = self.repo_config.get("clone_depth", 1)
        self.max_file_size_mb = self.repo_config.get("max_file_size_mb", 5)
        self.ignore_patterns = self.repo_config.get("ignore_patterns", [])
        self.supported_extensions = self.repo_config.get("supported_extensions", [])
        self.safe_repo_root = os.path.abspath(self.config.get("repo_root", "./repos"))
        ensure_dir(self.safe_repo_root)
        configured_backup_root = self.repo_config.get("backup_directory")
        if configured_backup_root:
            self.repo_backup_root = os.path.abspath(configured_backup_root)
        else:
            self.repo_backup_root = os.path.join(os.path.dirname(self.safe_repo_root), "repo_backup")
        ensure_dir(self.repo_backup_root)
        
        self.temp_dir = None
        self.repo_path = None
        self.repo_name = None

    def _backup_existing_repo(self, repo_path: str) -> Optional[str]:
        """Move existing repository directory to backup workspace."""
        if not os.path.exists(repo_path):
            return None

        repo_name = os.path.basename(os.path.normpath(repo_path))
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_name = f"{repo_name}_{timestamp}"
        backup_path = os.path.join(self.repo_backup_root, backup_name)

        suffix = 1
        while os.path.exists(backup_path):
            backup_path = os.path.join(self.repo_backup_root, f"{backup_name}_{suffix}")
            suffix += 1

        self.logger.warning(
            f"Repository path {repo_path} already exists, moving to backup: {backup_path}"
        )
        try:
            shutil.move(repo_path, backup_path)
            return backup_path
        except Exception as e:
            raise RuntimeError(f"Failed to backup existing repository {repo_path}: {e}")

    def _prepare_repo_path(self, repo_name: str, target_dir: Optional[str] = None) -> str:
        """Prepare destination path under repository workspace."""
        base_dir = target_dir or self.safe_repo_root
        base_dir = os.path.abspath(base_dir)
        ensure_dir(base_dir)
        repo_path = os.path.join(base_dir, repo_name)

        if os.path.exists(repo_path):
            self._backup_existing_repo(repo_path)

        return repo_path
    
    def load_from_url(self, url: str, target_dir: Optional[str] = None) -> str:
        """
        Clone repository from URL
        
        Args:
            url: Git repository URL
            target_dir: Target directory for cloning (optional, creates temp dir if None)
        
        Returns:
            Path to cloned repository
        """
        self.logger.info(f"Cloning repository from {url}")
        
        # Get repository name
        self.repo_name = get_repo_name_from_url(url)
        
        self.repo_path = self._prepare_repo_path(self.repo_name, target_dir)
        
        try:
            # Clone with shallow depth for faster cloning
            if self.clone_depth > 0:
                Repo.clone_from(
                    url,
                    self.repo_path,
                    depth=self.clone_depth,
                    single_branch=True
                )
            else:
                Repo.clone_from(url, self.repo_path)
            
            self.logger.info(f"Successfully cloned to {self.repo_path}")
            return self.repo_path
            
        except GitCommandError as e:
            self.logger.error(f"Failed to clone repository: {e}")
            raise RuntimeError(f"Failed to clone repository: {e}")
    
    def load_from_path(self, path: str, target_dir: Optional[str] = None) -> str:
        """
        Copy local repository into repository workspace and load it.
        
        Args:
            path: Local path to repository
            target_dir: Optional destination root (defaults to configured repo_root)
        
        Returns:
            Path to copied repository
        """
        if not os.path.exists(path):
            raise ValueError(f"Path does not exist: {path}")
        
        if not os.path.isdir(path):
            raise ValueError(f"Path is not a directory: {path}")
        
        source_path = os.path.abspath(path)
        self.repo_name = os.path.basename(source_path)
        destination_root = os.path.abspath(target_dir) if target_dir else self.safe_repo_root
        destination_path = os.path.join(destination_root, self.repo_name)

        # If source is already in workspace destination, use it directly.
        if os.path.abspath(source_path) == os.path.abspath(destination_path):
            self.repo_path = source_path
            self.logger.info(f"Loaded repository from workspace path {self.repo_path}")
            return self.repo_path

        self.repo_path = self._prepare_repo_path(self.repo_name, target_dir)

        # Copy entire working tree, including untracked files.
        shutil.copytree(source_path, self.repo_path, symlinks=True)

        self.logger.info(f"Copied repository from {source_path} to {self.repo_path}")
        return self.repo_path
    
    def load_from_zip(self, zip_path: str, target_dir: Optional[str] = None) -> str:
        """
        Extract and load repository from ZIP file
        
        Args:
            zip_path: Path to ZIP file
            target_dir: Target directory for extraction (optional, creates temp dir if None)
        
        Returns:
            Path to extracted repository
        """
        if not os.path.exists(zip_path):
            raise ValueError(f"ZIP file does not exist: {zip_path}")
        
        if not zipfile.is_zipfile(zip_path):
            raise ValueError(f"File is not a valid ZIP archive: {zip_path}")
        
        self.logger.info(f"Extracting repository from ZIP: {zip_path}")
        
        # Get repository name from ZIP filename (without .zip extension)
        zip_basename = os.path.basename(zip_path)
        self.repo_name = os.path.splitext(zip_basename)[0]
        
        extract_path = self._prepare_repo_path(self.repo_name, target_dir)
        
        try:
            # Extract ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract all files
                zip_ref.extractall(extract_path)
            
            # If ZIP contains a single root directory, flatten it so repository
            # always lives directly under repo workspace root/<repo_name>.
            contents = os.listdir(extract_path)
            if len(contents) == 1 and os.path.isdir(os.path.join(extract_path, contents[0])):
                self.logger.info(f"Detected single root directory: {contents[0]}")
                root_dir = os.path.join(extract_path, contents[0])
                for name in os.listdir(root_dir):
                    shutil.move(os.path.join(root_dir, name), os.path.join(extract_path, name))
                shutil.rmtree(root_dir)

            self.repo_path = extract_path
            
            self.logger.info(f"Successfully extracted to {self.repo_path}")
            
            # Get file count for logging
            file_count = sum(1 for _ in Path(self.repo_path).rglob('*') if _.is_file())
            self.logger.info(f"Extracted {file_count} files")
            
            return self.repo_path
            
        except zipfile.BadZipFile as e:
            self.logger.error(f"Invalid ZIP file: {e}")
            raise RuntimeError(f"Invalid ZIP file: {e}")
        except Exception as e:
            self.logger.error(f"Failed to extract ZIP file: {e}")
            raise RuntimeError(f"Failed to extract ZIP file: {e}")
    
    def _load_gitignore_patterns(self) -> List[str]:
        """
        Load .gitignore patterns from the repository root.

        Returns:
            List of gitignore patterns, empty if no .gitignore found
        """
        if not self.repo_path:
            return []

        gitignore_path = os.path.join(self.repo_path, ".gitignore")
        if not os.path.isfile(gitignore_path):
            return []

        patterns = []
        try:
            with open(gitignore_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Skip empty lines and comments
                    if line and not line.startswith("#"):
                        patterns.append(line)
            self.logger.info(f"Loaded {len(patterns)} patterns from .gitignore")
        except Exception as e:
            self.logger.warning(f"Failed to read .gitignore: {e}")

        return patterns

    def scan_files(self) -> List[Dict[str, Any]]:
        """
        Scan repository and collect file metadata

        Returns:
            List of file metadata dictionaries
        """
        if not self.repo_path:
            raise RuntimeError("No repository loaded")

        self.logger.info(f"Scanning files in {self.repo_path}")

        # Merge .gitignore patterns into ignore_patterns
        gitignore_patterns = self._load_gitignore_patterns()
        if gitignore_patterns:
            effective_ignore = list(self.ignore_patterns) + gitignore_patterns
        else:
            effective_ignore = self.ignore_patterns

        files = []
        total_size = 0
        max_file_size_bytes = self.max_file_size_mb * 1024 * 1024
        
        for root, dirs, filenames in os.walk(self.repo_path):
            # Filter out ignored directories
            dirs[:] = [d for d in dirs if not should_ignore_path(
                os.path.join(root, d), effective_ignore
            )]

            for filename in filenames:
                file_path = os.path.join(root, filename)
                relative_path = os.path.relpath(file_path, self.repo_path)

                # Check if should ignore
                if should_ignore_path(relative_path, effective_ignore):
                    continue
                
                # Check if supported extension
                if not is_supported_file(file_path, self.supported_extensions):
                    continue
                
                # Check file size
                try:
                    file_size = os.path.getsize(file_path)
                    if file_size > max_file_size_bytes:
                        self.logger.warning(
                            f"Skipping large file: {relative_path} "
                            f"({file_size / 1024 / 1024:.2f} MB)"
                        )
                        continue
                    
                    files.append({
                        "path": normalize_path(file_path),
                        "relative_path": normalize_path(relative_path),
                        "size": file_size,
                        "extension": Path(file_path).suffix,
                    })
                    
                    total_size += file_size
                    
                except OSError as e:
                    self.logger.warning(f"Error accessing file {relative_path}: {e}")
                    continue
        
        self.logger.info(
            f"Found {len(files)} supported files "
            f"({total_size / 1024 / 1024:.2f} MB total)"
        )
        
        return files
    
    def read_file_content(self, file_path: str) -> Optional[str]:
        """
        Read file content with error handling
        
        Args:
            file_path: Path to file
        
        Returns:
            File content or None if error
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
            except Exception as e:
                self.logger.error(f"Failed to read {file_path}: {e}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to read {file_path}: {e}")
            return None
    
    def get_repository_info(self) -> Dict[str, Any]:
        """
        Get repository metadata
        
        Returns:
            Dictionary with repository information
        """
        if not self.repo_path:
            raise RuntimeError("No repository loaded")
        
        info = {
            "name": self.repo_name,
            "path": self.repo_path,
        }
        
        # Try to get git info
        try:
            repo = Repo(self.repo_path)
            info.update({
                "branch": repo.active_branch.name,
                "commit": repo.head.commit.hexsha,
                "remote_url": repo.remotes.origin.url if repo.remotes else None,
            })
        except Exception:
            self.logger.debug("Not a git repository or git info unavailable")
        
        # Count files
        files = self.scan_files()
        info.update({
            "file_count": len(files),
            "total_size_mb": sum(f["size"] for f in files) / 1024 / 1024,
        })
        
        return info
    
    def get_head_commit(self, repo_path: Optional[str] = None) -> Optional[str]:
        """Return the full HEAD commit SHA for a repo on disk, or None."""
        target = repo_path or self.repo_path
        if not target or not os.path.isdir(target):
            return None
        try:
            repo = Repo(target)
            return repo.head.commit.hexsha
        except Exception:
            return None

    def check_for_updates(self, repo_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Fetch from origin and compare local HEAD with remote HEAD.

        Returns a dict with:
            - has_updates (bool)
            - local_commit (str | None)
            - remote_commit (str | None)
            - error (str | None)
        """
        target = repo_path or self.repo_path
        result: Dict[str, Any] = {
            "has_updates": False,
            "local_commit": None,
            "remote_commit": None,
            "error": None,
        }
        if not target or not os.path.isdir(target):
            result["error"] = f"Path does not exist: {target}"
            return result

        try:
            repo = Repo(target)
            if not repo.remotes:
                result["error"] = "No remote configured"
                return result

            local_commit = repo.head.commit.hexsha
            result["local_commit"] = local_commit

            origin = repo.remotes.origin
            origin.fetch()

            tracking = repo.active_branch.tracking_branch()
            if tracking is None:
                # Shallow clones may lack tracking info; compare with origin/<branch>
                branch_name = repo.active_branch.name
                remote_ref = f"origin/{branch_name}"
                if remote_ref in [str(r) for r in repo.refs]:
                    remote_commit = repo.refs[remote_ref].commit.hexsha
                else:
                    result["error"] = f"Cannot determine remote ref for {branch_name}"
                    return result
            else:
                remote_commit = tracking.commit.hexsha

            result["remote_commit"] = remote_commit
            result["has_updates"] = local_commit != remote_commit
        except Exception as e:
            result["error"] = str(e)

        return result

    def pull_updates(self, repo_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Pull latest changes from origin for the given repo.

        For shallow clones (depth=1) we unshallow first so that pull works
        reliably, then re-shallow to keep disk usage low.

        Returns a dict with:
            - success (bool)
            - old_commit (str | None)
            - new_commit (str | None)
            - changed (bool) – whether HEAD actually moved
            - error (str | None)
        """
        target = repo_path or self.repo_path
        result: Dict[str, Any] = {
            "success": False,
            "old_commit": None,
            "new_commit": None,
            "changed": False,
            "error": None,
        }
        if not target or not os.path.isdir(target):
            result["error"] = f"Path does not exist: {target}"
            return result

        try:
            repo = Repo(target)
            if not repo.remotes:
                result["error"] = "No remote configured"
                return result

            old_commit = repo.head.commit.hexsha
            result["old_commit"] = old_commit

            origin = repo.remotes.origin
            origin.pull()

            new_commit = repo.head.commit.hexsha
            result["new_commit"] = new_commit
            result["changed"] = old_commit != new_commit
            result["success"] = True
        except Exception as e:
            result["error"] = str(e)

        return result

    def cleanup(self):
        """Clean up temporary directories"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            self.logger.info(f"Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
    
    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()

