"""
Repository Loader - Handle git cloning, local repository loading, and ZIP file extraction
"""

import os
import shutil
import tempfile
import zipfile
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
        
        self.temp_dir = None
        self.repo_path = None
        self.repo_name = None
    
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
        
        # Create target directory
        if target_dir is None:
            # self.temp_dir = tempfile.mkdtemp(prefix=f"fastcode_{self.repo_name}_")
            target_dir = "./repos"
        else:
            ensure_dir(target_dir)
        
        self.repo_path = os.path.join(target_dir, self.repo_name)
        
        # Remove if already exists
        if os.path.exists(self.repo_path):
            self.logger.warning(f"Repository path {self.repo_path} already exists, removing...")
            shutil.rmtree(self.repo_path)
        
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
    
    def load_from_path(self, path: str) -> str:
        """
        Load repository from local path
        
        Args:
            path: Local path to repository
        
        Returns:
            Path to repository
        """
        if not os.path.exists(path):
            raise ValueError(f"Path does not exist: {path}")
        
        if not os.path.isdir(path):
            raise ValueError(f"Path is not a directory: {path}")
        
        self.repo_path = os.path.abspath(path)
        self.repo_name = os.path.basename(self.repo_path)
        
        self.logger.info(f"Loaded repository from {self.repo_path}")
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
        
        # Create target directory
        if target_dir is None:
            self.temp_dir = tempfile.mkdtemp(prefix=f"fastcode_{self.repo_name}_")
            target_dir = self.temp_dir
        else:
            ensure_dir(target_dir)
        
        extract_path = os.path.join(target_dir, self.repo_name)
        
        # Remove if already exists
        if os.path.exists(extract_path):
            self.logger.warning(f"Extract path {extract_path} already exists, removing...")
            shutil.rmtree(extract_path)
        
        try:
            # Extract ZIP file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Extract all files
                zip_ref.extractall(extract_path)
            
            # Check if ZIP contains a single root directory (common pattern)
            # If so, use that as the repo path instead of the extract path
            contents = os.listdir(extract_path)
            if len(contents) == 1 and os.path.isdir(os.path.join(extract_path, contents[0])):
                # Single root directory - use it directly
                actual_repo_path = os.path.join(extract_path, contents[0])
                self.logger.info(f"Detected single root directory: {contents[0]}")
                self.repo_path = actual_repo_path
            else:
                # Multiple items at root level - use extract path
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
    
    def scan_files(self) -> List[Dict[str, Any]]:
        """
        Scan repository and collect file metadata
        
        Returns:
            List of file metadata dictionaries
        """
        if not self.repo_path:
            raise RuntimeError("No repository loaded")
        
        self.logger.info(f"Scanning files in {self.repo_path}")
        
        files = []
        total_size = 0
        max_file_size_bytes = self.max_file_size_mb * 1024 * 1024
        
        for root, dirs, filenames in os.walk(self.repo_path):
            # Filter ignored directories using paths relative to repo root.
            # Matching absolute paths can miss gitwildmatch patterns such as
            # "output/" or ".venv/".
            filtered_dirs = []
            for d in dirs:
                abs_dir_path = os.path.join(root, d)
                rel_dir_path = normalize_path(
                    os.path.relpath(abs_dir_path, self.repo_path)
                )
                rel_dir_with_trailing = f"{rel_dir_path}/"
                if should_ignore_path(rel_dir_path, self.ignore_patterns) or should_ignore_path(
                    rel_dir_with_trailing, self.ignore_patterns
                ):
                    continue
                filtered_dirs.append(d)
            dirs[:] = filtered_dirs
            
            for filename in filenames:
                file_path = os.path.join(root, filename)
                relative_path = normalize_path(os.path.relpath(file_path, self.repo_path))
                
                # Check if should ignore
                if should_ignore_path(relative_path, self.ignore_patterns):
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
                "commit": repo.head.commit.hexsha[:8],
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
    
    def cleanup(self):
        """Clean up temporary directories"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            self.logger.info(f"Cleaning up temporary directory: {self.temp_dir}")
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
    
    def __del__(self):
        """Cleanup on deletion"""
        self.cleanup()
