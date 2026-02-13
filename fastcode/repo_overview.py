"""
Repository Overview Generator - Generate summaries and file structures for repositories
"""

import os
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from openai import OpenAI
from anthropic import Anthropic
from dotenv import load_dotenv


class RepositoryOverviewGenerator:
    """Generate repository overviews from README files and file structure"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.gen_config = config.get("generation", {})
        self.logger = logging.getLogger(__name__)
        
        # Load environment variables
        load_dotenv()
        
        # LLM settings for overview generation
        self.provider = self.gen_config.get("provider", "openai")
        self.model = os.getenv("MODEL")
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
        self.base_url = os.getenv("BASE_URL")
        
        self.temperature = 0.3  # Lower temperature for factual summaries
        self.max_tokens = 1000  # Longer for overview generation
        
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
    
    def generate_overview(self, repo_path: str, repo_name: str, 
                         file_structure: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate comprehensive repository overview
        
        Args:
            repo_path: Path to repository
            repo_name: Name of the repository
            file_structure: Parsed file structure information
        
        Returns:
            Dictionary with overview information
        """
        self.logger.info(f"Generating overview for repository: {repo_name}")
        
        # Find and read README file
        readme_content = self._find_and_read_readme(repo_path)
        
        # Generate overview from README if available
        if readme_content and self.llm_client:
            summary = self._summarize_readme_with_llm(repo_name, readme_content, file_structure)
        else:
            # Fallback: generate overview from file structure only
            summary = self._generate_structure_based_overview(repo_name, file_structure)
        
        # Generate detailed file structure text
        structure_text = self._format_file_structure(file_structure)
        
        overview = {
            "repo_name": repo_name,
            "summary": summary,
            "readme_content": readme_content[:10000] if readme_content else None,  # Truncate long READMEs
            "file_structure": file_structure,
            "structure_text": structure_text,
            "has_readme": readme_content is not None,
        }
        
        return overview
    
    def _find_and_read_readme(self, repo_path: str) -> Optional[str]:
        """Find and read README file in repository"""
        # Support common README variants across ecosystems (case-insensitive, multiple extensions)
        readme_names = [
            "README.md", "readme.md", "README.MD", "Readme.md",
            "README.rst", "readme.rst", "README.RST", "Readme.rst",
            "README.txt", "readme.txt", "README.TXT",
            "README.markdown", "readme.markdown", "README.MARKDOWN",
            "README.mdown", "readme.mdown",
            "README", "readme",
        ]
        
        for readme_name in readme_names:
            readme_path = os.path.join(repo_path, readme_name)
            if os.path.exists(readme_path):
                try:
                    with open(readme_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    self.logger.debug(f"Found README: {readme_name}")
                    return content
                except Exception as e:
                    self.logger.warning(f"Failed to read {readme_name}: {e}")
                    continue
        
        self.logger.debug("No README file found")
        return None
    
    def parse_file_structure(self, repo_path: str, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Parse repository file structure from scanned files
        
        Args:
            repo_path: Path to repository
            files: List of file metadata from loader
        
        Returns:
            Structured representation of repository
        """
        structure = {
            "total_files": len(files),
            "languages": {},
            "directories": {},
            "file_types": {},
            "all_files": [],
            "key_files": [],
        }
        
        for file_info in files:
            rel_path = file_info["relative_path"]
            extension = file_info["extension"]
            structure["all_files"].append(rel_path)
            
            # Count by extension
            if extension not in structure["file_types"]:
                structure["file_types"][extension] = 0
            structure["file_types"][extension] += 1
            
            # Count by language (simple heuristic)
            language = self._get_language_from_extension(extension)
            if language != "unknown":
                if language not in structure["languages"]:
                    structure["languages"][language] = 0
                structure["languages"][language] += 1
            
            # Build directory tree
            dir_name = os.path.dirname(rel_path)
            if dir_name:
                dir_parts = dir_name.split(os.sep)
                for i in range(len(dir_parts)):
                    partial_dir = os.sep.join(dir_parts[:i+1])
                    if partial_dir not in structure["directories"]:
                        structure["directories"][partial_dir] = []
                    
                    file_name = os.path.basename(rel_path)
                    if i == len(dir_parts) - 1:  # This is the file's direct parent
                        if file_name not in structure["directories"][partial_dir]:
                            structure["directories"][partial_dir].append(file_name)
            # else:
            #     root_key = repo_path 
                
            #     if root_key not in structure["directories"]:
            #         structure["directories"][root_key] = []
                
            #     file_name = os.path.basename(rel_path)
            #     if file_name not in structure["directories"][root_key]:
            #         structure["directories"][root_key].append(file_name)
            
            # Identify key files
            if self._is_key_file(rel_path):
                structure["key_files"].append(rel_path)
        
        return structure
    
    def _get_language_from_extension(self, ext: str) -> str:
        """Get programming language from extension"""
        language_map = {
            ".py": "python",
            ".js": "javascript",
            ".ts": "typescript",
            ".jsx": "javascript",
            ".tsx": "typescript",
            ".java": "java",
            ".go": "go",
            ".cpp": "cpp",
            ".c": "c",
            ".rs": "rust",
            ".rb": "ruby",
            ".php": "php",
            ".cs": "csharp",
        }
        return language_map.get(ext.lower(), "unknown")
    
    def _is_key_file(self, file_path: str) -> bool:
        """Check if file is a key/important file"""
        file_name = os.path.basename(file_path).lower()
        key_names = [
            "main", "index", "app", "init", "config", "setup",
            "package.json", "requirements.txt", "go.mod", "cargo.toml",
            "dockerfile", "makefile", "cmakelists.txt"
        ]
        
        for key in key_names:
            if key in file_name:
                return True
        
        return False
    
    def _summarize_readme_with_llm(self, repo_name: str, readme_content: str,
                                   file_structure: Dict[str, Any]) -> str:
        """Use LLM to summarize README and infer repository purpose"""
        
        # Truncate very long READMEs
        if len(readme_content) > 8000:
            readme_content = readme_content[:8000] + "\n... (truncated)"
        
        # Format file structure info
        languages = ", ".join(file_structure.get("languages", {}).keys())
        total_files = file_structure.get("total_files", 0)
        
        prompt = f"""Analyze this repository and provide a concise summary of its main purpose and functionality.

Repository Name: {repo_name}
Total Files: {total_files}
Languages: {languages}

README Content:
{readme_content}

Provide a 3-5 sentence summary that covers:
1. What this repository/project does (main purpose)
2. Key technologies or frameworks used
3. Primary use cases or target users
4. Main components or modules (if identifiable)

Summary:"""
        
        try:
            if self.provider == "openai":
                response = self.llm_client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                return response.choices[0].message.content.strip()
            
            elif self.provider == "anthropic":
                response = self.llm_client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                return response.content[0].text.strip()
            
        except Exception as e:
            self.logger.error(f"LLM summarization failed: {e}")
            return self._generate_structure_based_overview(repo_name, file_structure)
        
        return self._generate_structure_based_overview(repo_name, file_structure)
    
    def _generate_structure_based_overview(self, repo_name: str,
                                          file_structure: Dict[str, Any]) -> str:
        """Generate overview based on file structure when README is unavailable"""
        
        languages = file_structure.get("languages", {})
        total_files = file_structure.get("total_files", 0)
        key_files = file_structure.get("key_files", [])
        
        # Determine primary language
        if languages:
            primary_lang = max(languages.items(), key=lambda x: x[1])[0]
        else:
            primary_lang = "unknown"
        
        # Infer project type from key files
        project_type = self._infer_project_type(key_files, languages)
        
        summary = f"{repo_name} is a {primary_lang} {project_type} with {total_files} files. "
        
        if len(languages) > 1:
            lang_list = ", ".join(languages.keys())
            summary += f"It uses multiple languages: {lang_list}. "
        
        if key_files:
            summary += f"Key entry points include: {', '.join(key_files[:5])}."
        
        return summary
    
    def _infer_project_type(self, key_files: List[str], 
                           languages: Dict[str, int]) -> str:
        """Infer project type from files and languages"""
        
        key_files_str = " ".join(key_files).lower()
        
        # Web frameworks
        if "package.json" in key_files_str:
            if "react" in key_files_str or "tsx" in languages:
                return "React web application"
            elif "vue" in key_files_str:
                return "Vue.js web application"
            return "Node.js application"
        
        # Python projects
        if "requirements.txt" in key_files_str or "setup.py" in key_files_str:
            if "django" in key_files_str:
                return "Django web application"
            elif "flask" in key_files_str:
                return "Flask web application"
            return "Python application"
        
        # Mobile
        if "android" in key_files_str or "java" in languages:
            return "Android application"
        if "ios" in key_files_str or "swift" in key_files_str:
            return "iOS application"
        
        # Containers
        if "dockerfile" in key_files_str:
            return "containerized application"
        
        # Default
        return "software project"
    
    def _format_file_structure(self, file_structure: Dict[str, Any]) -> str:
        """Format file structure as readable text"""
        
        lines = []
        
        # Summary
        total_files = file_structure.get("total_files", 0)
        lines.append(f"Total Files: {total_files}")
        
        # Languages
        languages = file_structure.get("languages", {})
        if languages:
            lines.append("\nLanguages:")
            for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  - {lang}: {count} files")
        
        # All files
        all_files = file_structure.get("all_files", [])
        if all_files:
            lines.append("\nFiles:")
            for file_path in sorted(all_files):
                lines.append(f"  - {file_path}")
        
        # Top-level directories
        directories = file_structure.get("directories", {})
        top_dirs = [d for d in directories.keys() if os.sep not in d or d.count(os.sep) == 0]
        if top_dirs:
            lines.append("\nTop-Level Directories:")
            for td in sorted(top_dirs)[:15]:  # Limit to 15
                file_count = len(directories[td])
                lines.append(f"  - {td}/ ({file_count} files)")
        
        return "\n".join(lines)



