"""
Utility functions for FastCode
"""

import os
import hashlib
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import yaml
import tiktoken


def setup_logging(config: Dict[str, Any]) -> logging.Logger:
    """Setup logging configuration"""
    log_config = config.get("logging", {})
    level = getattr(logging, log_config.get("level", "INFO"))
    format_str = log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Create logs directory if it doesn't exist
    log_file = log_config.get("file", "./logs/fastcode.log")
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
    
    # Configure logging
    handlers = []
    if log_config.get("console", True):
        handlers.append(logging.StreamHandler())
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format=format_str,
        handlers=handlers
    )
    
    return logging.getLogger("fastcode")


def load_config(config_path: str = "config/config.yaml") -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Resolve relative paths against FastCode project root inferred from config path.
    config_file = Path(config_path).resolve()
    if config_file.parent.name == "config":
        project_root = config_file.parent.parent
    else:
        project_root = config_file.parent

    return resolve_config_paths(config, str(project_root))


def resolve_config_paths(config: Dict[str, Any], project_root: str) -> Dict[str, Any]:
    """
    Resolve relative directory/file paths in config to absolute paths.

    Paths are resolved against FastCode project_root so behavior is stable
    regardless of the process current working directory.
    """
    if not config:
        return config

    root = os.path.abspath(project_root)

    def _abs(path_value: Optional[str]) -> Optional[str]:
        if not path_value:
            return path_value
        if os.path.isabs(path_value):
            return os.path.abspath(path_value)
        return os.path.abspath(os.path.join(root, path_value))

    if "repo_root" in config:
        config["repo_root"] = _abs(config.get("repo_root"))

    vector_store_cfg = config.get("vector_store", {})
    if isinstance(vector_store_cfg, dict) and "persist_directory" in vector_store_cfg:
        vector_store_cfg["persist_directory"] = _abs(vector_store_cfg.get("persist_directory"))

    repository_cfg = config.get("repository", {})
    if isinstance(repository_cfg, dict) and "backup_directory" in repository_cfg:
        repository_cfg["backup_directory"] = _abs(repository_cfg.get("backup_directory"))

    cache_cfg = config.get("cache", {})
    if isinstance(cache_cfg, dict) and "cache_directory" in cache_cfg:
        cache_cfg["cache_directory"] = _abs(cache_cfg.get("cache_directory"))

    logging_cfg = config.get("logging", {})
    if isinstance(logging_cfg, dict) and "file" in logging_cfg:
        logging_cfg["file"] = _abs(logging_cfg.get("file"))

    return config


def compute_file_hash(file_path: str) -> str:
    """Compute MD5 hash of a file"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return ""


def is_text_file(file_path: str) -> bool:
    """Check if a file is a text file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            f.read(512)
        return True
    except (UnicodeDecodeError, IOError):
        return False


def get_file_extension(file_path: str) -> str:
    """Get file extension"""
    return Path(file_path).suffix


def is_supported_file(file_path: str, supported_extensions: List[str]) -> bool:
    """Check if file extension is supported"""
    ext = get_file_extension(file_path)
    return ext in supported_extensions


def should_ignore_path(path: str, ignore_patterns: List[str]) -> bool:
    """Check if path should be ignored based on patterns"""
    from pathspec import PathSpec
    from pathspec.patterns import GitWildMatchPattern
    
    spec = PathSpec.from_lines(GitWildMatchPattern, ignore_patterns)
    return spec.match_file(path)


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens in text"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    # Some retrieved snippets may contain literal special-token strings like
    # "<|endoftext|>", which raise in tiktoken.encode by default. Allow them so
    # counting doesn't fail on the first query in non-English cases.
    return len(encoding.encode(text, disallowed_special=()))


def truncate_to_tokens(text: str, max_tokens: int, model: str = "gpt-4") -> str:
    """Truncate text to fit within token limit"""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        encoding = tiktoken.get_encoding("cl100k_base")
    
    tokens = encoding.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return text
    
    truncated_tokens = tokens[:max_tokens]
    return encoding.decode(truncated_tokens)


def normalize_path(path: str) -> str:
    """Normalize file path"""
    return os.path.normpath(path).replace("\\", "/")


def get_language_from_extension(ext: str) -> str:
    """Get programming language from file extension"""
    language_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "javascript",
        ".tsx": "typescript",
        ".java": "java",
        ".go": "go",
        ".cpp": "cpp",
        ".cc": "cpp",
        ".c": "c",
        ".h": "c",
        ".hpp": "cpp",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".cs": "csharp",
        ".swift": "swift",
        ".kt": "kotlin",
        ".scala": "scala",
        ".r": "r",
        ".m": "objective-c",
        ".sh": "bash",
        ".sql": "sql",
        ".html": "html",
        ".css": "css",
        ".json": "json",
        ".xml": "xml",
        ".yaml": "yaml",
        ".yml": "yaml",
        ".toml": "toml",
        ".md": "markdown",
    }
    return language_map.get(ext.lower(), "unknown")


def extract_code_snippet(content: str, start_line: int, end_line: int, 
                         context_lines: int = 3) -> Dict[str, Any]:
    """Extract code snippet with context"""
    lines = content.split("\n")
    total_lines = len(lines)
    
    # Calculate actual range with context
    actual_start = max(0, start_line - context_lines)
    actual_end = min(total_lines, end_line + context_lines)
    
    snippet_lines = lines[actual_start:actual_end]
    
    return {
        "code": "\n".join(snippet_lines),
        "start_line": actual_start + 1,  # 1-indexed
        "end_line": actual_end,
        "highlighted_start": start_line + 1,
        "highlighted_end": end_line,
    }


def format_code_block(code: str, language: str = "", file_path: str = "", 
                      start_line: Optional[int] = None) -> str:
    """Format code block for display"""
    header = f"```{language}"
    if file_path:
        header += f" - {file_path}"
    if start_line:
        header += f" (Line {start_line})"
    
    return f"{header}\n{code}\n```"


def calculate_code_complexity(code: str) -> int:
    """Calculate simple cyclomatic complexity"""
    # Simple heuristic: count control flow keywords
    keywords = ['if', 'elif', 'else', 'for', 'while', 'try', 'except', 
                'catch', 'case', 'switch', '&&', '||', '?']
    
    complexity = 1
    for keyword in keywords:
        complexity += code.count(keyword)
    
    return complexity


def chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict[str, Any]]:
    """Chunk text with sliding window"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk_words = words[i:i + chunk_size]
        chunk_text = " ".join(chunk_words)
        
        chunks.append({
            "text": chunk_text,
            "start_word": i,
            "end_word": i + len(chunk_words),
        })
        
        if i + chunk_size >= len(words):
            break
    
    return chunks


def merge_dicts(*dicts: Dict) -> Dict:
    """Merge multiple dictionaries"""
    result = {}
    for d in dicts:
        result.update(d)
    return result


def safe_get(d: Dict, *keys, default=None):
    """Safely get nested dictionary value"""
    for key in keys:
        if isinstance(d, dict):
            d = d.get(key)
            if d is None:
                return default
        else:
            return default
    return d


def ensure_dir(directory: str):
    """Ensure directory exists"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def get_repo_name_from_url(url: str) -> str:
    """Extract repository name from URL"""
    # Handle GitHub URLs
    if url.endswith(".git"):
        url = url[:-4]
    
    parts = url.rstrip("/").split("/")
    return parts[-1] if parts else "unknown_repo"


def clean_docstring(docstring: str) -> str:
    """Clean and format docstring"""
    if not docstring:
        return ""
    
    lines = docstring.split("\n")
    # Remove leading/trailing empty lines
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    
    # Find minimum indentation
    min_indent = float('inf')
    for line in lines:
        if line.strip():
            indent = len(line) - len(line.lstrip())
            min_indent = min(min_indent, indent)
    
    # Remove common indentation
    if min_indent < float('inf'):
        lines = [line[min_indent:] if len(line) > min_indent else line 
                 for line in lines]
    
    return "\n".join(lines).strip()

