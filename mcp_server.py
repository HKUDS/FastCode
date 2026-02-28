"""
FastCode MCP Server - Expose repo-level code understanding via MCP protocol.

Usage:
    python mcp_server.py                    # stdio transport (default)
    python mcp_server.py --transport sse    # SSE transport on port 8080
    python mcp_server.py --port 9090        # SSE on custom port

MCP config example (for Claude Code / Cursor):
    {
      "mcpServers": {
        "fastcode": {
          "command": "python",
          "args": ["/path/to/FastCode/mcp_server.py"],
          "env": {
            "MODEL": "your-model",
            "BASE_URL": "your-base-url",
            "API_KEY": "your-api-key"
          }
        }
      }
    }
"""

import os
import sys
import logging
import asyncio
import uuid
import inspect
from typing import Optional, List

# Ensure project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Logging (file only – stdout is reserved for MCP JSON-RPC in stdio mode)
# ---------------------------------------------------------------------------
log_dir = os.path.join(PROJECT_ROOT, "logs")
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(os.path.join(log_dir, "mcp_server.log"))],
)
logger = logging.getLogger("fastcode.mcp")

# ---------------------------------------------------------------------------
# Lazy FastCode singleton
# ---------------------------------------------------------------------------
_fastcode_instance = None


def _get_fastcode():
    """Lazy-init the FastCode engine (heavy imports happen here)."""
    global _fastcode_instance
    if _fastcode_instance is None:
        logger.info("Initializing FastCode engine …")
        from fastcode import FastCode
        _fastcode_instance = FastCode()
        logger.info("FastCode engine ready.")
    return _fastcode_instance


def _repo_name_from_source(source: str, is_url: bool) -> str:
    """Derive a canonical repo name from a URL or local path."""
    from fastcode.utils import get_repo_name_from_url
    if is_url:
        return get_repo_name_from_url(source)
    # Local path: use the directory basename
    return os.path.basename(os.path.normpath(source))


def _is_repo_indexed(repo_name: str) -> bool:
    """Check whether a repo already has a persisted FAISS index."""
    fc = _get_fastcode()
    persist_dir = fc.vector_store.persist_dir
    faiss_path = os.path.join(persist_dir, f"{repo_name}.faiss")
    meta_path = os.path.join(persist_dir, f"{repo_name}_metadata.pkl")
    return os.path.exists(faiss_path) and os.path.exists(meta_path)


def _apply_forced_env_excludes(fc) -> None:
    """
    Force-ignore environment-related paths before indexing.

    Always excludes virtual environment folders. Optionally excludes
    site-packages when FASTCODE_EXCLUDE_SITE_PACKAGES=1.
    """
    repo_cfg = fc.config.setdefault("repository", {})
    ignore_patterns = list(repo_cfg.get("ignore_patterns", []))

    forced_patterns = [
        ".venv",
        "venv",
        ".env",
        "env",
        "**/.venv/**",
        "**/venv/**",
        "**/.env/**",
        "**/env/**",
    ]

    # Optional (opt-in): site-packages can be huge/noisy in some repos.
    if os.getenv("FASTCODE_EXCLUDE_SITE_PACKAGES", "0").lower() in {"1", "true", "yes"}:
        forced_patterns.extend([
            "site-packages",
            "**/site-packages/**",
        ])

    added = []
    for pattern in forced_patterns:
        if pattern not in ignore_patterns:
            ignore_patterns.append(pattern)
            added.append(pattern)

    repo_cfg["ignore_patterns"] = ignore_patterns
    # Keep loader in sync when FastCode instance already exists.
    fc.loader.ignore_patterns = ignore_patterns

    if added:
        logger.info(f"Added forced ignore patterns: {added}")


def _ensure_repos_ready(repos: List[str], ctx=None) -> List[str]:
    """
    For each repo source string:
      - If already indexed → skip
      - If URL and not on disk → clone + index
      - If local path → load + index

    Returns the list of canonical repo names that are ready.
    """
    fc = _get_fastcode()
    _apply_forced_env_excludes(fc)
    ready_names: List[str] = []

    for source in repos:
        resolved_is_url = fc._infer_is_url(source)
        name = _repo_name_from_source(source, resolved_is_url)

        # Already indexed – nothing to do
        if _is_repo_indexed(name):
            logger.info(f"Repo '{name}' already indexed, skipping.")
            ready_names.append(name)
            continue

        # Need to index
        logger.info(f"Repo '{name}' not indexed. Preparing …")

        if resolved_is_url:
            # Clone and index
            logger.info(f"Cloning {source} …")
            fc.load_repository(source, is_url=True)
        else:
            # Local path
            abs_path = os.path.abspath(source)
            if not os.path.isdir(abs_path):
                logger.error(f"Local path does not exist: {abs_path}")
                continue
            fc.load_repository(abs_path, is_url=False)

        logger.info(f"Indexing '{name}' …")
        fc.index_repository(force=False)
        logger.info(f"Indexing '{name}' complete.")
        ready_names.append(name)

    return ready_names


# ---------------------------------------------------------------------------
# MCP Server
# ---------------------------------------------------------------------------
MCP_SERVER_DESCRIPTION = "Repo-level code understanding - ask questions about any codebase."
_fastmcp_kwargs = {}
try:
    # Backward compatibility: older mcp versions do not accept `description`.
    if "description" in inspect.signature(FastMCP.__init__).parameters:
        _fastmcp_kwargs["description"] = MCP_SERVER_DESCRIPTION
except (TypeError, ValueError):
    # If signature introspection fails, fall back to the safest constructor shape.
    pass

mcp = FastMCP("FastCode", **_fastmcp_kwargs)


@mcp.tool()
def code_qa(
    question: str,
    repos: list[str],
    multi_turn: bool = True,
    session_id: str | None = None,
) -> str:
    """Ask a question about one or more code repositories.

    This is the core tool for repo-level code understanding. FastCode will
    automatically clone (if URL) and index repositories that haven't been
    indexed yet, then answer your question using hybrid retrieval + LLM.

    Args:
        question: The question to ask about the code.
        repos: List of repository sources. Each can be:
               - A GitHub/GitLab URL (e.g. "https://github.com/user/repo")
               - A local filesystem path (e.g. "/home/user/projects/myrepo")
               If the repo is already indexed, it won't be re-indexed.
        multi_turn: Enable multi-turn conversation mode. When True, previous
                    Q&A context from the same session_id is used. Default: True.
        session_id: Session identifier for multi-turn conversations. If not
                    provided, a new session is created automatically. Pass the
                    same session_id across calls to continue a conversation.

    Returns:
        The answer to your question, with source references.
    """
    fc = _get_fastcode()

    # 1. Ensure all repos are indexed
    ready_names = _ensure_repos_ready(repos)
    if not ready_names:
        return "Error: None of the specified repositories could be loaded or indexed."

    # 2. Load indexed repos into memory (multi-repo merge)
    if not fc.repo_indexed or set(ready_names) != set(fc.loaded_repositories.keys()):
        logger.info(f"Loading repos into memory: {ready_names}")
        success = fc._load_multi_repo_cache(repo_names=ready_names)
        if not success:
            return "Error: Failed to load repository indexes."

    # 3. Session management
    sid = session_id or str(uuid.uuid4())[:8]

    # 4. Query
    result = fc.query(
        question=question,
        # Always enforce repository filtering for both single-repo and
        # multi-repo queries to avoid cross-repo source leakage.
        repo_filter=ready_names,
        session_id=sid,
        enable_multi_turn=multi_turn,
    )

    answer = result.get("answer", "")
    sources = result.get("sources", [])

    # Format output
    parts = [answer]

    if sources:
        parts.append("\n\n---\nSources:")
        for s in sources[:]:
            file_path = s.get("file", s.get("relative_path", ""))
            repo = s.get("repo", s.get("repository", ""))
            name = s.get("name", "")
            start = s.get("start_line", "")
            end = s.get("end_line", "")
            if (not start or not end) and s.get("lines"):
                lines = str(s.get("lines", ""))
                if "-" in lines:
                    parsed_start, parsed_end = lines.split("-", 1)
                    start = start or parsed_start
                    end = end or parsed_end
            loc = f"L{start}-L{end}" if start and end else ""
            parts.append(f"  - {repo}/{file_path}:{loc} ({name})" if repo else f"  - {file_path}:{loc} ({name})")

    parts.append(f"\n[session_id: {sid}]")
    return "\n".join(parts)


@mcp.tool()
def list_sessions() -> str:
    """List all existing conversation sessions.

    Returns a list of sessions with their IDs, titles (first query),
    turn counts, and timestamps. Useful for finding a session_id to
    continue a previous conversation.
    """
    fc = _get_fastcode()
    sessions = fc.list_sessions()

    if not sessions:
        return "No sessions found."

    lines = ["Sessions:"]
    for s in sessions:
        sid = s.get("session_id", "?")
        title = s.get("title", "Untitled")
        turns = s.get("total_turns", 0)
        mode = "multi-turn" if s.get("multi_turn", False) else "single-turn"
        lines.append(f"  - {sid}: \"{title}\" ({turns} turns, {mode})")

    return "\n".join(lines)


@mcp.tool()
def get_session_history(session_id: str) -> str:
    """Get the full conversation history for a session.

    Args:
        session_id: The session identifier to retrieve history for.

    Returns:
        The complete Q&A history of the session.
    """
    fc = _get_fastcode()
    history = fc.get_session_history(session_id)

    if not history:
        return f"No history found for session '{session_id}'."

    lines = [f"Session {session_id} history:"]
    for turn in history:
        turn_num = turn.get("turn_number", "?")
        query = turn.get("query", "")
        answer = turn.get("answer", "")
        # Truncate long answers for readability
        if len(answer) > 500:
            answer = answer[:500] + " …"
        lines.append(f"\n--- Turn {turn_num} ---")
        lines.append(f"Q: {query}")
        lines.append(f"A: {answer}")

    return "\n".join(lines)


@mcp.tool()
def delete_session(session_id: str) -> str:
    """Delete a conversation session and all its history.

    Args:
        session_id: The session identifier to delete.

    Returns:
        Confirmation message.
    """
    fc = _get_fastcode()
    success = fc.delete_session(session_id)
    if success:
        return f"Session '{session_id}' deleted."
    return f"Failed to delete session '{session_id}'. It may not exist."


@mcp.tool()
def list_indexed_repos() -> str:
    """List all repositories that have been indexed and are available for querying.

    Returns:
        A list of indexed repository names with metadata.
    """
    fc = _get_fastcode()
    available = fc.vector_store.scan_available_indexes(use_cache=False)

    if not available:
        return "No indexed repositories found."

    lines = ["Indexed repositories:"]
    for repo in available:
        name = repo.get("name", repo.get("repo_name", "?"))
        elements = repo.get("element_count", repo.get("elements", "?"))
        size = repo.get("size_mb", "?")
        lines.append(f"  - {name} ({elements} elements, {size} MB)")

    return "\n".join(lines)


@mcp.tool()
def delete_repo_metadata(repo_name: str) -> str:
    """Delete indexed metadata for a repository while keeping source code.

    This removes vector/BM25/graph index artifacts and the repository's
    overview entry from repo_overviews.pkl, but does NOT delete source files
    from the configured repository workspace.

    Args:
        repo_name: Repository name to clean metadata for.

    Returns:
        Confirmation message with deleted artifacts and freed disk space.
    """
    fc = _get_fastcode()
    result = fc.remove_repository(repo_name, delete_source=False)

    deleted_files = result.get("deleted_files", [])
    freed_mb = result.get("freed_mb", 0)

    if not deleted_files:
        return (
            f"No metadata files found for repository '{repo_name}'. "
            "Source code was not modified."
        )

    lines = [f"Deleted metadata for repository '{repo_name}' (source code kept)."]
    lines.append(f"Freed: {freed_mb} MB")
    lines.append("Removed artifacts:")
    for fname in deleted_files:
        lines.append(f"  - {fname}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FastCode MCP Server")
    parser.add_argument(
        "--transport", choices=["stdio", "sse"], default="stdio",
        help="MCP transport (default: stdio)",
    )
    parser.add_argument(
        "--port", type=int, default=8080,
        help="Port for SSE transport (default: 8080)",
    )
    args = parser.parse_args()

    if args.transport == "sse":
        mcp.run(transport="sse", sse_params={"port": args.port})
    else:
        mcp.run(transport="stdio")
