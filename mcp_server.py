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


_staleness_warnings: List[str] = []
"""Per-call staleness warnings collected by _ensure_repos_ready."""


def _ensure_repos_ready(repos: List[str], ctx=None) -> List[str]:
    """
    For each repo source string:
      - If already indexed → check for staleness, warn if outdated
      - If URL and not on disk → clone + index
      - If local path → load + index

    Staleness warnings are collected in the module-level ``_staleness_warnings``
    list so callers (e.g. ``code_qa``) can append them to the response.

    Returns the list of canonical repo names that are ready.
    """
    global _staleness_warnings
    _staleness_warnings = []

    fc = _get_fastcode()
    _apply_forced_env_excludes(fc)
    ready_names: List[str] = []

    for source in repos:
        resolved_is_url = fc._infer_is_url(source)
        name = _repo_name_from_source(source, resolved_is_url)

        # Already indexed – check freshness before moving on
        if _is_repo_indexed(name):
            logger.info(f"Repo '{name}' already indexed, checking freshness …")
            ready_names.append(name)

            try:
                update_info = fc.check_repo_for_updates(name)
                if update_info.get("stale"):
                    short_old = (update_info.get("indexed_commit") or "unknown")[:8]
                    short_new = (
                        update_info.get("remote_commit")
                        or update_info.get("current_commit")
                        or "unknown"
                    )[:8]
                    msg = (
                        f"Note: '{name}' index was built at commit {short_old} "
                        f"but the repo is now at {short_new}. "
                        f"Run the refresh_repo tool with repo_name=\"{name}\" "
                        f"to pull latest changes and re-index."
                    )
                    _staleness_warnings.append(msg)
                    logger.warning(msg)
            except Exception as e:
                logger.debug(f"Staleness check failed for '{name}': {e}")

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

mcp = FastMCP(
    "FastCode",
    host=os.getenv("FASTMCP_HOST", "0.0.0.0"),
    port=int(os.getenv("FASTMCP_PORT", "8080")),
    **_fastmcp_kwargs,
)


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

    if _staleness_warnings:
        parts.append("\n\n---\nRepository freshness:")
        for warning in _staleness_warnings:
            parts.append(f"  - {warning}")

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
def check_repo_freshness(repos: list[str]) -> str:
    """Check whether indexed repositories are up-to-date with their remotes.

    This is a lightweight check (git fetch + SHA comparison) that does NOT
    modify the index. Use refresh_repo to actually pull and re-index.

    Args:
        repos: Repository sources (URLs or local paths) or repo names to check.

    Returns:
        A freshness report for each repository.
    """
    fc = _get_fastcode()
    lines = ["Repository freshness report:"]

    for source in repos:
        resolved_is_url = fc._infer_is_url(source)
        name = _repo_name_from_source(source, resolved_is_url)

        if not _is_repo_indexed(name):
            lines.append(f"  - {name}: not indexed")
            continue

        info = fc.check_repo_for_updates(name)
        if info.get("error"):
            lines.append(f"  - {name}: error checking — {info['error']}")
        elif info.get("stale"):
            indexed = (info.get("indexed_commit") or "unknown")[:8]
            remote = (info.get("remote_commit") or info.get("current_commit") or "unknown")[:8]
            lines.append(
                f"  - {name}: OUTDATED (indexed {indexed}, latest {remote}) "
                f"— use refresh_repo to update"
            )
        else:
            commit = (info.get("indexed_commit") or "unknown")[:8]
            lines.append(f"  - {name}: up-to-date ({commit})")

    return "\n".join(lines)


@mcp.tool()
def refresh_repo(repo_name: str) -> str:
    """Pull the latest changes for a repository and re-index it.

    This performs a git pull on the cloned repo, then re-indexes if new
    commits were found. Use check_repo_freshness first to see which repos
    need refreshing.

    Args:
        repo_name: The repository name (as shown by list_indexed_repos).

    Returns:
        A summary of what changed and whether re-indexing occurred.
    """
    fc = _get_fastcode()
    _apply_forced_env_excludes(fc)

    if not _is_repo_indexed(repo_name):
        return f"Repository '{repo_name}' is not indexed. Use code_qa to index it first."

    result = fc.refresh_repository(repo_name)

    if result.get("error"):
        return f"Failed to refresh '{repo_name}': {result['error']}"

    old = (result.get("old_commit") or "unknown")[:8]
    new = (result.get("new_commit") or "unknown")[:8]

    if not result.get("reindexed"):
        return f"Repository '{repo_name}' is already up-to-date at {new}."

    return (
        f"Repository '{repo_name}' refreshed successfully.\n"
        f"  Previous commit: {old}\n"
        f"  Current commit:  {new}\n"
        f"  Re-indexed: yes"
    )


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
        mcp.run(transport="sse")
    else:
        mcp.run(transport="stdio")
