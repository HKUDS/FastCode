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
    from fastcode.utils import get_repo_name_from_path, get_repo_name_from_url

    def repo_index_files_exist(persist_dir: Optional[str], repo_name: Optional[str]) -> bool:
        if not persist_dir or not repo_name:
            return False
        faiss_path = os.path.join(persist_dir, f"{repo_name}.faiss")
        meta_path = os.path.join(persist_dir, f"{repo_name}_metadata.pkl")
        return os.path.exists(faiss_path) and os.path.exists(meta_path)

    def strip_repo_hash_suffix(repo_name: str) -> Optional[str]:
        base_name, sep, suffix = repo_name.rpartition("-")
        if not sep or len(suffix) != 8:
            return None
        if not all(ch in "0123456789abcdef" for ch in suffix.lower()):
            return None
        return base_name or None

    if is_url:
        return get_repo_name_from_url(source)

    workspace_root = None
    persist_dir = None
    try:
        fc = _get_fastcode()
        workspace_root = getattr(getattr(fc, "loader", None), "safe_repo_root", None)
        persist_dir = getattr(getattr(fc, "vector_store", None), "persist_dir", None)
    except Exception:
        workspace_root = None
        persist_dir = None

    repo_name = get_repo_name_from_path(source, workspace_root=workspace_root)
    if repo_index_files_exist(persist_dir, repo_name):
        return repo_name

    legacy_name = strip_repo_hash_suffix(repo_name)
    if repo_index_files_exist(persist_dir, legacy_name):
        return legacy_name

    return repo_name


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


def _invalidate_loaded_state(fc) -> None:
    """Mark in-memory indexes as stale so the next request reloads from disk."""
    fc.repo_indexed = False
    fc.repo_loaded = False
    fc.multi_repo_mode = False
    fc.loaded_repositories.clear()

    retriever = getattr(fc, "retriever", None)
    if retriever is not None and hasattr(retriever, "current_loaded_repos"):
        retriever.current_loaded_repos = None


def _run_full_reindex(
    repo_source: str, resolved_is_url: bool, repo_name: Optional[str] = None
) -> dict:
    """Reindex using a clean FastCode instance to avoid mixed in-memory artifacts."""
    from fastcode import FastCode

    reindex_fc = FastCode()
    _apply_forced_env_excludes(reindex_fc)

    if resolved_is_url:
        reindex_fc.load_repository(repo_source, is_url=True)
    else:
        abs_path = os.path.abspath(repo_source)
        if not os.path.isdir(abs_path):
            return {"status": "path_not_found", "count": 0, "path": abs_path}
        reindex_fc.load_repository(abs_path, is_url=False)

    if repo_name:
        reindex_fc.repo_info["name"] = repo_name
        if hasattr(reindex_fc.loader, "repo_name"):
            reindex_fc.loader.repo_name = repo_name

    reindex_fc.index_repository(force=True)
    count = reindex_fc.vector_store.get_count()

    if _fastcode_instance is not None:
        _invalidate_loaded_state(_fastcode_instance)

    return {"status": "success", "count": count}


def _ensure_repos_ready(repos: List[str], allow_incremental: bool = True, ctx=None) -> List[str]:
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

    def strip_repo_hash_suffix(repo_name: str) -> Optional[str]:
        base_name, sep, suffix = repo_name.rpartition("-")
        if not sep or len(suffix) != 8:
            return None
        if not all(ch in "0123456789abcdef" for ch in suffix.lower()):
            return None
        return base_name or None

    def find_legacy_local_collisions(source: str, resolved_name: str) -> List[str]:
        try:
            from fastcode.utils import get_repo_name_from_path
        except Exception:
            return []

        workspace_root = getattr(getattr(fc, "loader", None), "safe_repo_root", None)
        derived_name = get_repo_name_from_path(source, workspace_root=workspace_root)
        legacy_name = strip_repo_hash_suffix(derived_name)
        if not legacy_name or resolved_name != legacy_name or resolved_name == derived_name:
            return []

        source_path = os.path.abspath(source)
        parent_dir = os.path.dirname(source_path)
        search_root = os.path.dirname(parent_dir)
        if not search_root or search_root == parent_dir or search_root == os.path.sep:
            return []

        basename = os.path.basename(source_path.rstrip(os.sep))
        candidates = set()

        direct_candidate = os.path.join(search_root, basename)
        if os.path.isdir(direct_candidate):
            candidates.add(os.path.abspath(direct_candidate))

        try:
            for child in os.listdir(search_root):
                candidate = os.path.join(search_root, child, basename)
                if os.path.isdir(candidate):
                    candidates.add(os.path.abspath(candidate))
        except OSError:
            return []

        return sorted(candidates) if len(candidates) > 1 else []

    for source in repos:
        resolved_is_url = fc._infer_is_url(source)
        name = _repo_name_from_source(source, resolved_is_url)
        abs_path = os.path.abspath(source) if not resolved_is_url else None

        # Already indexed
        if _is_repo_indexed(name):
            if not resolved_is_url and not os.path.isdir(abs_path):
                logger.error(f"Local path does not exist: {abs_path}")
                continue

            if not resolved_is_url:
                collision_paths = find_legacy_local_collisions(abs_path, name)
                if collision_paths:
                    logger.error(
                        "Legacy local index '%s' is ambiguous for '%s'; matching paths=%s",
                        name,
                        abs_path,
                        collision_paths,
                    )
                    continue

            # Try incremental update for local repos
            if not resolved_is_url and allow_incremental:
                force_full_reindex = False
                try:
                    result = fc.incremental_reindex(name, repo_path=abs_path)
                    status = (result or {}).get("status")
                    if status in {
                        "no_manifest",
                        "no_metadata",
                        "embedding_mismatch",
                        "embedding_dimension_mismatch",
                        "full_reindex_required",
                    }:
                        force_full_reindex = True
                    elif result and result.get("changes", 0) > 0:
                        logger.info(f"Incremental update for '{name}': {result}")
                        _invalidate_loaded_state(fc)
                except Exception as e:
                    logger.warning(f"Incremental reindex failed for '{name}': {e}")
                    force_full_reindex = True

                if force_full_reindex:
                    logger.info(f"Falling back to full reindex for '{name}'")
                    result = _run_full_reindex(
                        abs_path, resolved_is_url=False, repo_name=name
                    )
                    if result.get("status") != "success":
                        logger.error(f"Full reindex failed for '{name}': {result}")
                        continue
            logger.info(f"Repo '{name}' ready.")
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


def _ensure_loaded(fc, ready_names: List[str]) -> bool:
    """Ensure repos are loaded into memory (vectors + BM25 + graphs)."""
    if not fc.repo_indexed or set(ready_names) != set(fc.loaded_repositories.keys()):
        logger.info(f"Loading repos into memory: {ready_names}")
        if not fc._load_multi_repo_cache(repo_names=ready_names):
            return False

    loaded_names = set(fc.loaded_repositories.keys())
    expected_names = set(ready_names)
    if loaded_names != expected_names:
        logger.error(
            "Loaded repository set mismatch. expected=%s loaded=%s",
            sorted(expected_names),
            sorted(loaded_names),
        )
        return False

    vector_store = getattr(fc, "vector_store", None)
    if vector_store is not None and hasattr(vector_store, "get_count"):
        if vector_store.get_count() == 0:
            logger.error("Repository load finished with an empty in-memory vector store")
            return False

    if vector_store is not None and hasattr(vector_store, "get_repository_names"):
        in_memory_names = set(vector_store.get_repository_names())
        if in_memory_names != expected_names:
            logger.error(
                "In-memory repository set mismatch. expected=%s loaded=%s",
                sorted(expected_names),
                sorted(in_memory_names),
            )
            return False

    return True


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
    if not _ensure_loaded(fc, ready_names):
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


@mcp.tool()
def search_symbol(
    symbol_name: str,
    repos: list[str],
    symbol_type: str | None = None,
) -> str:
    """Search for a symbol (function, class, method) by name across repositories.

    Finds definitions matching the given name with case-insensitive search.
    Results are ranked: exact match > prefix match > contains match (top 20).

    Args:
        symbol_name: Name of the symbol to search for (e.g. "FastCode", "query").
        repos: List of repository sources (URLs or local paths).
        symbol_type: Optional filter: "function", "class", "file", or "documentation".

    Returns:
        Matching definitions with file path, line range, and signature.
    """
    fc = _get_fastcode()
    ready_names = _ensure_repos_ready(repos)
    if not ready_names:
        return "Error: None of the specified repositories could be loaded."
    if not _ensure_loaded(fc, ready_names):
        return "Error: Failed to load repository indexes."

    query_lower = symbol_name.lower()
    exact, prefix, contains = [], [], []

    for meta in fc.vector_store.metadata:
        name = meta.get("name", "")
        elem_type = meta.get("type", "")
        if elem_type == "repository_overview":
            continue
        if symbol_type and elem_type != symbol_type:
            continue

        name_lower = name.lower()
        if name_lower == query_lower:
            exact.append(meta)
        elif name_lower.startswith(query_lower):
            prefix.append(meta)
        elif query_lower in name_lower:
            contains.append(meta)

    ranked = (exact + prefix + contains)[:20]
    if not ranked:
        return f"No symbols matching '{symbol_name}' found."

    lines = [f"Found {len(ranked)} result(s) for '{symbol_name}':"]
    for meta in ranked:
        name = meta.get("name", "")
        etype = meta.get("type", "")
        repo = meta.get("repo_name", "")
        rel_path = meta.get("relative_path", "")
        start = meta.get("start_line", "")
        end = meta.get("end_line", "")
        sig = meta.get("signature", "")
        loc = f"L{start}-L{end}" if start and end else ""
        line = f"  - [{etype}] {name}"
        if sig:
            line += f"  |  {sig}"
        line += f"\n    {repo}/{rel_path}:{loc}" if repo else f"\n    {rel_path}:{loc}"
        lines.append(line)

    return "\n".join(lines)


@mcp.tool()
def get_repo_structure(repo_name: str) -> str:
    """Get the high-level structure and summary of an indexed repository.

    Returns the repository summary, directory tree, and language statistics.
    Does not require loading the full index into memory.

    Args:
        repo_name: Name of an indexed repository (see list_indexed_repos).

    Returns:
        Repository summary, directory structure, and language breakdown.
    """
    fc = _get_fastcode()
    if not _is_repo_indexed(repo_name):
        return f"Repository '{repo_name}' is not indexed. Use code_qa or reindex_repo first."

    overviews = fc.vector_store.load_repo_overviews()
    overview = overviews.get(repo_name)
    if not overview:
        return f"No overview found for repository '{repo_name}'. It may need re-indexing."

    metadata = overview.get("metadata", {})
    summary = metadata.get("summary", "No summary available.")
    structure_text = metadata.get("structure_text", "")
    file_structure = metadata.get("file_structure", {})
    languages = file_structure.get("languages", {})

    parts = [f"Repository: {repo_name}", ""]
    parts.append(f"Summary:\n{summary}")

    if languages:
        parts.append("\nLanguages:")
        for lang, count in sorted(languages.items(), key=lambda x: -x[1]):
            parts.append(f"  - {lang}: {count} files")

    if structure_text:
        parts.append(f"\nDirectory Structure:\n{structure_text}")

    return "\n".join(parts)


@mcp.tool()
def get_file_summary(file_path: str, repos: list[str]) -> str:
    """Get the structure summary of a specific file (classes, functions, imports).

    Args:
        file_path: Path to the file (e.g. "fastcode/main.py").
                   Flexible matching: endswith or contains.
        repos: List of repository sources to search in.

    Returns:
        File structure: classes (with methods), top-level functions, and import count.
    """
    fc = _get_fastcode()
    ready_names = _ensure_repos_ready(repos)
    if not ready_names:
        return "Error: None of the specified repositories could be loaded."
    if not _ensure_loaded(fc, ready_names):
        return "Error: Failed to load repository indexes."

    # Find matching elements by relative_path
    matching = []
    for meta in fc.vector_store.metadata:
        rel = meta.get("relative_path", "")
        if meta.get("type") == "repository_overview":
            continue
        if rel.endswith(file_path) or file_path in rel:
            matching.append(meta)

    if not matching:
        return f"No elements found for file path '{file_path}'."

    files = [m for m in matching if m.get("type") == "file"]
    classes = [m for m in matching if m.get("type") == "class"]
    functions = [m for m in matching if m.get("type") == "function"]

    file_meta = files[0] if files else matching[0]
    actual_path = file_meta.get("relative_path", file_path)
    repo = file_meta.get("repo_name", "")

    parts = [f"File: {repo}/{actual_path}" if repo else f"File: {actual_path}"]

    if files:
        fm = files[0]
        parts.append(f"Language: {fm.get('language', '?')}")
        mi = fm.get("metadata", {})
        parts.append(f"Lines: {mi.get('total_lines', '?')} (code: {mi.get('code_lines', '?')})")
        num_imports = mi.get("num_imports", 0)
        if num_imports:
            parts.append(f"Imports: {num_imports}")

    if classes:
        parts.append(f"\nClasses ({len(classes)}):")
        for c in classes:
            sig = c.get("signature", c.get("name", ""))
            mi = c.get("metadata", {})
            methods = mi.get("methods", [])
            loc = f"L{c.get('start_line', '')}-L{c.get('end_line', '')}"
            parts.append(f"  - {sig} ({loc})")
            for m in methods:
                parts.append(f"      .{m}")

    if functions:
        top_level = [f for f in functions if not f.get("metadata", {}).get("class_name")]
        if top_level:
            parts.append(f"\nFunctions ({len(top_level)}):")
            for fn in top_level:
                sig = fn.get("signature", fn.get("name", ""))
                loc = f"L{fn.get('start_line', '')}-L{fn.get('end_line', '')}"
                parts.append(f"  - {sig} ({loc})")

    return "\n".join(parts)


def _walk_call_chain(gb, element_id: str, direction: str, hops_left: int,
                     parts: list, indent: int = 2, visited: set = None):
    """Recursively walk the call chain and format output."""
    if visited is None:
        visited = {element_id}

    neighbors = (gb.get_callers(element_id) if direction == "callers"
                 else gb.get_callees(element_id))

    if not neighbors:
        parts.append(f"{'  ' * indent}(none)")
        return

    for nid in neighbors:
        if nid in visited:
            continue
        visited.add(nid)
        elem = gb.element_by_id.get(nid)
        if elem:
            loc = f"{elem.relative_path}:L{elem.start_line}" if elem.relative_path else ""
            parts.append(f"{'  ' * indent}- {elem.name} [{loc}]")
            if hops_left > 1:
                _walk_call_chain(gb, nid, direction, hops_left - 1, parts, indent + 1, visited)


@mcp.tool()
def get_call_chain(
    symbol_name: str,
    repos: list[str],
    direction: str = "both",
    max_hops: int = 2,
) -> str:
    """Trace the call chain for a function or method.

    Shows who calls this symbol (callers) and/or what it calls (callees),
    up to max_hops levels deep.

    Args:
        symbol_name: Name of the function/method to trace.
        repos: List of repository sources.
        direction: "callers", "callees", or "both" (default: "both").
        max_hops: Maximum depth of the call chain (default: 2, max: 5).

    Returns:
        Formatted call chain showing callers and/or callees.
    """
    fc = _get_fastcode()
    ready_names = _ensure_repos_ready(repos)
    if not ready_names:
        return "Error: None of the specified repositories could be loaded."
    if not _ensure_loaded(fc, ready_names):
        return "Error: Failed to load repository indexes."

    max_hops = min(max_hops, 5)
    gb = fc.graph_builder
    name_lower = symbol_name.lower()
    target_id, target_elem = None, None

    # Exact match via element_by_name
    elem = gb.element_by_name.get(symbol_name)
    if elem:
        target_elem, target_id = elem, elem.id

    # Fallback: case-insensitive search
    if not target_id:
        for eid, elem in gb.element_by_id.items():
            if elem.name.lower() == name_lower:
                target_elem, target_id = elem, eid
                break

    # Fallback: partial match
    if not target_id:
        for eid, elem in gb.element_by_id.items():
            if name_lower in elem.name.lower():
                target_elem, target_id = elem, eid
                break

    if not target_id:
        return f"Symbol '{symbol_name}' not found in call graph."

    parts = [
        f"Call chain for '{target_elem.name}' ({target_elem.type})"
        f" at {target_elem.relative_path}:L{target_elem.start_line}"
    ]

    if direction in ("callers", "both"):
        parts.append("\n  Callers (who calls this):")
        _walk_call_chain(gb, target_id, "callers", max_hops, parts, indent=2)

    if direction in ("callees", "both"):
        parts.append("\n  Callees (what this calls):")
        _walk_call_chain(gb, target_id, "callees", max_hops, parts, indent=2)

    return "\n".join(parts)


@mcp.tool()
def reindex_repo(repo_source: str) -> str:
    """Force a full re-index of a repository.

    Clones (if URL) or loads (if local path) the repository and rebuilds
    all indexes from scratch.

    Args:
        repo_source: Repository URL or local filesystem path.

    Returns:
        Confirmation with element count.
    """
    fc = _get_fastcode()
    resolved_is_url = fc._infer_is_url(repo_source)
    name = _repo_name_from_source(repo_source, resolved_is_url)
    logger.info(f"Force re-indexing '{name}' from {repo_source}")
    result = _run_full_reindex(repo_source, resolved_is_url, repo_name=name)
    if result.get("status") == "path_not_found":
        return f"Error: Local path does not exist: {result['path']}"
    if result.get("status") != "success":
        return f"Error: Failed to re-index '{name}'."

    return f"Successfully re-indexed '{name}': {result['count']} elements indexed."


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
