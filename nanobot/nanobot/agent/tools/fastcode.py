"""
FastCode integration tools for nanobot.

These tools allow nanobot to interact with the FastCode backend API
for repository-level code understanding, querying, and session management.

Communication: HTTP requests to FastCode's FastAPI backend.
In Docker Compose: nanobot -> http://fastcode:8001/...
"""

import json
import os
from typing import Any

import httpx

from nanobot.agent.tools.base import Tool


def _get_fastcode_url() -> str:
    """Get FastCode API base URL from environment."""
    return os.environ.get("FASTCODE_API_URL", "http://fastcode:8001")


# ============================================================
# Tool 1: Load and Index Repository
# ============================================================

class FastCodeLoadRepoTool(Tool):
    """Load and index a code repository for querying."""

    def __init__(self, api_url: str | None = None):
        self._api_url = api_url or _get_fastcode_url()

    @property
    def name(self) -> str:
        return "fastcode_load_repo"

    @property
    def description(self) -> str:
        return (
            "Load and index a code repository into FastCode for code understanding and querying. "
            "Accepts a GitHub URL (e.g. https://github.com/user/repo) or a local directory path. "
            "After indexing, you can use fastcode_query to ask questions about the code. "
            "Indexing may take a while for large repositories."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "source": {
                    "type": "string",
                    "description": "Repository URL (e.g. https://github.com/user/repo) or local directory path",
                },
                "is_url": {
                    "type": "boolean",
                    "description": "True if source is a URL, False if local path. Default: true",
                },
            },
            "required": ["source"],
        }

    async def execute(
        self,
        source: str,
        is_url: bool = True,
        **kwargs: Any,
    ) -> str:
        try:
            async with httpx.AsyncClient(timeout=1800.0) as client:
                response = await client.post(
                    f"{self._api_url}/load-and-index",
                    json={
                        "source": source,
                        "is_url": is_url,
                    },
                )
                response.raise_for_status()
                data = response.json()
                summary = data.get("summary", {})
                msg = data.get("message", "Repository loaded and indexed")
                lines = [
                    f"✓ {msg}",
                    "",
                ]
                if isinstance(summary, dict):
                    if "total_files" in summary:
                        lines.append(f"Files: {summary['total_files']}")
                    if "total_elements" in summary:
                        lines.append(f"Code elements: {summary['total_elements']}")
                    if "languages" in summary:
                        lines.append(f"Languages: {summary['languages']}")
                else:
                    lines.append(str(summary))
                lines.append("")
                lines.append("You can now use fastcode_query to ask questions about this repository.")
                return "\n".join(lines)
        except httpx.ConnectError:
            return "Error: Cannot connect to FastCode backend. Is the FastCode service running?"
        except httpx.HTTPStatusError as e:
            return f"Error: FastCode API returned status {e.response.status_code}: {e.response.text}"
        except Exception as e:
            return f"Error loading repository: {str(e)}"


# ============================================================
# Tool 2: Query Repository (Core Tool)
# ============================================================

class FastCodeQueryTool(Tool):
    """Query a code repository using natural language."""

    def __init__(self, api_url: str | None = None):
        self._api_url = api_url or _get_fastcode_url()

    @property
    def name(self) -> str:
        return "fastcode_query"

    @property
    def description(self) -> str:
        return (
            "Ask a question about the loaded code repository. "
            "Supports natural language questions like 'How does authentication work?', "
            "'Where is the main entry point?', 'Explain the data flow', etc. "
            "Supports multi-turn dialogue: set multi_turn=true and reuse the session_id "
            "to have a continuous conversation about the code. "
            "The repository must be loaded and indexed first using fastcode_load_repo."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": "Natural language question about the code",
                },
                "multi_turn": {
                    "type": "boolean",
                    "description": "Enable multi-turn dialogue mode for follow-up questions. Default: true",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID for multi-turn dialogue. Omit to auto-generate a new one.",
                },
                "repo_filter": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Filter to specific repository names (for multi-repo mode). Optional.",
                },
            },
            "required": ["question"],
        }

    async def execute(
        self,
        question: str,
        multi_turn: bool = True,
        session_id: str | None = None,
        repo_filter: list[str] | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            payload: dict[str, Any] = {
                "question": question,
                "multi_turn": multi_turn,
            }
            if session_id:
                payload["session_id"] = session_id
            if repo_filter:
                payload["repo_filter"] = repo_filter

            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{self._api_url}/query",
                    json=payload,
                )
                response.raise_for_status()
                data = response.json()

                answer = data.get("answer", "No answer generated.")
                sid = data.get("session_id", "")
                sources = data.get("sources", [])
                total_tokens = data.get("total_tokens")

                lines = [answer]

                if sources:
                    lines.append("")
                    lines.append("--- Sources ---")
                    for i, src in enumerate(sources[:5], 1):
                        name = src.get("name", src.get("relative_path", "unknown"))
                        stype = src.get("type", "")
                        lines.append(f"  {i}. {name} ({stype})")

                if sid:
                    lines.append(f"\n[Session: {sid}]")
                if total_tokens:
                    lines.append(f"[Tokens: {total_tokens}]")

                return "\n".join(lines)

        except httpx.ConnectError:
            return "Error: Cannot connect to FastCode backend. Is the FastCode service running?"
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 400:
                return "Error: No repository indexed. Use fastcode_load_repo first."
            return f"Error: FastCode API returned status {e.response.status_code}: {e.response.text}"
        except Exception as e:
            return f"Error querying repository: {str(e)}"


# ============================================================
# Tool 3: List Repositories
# ============================================================

class FastCodeListReposTool(Tool):
    """List available and loaded repositories."""

    def __init__(self, api_url: str | None = None):
        self._api_url = api_url or _get_fastcode_url()

    @property
    def name(self) -> str:
        return "fastcode_list_repos"

    @property
    def description(self) -> str:
        return (
            "List all available (indexed on disk) and currently loaded repositories in FastCode. "
            "Shows repository names, sizes, and loading status."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
        }

    async def execute(self, **kwargs: Any) -> str:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(f"{self._api_url}/repositories")
                response.raise_for_status()
                data = response.json()

                available = data.get("available", [])
                loaded = data.get("loaded", [])

                lines = []

                if loaded:
                    lines.append(f"=== Loaded Repositories ({len(loaded)}) ===")
                    for repo in loaded:
                        name = repo.get("name", repo.get("repo_name", "unknown"))
                        elements = repo.get("total_elements", repo.get("elements", "?"))
                        lines.append(f"  [active] {name} ({elements} elements)")
                    lines.append("")

                if available:
                    lines.append(f"=== Available on Disk ({len(available)}) ===")
                    for repo in available:
                        name = repo.get("name", repo.get("repo_name", "unknown"))
                        size = repo.get("size_mb", "?")
                        lines.append(f"  {name} ({size} MB)")

                if not lines:
                    return "No repositories found. Use fastcode_load_repo to load one."

                return "\n".join(lines)

        except httpx.ConnectError:
            return "Error: Cannot connect to FastCode backend. Is the FastCode service running?"
        except Exception as e:
            return f"Error listing repositories: {str(e)}"


# ============================================================
# Tool 4: System Status
# ============================================================

class FastCodeStatusTool(Tool):
    """Check FastCode system status."""

    def __init__(self, api_url: str | None = None):
        self._api_url = api_url or _get_fastcode_url()

    @property
    def name(self) -> str:
        return "fastcode_status"

    @property
    def description(self) -> str:
        return (
            "Check the current status of the FastCode system. "
            "Shows whether repositories are loaded and indexed, "
            "system health, and available repositories."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {},
        }

    async def execute(self, **kwargs: Any) -> str:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(f"{self._api_url}/status")
                response.raise_for_status()
                data = response.json()

                status = data.get("status", "unknown")
                repo_loaded = data.get("repo_loaded", False)
                repo_indexed = data.get("repo_indexed", False)
                repo_info = data.get("repo_info", {})
                available = data.get("available_repositories", [])
                loaded = data.get("loaded_repositories", [])

                lines = [
                    "=== FastCode System Status ===",
                    f"Status: {status}",
                    f"Repository loaded: {'Yes' if repo_loaded else 'No'}",
                    f"Repository indexed: {'Yes' if repo_indexed else 'No'}",
                ]

                if repo_info and repo_info.get("name"):
                    lines.append(f"Current repo: {repo_info.get('name', 'N/A')}")

                if loaded:
                    lines.append(f"Loaded repos: {len(loaded)}")
                if available:
                    lines.append(f"Available repos on disk: {len(available)}")

                return "\n".join(lines)

        except httpx.ConnectError:
            return "Error: Cannot connect to FastCode backend. Is the FastCode service running?"
        except Exception as e:
            return f"Error checking status: {str(e)}"


# ============================================================
# Tool 5: Session Management
# ============================================================

class FastCodeSessionTool(Tool):
    """Manage FastCode dialogue sessions."""

    def __init__(self, api_url: str | None = None):
        self._api_url = api_url or _get_fastcode_url()

    @property
    def name(self) -> str:
        return "fastcode_session"

    @property
    def description(self) -> str:
        return (
            "Manage FastCode dialogue sessions for multi-turn code conversations. "
            "Actions: 'new' to create a new session, 'list' to list all sessions, "
            "'history' to view a session's conversation history, "
            "'delete' to delete a session."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["new", "list", "history", "delete"],
                    "description": "Action to perform: new, list, history, or delete",
                },
                "session_id": {
                    "type": "string",
                    "description": "Session ID (required for 'history' and 'delete' actions)",
                },
            },
            "required": ["action"],
        }

    async def execute(
        self,
        action: str,
        session_id: str | None = None,
        **kwargs: Any,
    ) -> str:
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:

                if action == "new":
                    response = await client.post(f"{self._api_url}/new-session")
                    response.raise_for_status()
                    data = response.json()
                    sid = data.get("session_id", "unknown")
                    return (
                        f"New session created: {sid}\n"
                        f"Use this session_id with fastcode_query for multi-turn dialogue."
                    )

                elif action == "list":
                    response = await client.get(f"{self._api_url}/sessions")
                    response.raise_for_status()
                    data = response.json()
                    sessions = data.get("sessions", [])
                    if not sessions:
                        return "No dialogue sessions found."
                    lines = [f"=== Dialogue Sessions ({len(sessions)}) ==="]
                    for s in sessions[:20]:
                        sid = s.get("session_id", "?")
                        title = s.get("title", "Untitled")
                        turns = s.get("total_turns", 0)
                        lines.append(f"  [{sid}] {title} ({turns} turns)")
                    return "\n".join(lines)

                elif action == "history":
                    if not session_id:
                        return "Error: session_id is required for 'history' action."
                    response = await client.get(f"{self._api_url}/session/{session_id}")
                    response.raise_for_status()
                    data = response.json()
                    history = data.get("history", [])
                    if not history:
                        return f"No history found for session '{session_id}'."
                    lines = [f"=== Session {session_id} ({len(history)} turns) ==="]
                    for i, turn in enumerate(history, 1):
                        q = turn.get("query", turn.get("question", ""))
                        a = turn.get("answer", "")
                        lines.append(f"\n--- Turn {i} ---")
                        lines.append(f"Q: {q[:200]}{'...' if len(q) > 200 else ''}")
                        lines.append(f"A: {a[:300]}{'...' if len(a) > 300 else ''}")
                    return "\n".join(lines)

                elif action == "delete":
                    if not session_id:
                        return "Error: session_id is required for 'delete' action."
                    response = await client.delete(f"{self._api_url}/session/{session_id}")
                    response.raise_for_status()
                    return f"Session '{session_id}' deleted successfully."

                else:
                    return f"Unknown action: {action}. Use: new, list, history, or delete."

        except httpx.ConnectError:
            return "Error: Cannot connect to FastCode backend. Is the FastCode service running?"
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return f"Error: Session '{session_id}' not found."
            return f"Error: FastCode API returned status {e.response.status_code}: {e.response.text}"
        except Exception as e:
            return f"Error managing session: {str(e)}"


# ============================================================
# Helper: create all FastCode tools at once
# ============================================================

def create_all_tools(api_url: str | None = None) -> list[Tool]:
    """
    Create all FastCode tools with the given API URL.

    Usage in AgentLoop._register_default_tools():
        fastcode_url = os.environ.get("FASTCODE_API_URL")
        if fastcode_url:
            from nanobot.agent.tools.fastcode import create_all_tools
            for tool in create_all_tools(api_url=fastcode_url):
                self.tools.register(tool)
    """
    url = api_url or _get_fastcode_url()
    return [
        FastCodeLoadRepoTool(api_url=url),
        FastCodeQueryTool(api_url=url),
        FastCodeListReposTool(api_url=url),
        FastCodeStatusTool(api_url=url),
        FastCodeSessionTool(api_url=url),
    ]
