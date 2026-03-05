"""
FastCode 2.0 - REST API
Complete API with all features from web_app.py
"""

import os
import platform

if platform.system() == 'Darwin':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uvicorn
import logging
from pathlib import Path
import tempfile
import zipfile
import shutil
import uuid
import json as json_module
import asyncio

from fastcode import FastCode


# Pydantic models
class LoadRepositoryRequest(BaseModel):
    source: str = Field(..., description="Repository URL or local path")
    is_url: Optional[bool] = Field(
        None,
        description="True if source is URL, False if local path. If omitted, auto-detect."
    )


class QueryRequest(BaseModel):
    question: str = Field(..., description="Question to ask about the repository")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters")
    repo_filter: Optional[List[str]] = Field(None, description="Repository names to search")
    multi_turn: bool = Field(False, description="Enable multi-turn mode")
    session_id: Optional[str] = Field(None, description="Session ID for multi-turn dialogue")


class QueryResponse(BaseModel):
    answer: str
    query: str
    context_elements: int
    sources: List[Dict[str, Any]]
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    total_tokens: Optional[int] = None
    session_id: Optional[str] = None


class LoadRepositoriesRequest(BaseModel):
    repo_names: List[str] = Field(..., description="Repository names to load from existing indexes")


class IndexMultipleRequest(BaseModel):
    sources: List[LoadRepositoryRequest] = Field(..., description="Multiple repositories to load and index")


class NewSessionResponse(BaseModel):
    session_id: str


class DeleteReposRequest(BaseModel):
    repo_names: List[str] = Field(..., description="Repository names to delete")
    delete_source: bool = Field(True, description="Also delete cloned source code in repos/")


class StatusResponse(BaseModel):
    status: str
    repo_loaded: bool
    repo_indexed: bool
    repo_info: Dict[str, Any]
    available_repositories: List[Dict[str, Any]] = Field(default_factory=list)
    loaded_repositories: List[Dict[str, Any]] = Field(default_factory=list)


# Initialize FastAPI app

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown"""
    # Startup
    logger.info("FastCode API started - system will initialize on first request")
    yield
    # Shutdown
    logger.info("FastCode API shutting down")


app = FastAPI(
    title="FastCode API",
    description="Repository-Level Code Understanding System API",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global FastCode instance
fastcode_instance: Optional[FastCode] = None

# Setup logging
log_dir = Path("./logs")
log_dir.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "api.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def _ensure_fastcode_initialized():
    """Ensure FastCode is initialized (lazy initialization)"""
    global fastcode_instance
    if fastcode_instance is None:
        logger.info("Initializing FastCode system (lazy initialization)")
        fastcode_instance = FastCode()
    return fastcode_instance



@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "FastCode API",
        "version": "2.0.0",
        "status": "running",
        "docs": "/docs",
    }


@app.get("/health")
async def health_check():
    """Lightweight health check endpoint"""
    if fastcode_instance is None:
        return {
            "status": "initializing",
            "message": "FastCode system will initialize on first use",
            "repo_loaded": False,
            "repo_indexed": False,
        }

    return {
        "status": "healthy",
        "repo_loaded": fastcode_instance.repo_loaded,
        "repo_indexed": fastcode_instance.repo_indexed,
        "multi_repo_mode": fastcode_instance.multi_repo_mode,
    }


@app.get("/status", response_model=StatusResponse)
async def get_status(full_scan: bool = False):
    """
    Get system status

    Args:
        full_scan: If True, force a full scan of available indexes (slower but fresh data)
    """
    fastcode = _ensure_fastcode_initialized()

    available_repos = fastcode.vector_store.scan_available_indexes(use_cache=not full_scan)
    loaded_repos = fastcode.list_repositories()

    return StatusResponse(
        status="ready" if fastcode.repo_indexed else "not_ready",
        repo_loaded=fastcode.repo_loaded,
        repo_indexed=fastcode.repo_indexed,
        repo_info=fastcode.repo_info,
        available_repositories=available_repos,
        loaded_repositories=loaded_repos,
    )


@app.get("/repositories")
async def list_repositories(full_scan: bool = False):
    """
    List available (indexed on disk) and loaded repositories

    Args:
        full_scan: If True, force a full scan of available indexes (slower but fresh data)
    """
    fastcode = _ensure_fastcode_initialized()

    try:
        available_repos = fastcode.vector_store.scan_available_indexes(use_cache=not full_scan)
        loaded_repos = fastcode.list_repositories()

        return {
            "status": "success",
            "available": available_repos,
            "loaded": loaded_repos,
        }
    except Exception as e:
        logger.error(f"Failed to list repositories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load")
async def load_repository(request: LoadRepositoryRequest):
    """Load a repository"""
    fastcode = _ensure_fastcode_initialized()

    try:
        logger.info(f"Loading repository: {request.source}")
        fastcode.load_repository(request.source, request.is_url)

        return {
            "status": "success",
            "message": "Repository loaded successfully",
            "repo_info": fastcode.repo_info,
        }

    except Exception as e:
        logger.error(f"Failed to load repository: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index")
async def index_repository(force: bool = False):
    """Index the loaded repository"""
    fastcode = _ensure_fastcode_initialized()

    if not fastcode.repo_loaded:
        raise HTTPException(status_code=400, detail="No repository loaded")

    try:
        logger.info("Indexing repository")
        fastcode.index_repository(force=force)

        fastcode.vector_store.invalidate_scan_cache()

        return {
            "status": "success",
            "message": "Repository indexed successfully",
            "summary": fastcode.get_repository_summary(),
        }

    except Exception as e:
        logger.error(f"Failed to index repository: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load-and-index")
async def load_and_index(request: LoadRepositoryRequest, force: bool = False):
    """Load and index repository in one call"""
    fastcode = _ensure_fastcode_initialized()

    try:
        logger.info(f"Loading repository: {request.source}")
        await asyncio.to_thread(fastcode.load_repository, request.source, request.is_url)

        logger.info("Indexing repository")
        await asyncio.to_thread(fastcode.index_repository, force=force)

        fastcode.vector_store.invalidate_scan_cache()

        return {
            "status": "success",
            "message": "Repository loaded and indexed successfully",
            "summary": fastcode.get_repository_summary(),
        }

    except Exception as e:
        logger.error(f"Failed to load and index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/load-repositories")
async def load_repositories(request: LoadRepositoriesRequest):
    """Load existing indexed repositories from cache"""
    fastcode = _ensure_fastcode_initialized()

    if not request.repo_names:
        raise HTTPException(status_code=400, detail="No repository names provided")

    try:
        logger.info(f"Loading repositories from cache: {request.repo_names}")
        success = fastcode._load_multi_repo_cache(repo_names=request.repo_names)

        if not success:
            raise HTTPException(status_code=500, detail="Failed to load repositories from cache")

        return {
            "status": "success",
            "loaded": fastcode.list_repositories(),
            "stats": fastcode.get_repository_stats(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load repositories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index-multiple")
async def index_multiple(request: IndexMultipleRequest):
    """Load and index multiple repositories"""
    fastcode = _ensure_fastcode_initialized()

    if not request.sources:
        raise HTTPException(status_code=400, detail="No repositories provided")

    try:
        logger.info(f"Indexing {len(request.sources)} repositories")
        fastcode.load_multiple_repositories([s.dict() for s in request.sources])

        fastcode.vector_store.invalidate_scan_cache()

        return {
            "status": "success",
            "message": "Repositories indexed successfully",
            "stats": fastcode.get_repository_stats(),
        }
    except Exception as e:
        logger.error(f"Failed to index multiple repositories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-zip")
async def upload_repository_zip(file: UploadFile = File(...)):
    """Upload and extract repository ZIP file"""
    fastcode = _ensure_fastcode_initialized()

    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported")

    file.file.seek(0, 2)
    file_size = file.file.tell()
    file.file.seek(0)

    max_size = 100 * 1024 * 1024  # 100MB
    if file_size > max_size:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum size is {max_size / (1024*1024)}MB")

    try:
        repo_name = file.filename.rsplit('.', 1)[0]
        for suffix in ['-main', '-master', '_main', '_master']:
            if repo_name.endswith(suffix):
                repo_name = repo_name[:-len(suffix)]
                break

        repo_workspace = getattr(fastcode.loader, "safe_repo_root", "./repos")
        repos_dir = Path(repo_workspace)
        repos_dir.mkdir(parents=True, exist_ok=True)
        repo_path = repos_dir / repo_name

        if repo_path.exists():
            fastcode.loader._backup_existing_repo(str(repo_path))

        temp_dir = tempfile.mkdtemp(prefix="fastcode_upload_")
        zip_path = Path(temp_dir) / file.filename

        logger.info(f"Saving uploaded ZIP file: {file.filename} ({file_size} bytes)")

        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        extract_dir = Path(temp_dir) / "extracted"
        extract_dir.mkdir(exist_ok=True)

        logger.info(f"Extracting ZIP file to temporary directory: {extract_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

        extracted_items = list(extract_dir.iterdir())
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            source_repo_path = extracted_items[0]
        else:
            source_repo_path = extract_dir

        logger.info(f"Moving repository to: {repo_path}")
        shutil.move(str(source_repo_path), str(repo_path))

        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up temp directory: {cleanup_error}")

        logger.info(f"Loading repository from: {repo_path}")
        fastcode.load_repository(str(repo_path), is_url=False)

        return {
            "status": "success",
            "message": f"ZIP file '{file.filename}' uploaded and extracted to repos/{repo_name}",
            "repo_info": fastcode.repo_info,
            "repo_path": str(repo_path),
        }

    except zipfile.BadZipFile:
        logger.error("Invalid ZIP file")
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except Exception as e:
        logger.error(f"Failed to upload and extract ZIP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload-and-index")
async def upload_and_index(file: UploadFile = File(...), force: bool = False):
    """Upload ZIP and index in one call"""
    fastcode = _ensure_fastcode_initialized()

    upload_result = await upload_repository_zip(file)

    if upload_result["status"] != "success":
        return upload_result

    try:
        logger.info("Indexing uploaded repository")
        fastcode.index_repository(force=force)

        fastcode.vector_store.invalidate_scan_cache()

        return {
            "status": "success",
            "message": "Repository uploaded and indexed successfully",
            "summary": fastcode.get_repository_summary(),
        }

    except Exception as e:
        logger.error(f"Failed to index uploaded repository: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query_repository(request: QueryRequest):
    """Query the repository"""
    fastcode = _ensure_fastcode_initialized()

    if not fastcode.repo_indexed:
        raise HTTPException(status_code=400, detail="Repository not indexed")

    try:
        session_id = request.session_id or str(uuid.uuid4())[:8]
        if request.multi_turn and not request.session_id:
            logger.info(f"Generated new multi-turn session: {session_id}")
        elif not request.session_id:
            logger.info(f"Generated session for single-turn request: {session_id}")

        logger.info(f"Processing query: {request.question}")
        result = await asyncio.to_thread(
            fastcode.query,
            request.question,
            request.filters,
            repo_filter=request.repo_filter,
            session_id=session_id,
            enable_multi_turn=request.multi_turn,
        )

        prompt_tokens = result.get("prompt_tokens")
        completion_tokens = result.get("completion_tokens")
        total_tokens = result.get("total_tokens")

        if total_tokens is None and prompt_tokens and completion_tokens:
            total_tokens = prompt_tokens + completion_tokens

        sources = result.get("sources", [])
        serialized_sources = [_safe_jsonable(source) for source in sources]

        return QueryResponse(
            answer=result.get("answer", ""),
            query=result.get("query", ""),
            context_elements=result.get("context_elements", 0),
            sources=serialized_sources,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            session_id=session_id,
        )

    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-stream")
async def query_repository_stream(request: QueryRequest):
    """Query the repository with streaming response (SSE)"""
    fastcode = _ensure_fastcode_initialized()

    if not fastcode.repo_indexed:
        raise HTTPException(status_code=400, detail="Repository not indexed")

    session_id = request.session_id or str(uuid.uuid4())[:8]
    if request.multi_turn and not request.session_id:
        logger.info(f"Generated new multi-turn session: {session_id}")
    elif not request.session_id:
        logger.info(f"Generated session for single-turn request: {session_id}")

    logger.info(f"Processing streaming query: {request.question}")

    async def event_generator():
        """Generate SSE events from query_stream"""
        try:
            for chunk, metadata in fastcode.query_stream(
                request.question,
                request.filters,
                repo_filter=request.repo_filter,
                session_id=session_id,
                enable_multi_turn=request.multi_turn,
            ):
                if metadata:
                    status = metadata.get("status", "")
                    if status == "retrieving":
                        event_data = {"type": "status", "status": "retrieving"}
                    elif status == "generating":
                        sources = metadata.get("sources", [])
                        serialized_sources = [_safe_jsonable(s) for s in sources]
                        event_data = {
                            "type": "status",
                            "status": "generating",
                            "sources": serialized_sources,
                            "session_id": session_id
                        }
                    elif status == "complete":
                        sources = metadata.get("sources", [])
                        serialized_sources = [_safe_jsonable(s) for s in sources]
                        event_data = {
                            "type": "done",
                            "sources": serialized_sources,
                            "context_elements": metadata.get("context_elements", 0),
                            "session_id": session_id
                        }
                    elif "error" in metadata:
                        event_data = {"type": "error", "error": metadata["error"]}
                    else:
                        continue
                    yield f"data: {json_module.dumps(event_data)}\n\n"
                elif chunk:
                    event_data = {"type": "chunk", "content": chunk}
                    yield f"data: {json_module.dumps(event_data)}\n\n"

                await asyncio.sleep(0)

        except Exception as e:
            logger.error(f"Streaming query failed: {e}")
            error_data = {"type": "error", "error": str(e)}
            yield f"data: {json_module.dumps(error_data)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


@app.get("/summary")
async def get_repository_summary():
    """Get repository summary"""
    fastcode = _ensure_fastcode_initialized()

    if not fastcode.repo_loaded:
        raise HTTPException(status_code=400, detail="No repository loaded")

    summary_payload: Dict[str, Any] = {
        "status": "success",
    }

    try:
        if fastcode.multi_repo_mode:
            summary_payload["summary"] = fastcode.get_repository_stats()
        else:
            summary_payload["summary"] = fastcode.get_repository_summary()
    except Exception as e:
        logger.error(f"Failed to build summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    return summary_payload


@app.post("/new-session", response_model=NewSessionResponse)
async def new_session(clear_session_id: Optional[str] = None):
    """Start a new conversation session"""
    fastcode = _ensure_fastcode_initialized()

    if clear_session_id:
        fastcode.delete_session(clear_session_id)

    session_id = str(uuid.uuid4())[:8]
    return NewSessionResponse(session_id=session_id)


@app.get("/sessions")
async def list_sessions():
    """List all dialogue sessions with titles (sorted by last update time)"""
    fastcode = _ensure_fastcode_initialized()
    try:
        sessions = fastcode.list_sessions()

        formatted_sessions = []
        for session in sessions:
            formatted_session = {
                "session_id": session.get("session_id", ""),
                "title": session.get("title", f"Session {session.get('session_id', '')}"),
                "total_turns": session.get("total_turns", 0),
                "created": session.get("created", 0),
                "last_updated": session.get("last_updated", 0),
            }
            formatted_sessions.append(formatted_session)

        return {"status": "success", "sessions": formatted_sessions}
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get full dialogue history for a session"""
    fastcode = _ensure_fastcode_initialized()
    try:
        history = fastcode.get_session_history(session_id) or []
        safe_history = [_safe_jsonable(turn) for turn in history]
        return {"status": "success", "session_id": session_id, "history": safe_history}
    except Exception as e:
        logger.error(f"Failed to load session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a single dialogue session"""
    fastcode = _ensure_fastcode_initialized()

    try:
        history = fastcode.get_session_history(session_id)
        if not history:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

        success = fastcode.delete_session(session_id)
        if success:
            return {
                "status": "success",
                "message": f"Session '{session_id}' deleted ({len(history)} turns)",
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to delete session")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/delete-repos")
async def delete_repositories(request: DeleteReposRequest):
    """Delete one or more repositories and all associated data"""
    fastcode = _ensure_fastcode_initialized()

    if not request.repo_names:
        raise HTTPException(status_code=400, detail="No repository names provided")

    try:
        results = []
        for repo_name in request.repo_names:
            result = fastcode.remove_repository(
                repo_name, delete_source=request.delete_source
            )
            results.append(result)
            logger.info(
                f"Deleted repository '{repo_name}': "
                f"{len(result['deleted_files'])} files, {result['freed_mb']} MB freed"
            )

        total_freed = sum(r["freed_mb"] for r in results)
        return {
            "status": "success",
            "message": f"Deleted {len(results)} repository(ies), freed {total_freed:.2f} MB",
            "results": results,
        }
    except Exception as e:
        logger.error(f"Failed to delete repositories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class CheckFreshnessRequest(BaseModel):
    repo_names: List[str] = Field(..., description="Repository names to check for updates")


class RefreshRepoRequest(BaseModel):
    repo_name: str = Field(..., description="Repository name to refresh")


@app.post("/check-repo-freshness")
async def check_repo_freshness(request: CheckFreshnessRequest):
    """Check whether indexed repositories are up-to-date with their remotes.

    Performs a lightweight git fetch + SHA comparison without modifying the index.
    """
    fastcode = _ensure_fastcode_initialized()

    results = []
    for name in request.repo_names:
        info = await asyncio.to_thread(fastcode.check_repo_for_updates, name)
        results.append(info)

    return {"status": "success", "results": results}


@app.post("/refresh-repo")
async def refresh_repo(request: RefreshRepoRequest):
    """Pull latest changes for a repository and re-index it."""
    fastcode = _ensure_fastcode_initialized()

    result = await asyncio.to_thread(fastcode.refresh_repository, request.repo_name)

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return {"status": "success", **result}


@app.post("/clear-cache")
async def clear_cache():
    """Clear cache"""
    fastcode = _ensure_fastcode_initialized()

    success = fastcode.cache_manager.clear()

    if success:
        return {"status": "success", "message": "Cache cleared"}
    else:
        return {"status": "failed", "message": "Failed to clear cache or cache disabled"}


@app.get("/cache-stats")
async def get_cache_stats():
    """Get cache statistics"""
    fastcode = _ensure_fastcode_initialized()

    stats = fastcode.cache_manager.get_stats()
    return stats


@app.post("/refresh-index-cache")
async def refresh_index_cache():
    """Force refresh the index scan cache"""
    fastcode = _ensure_fastcode_initialized()

    try:
        fastcode.vector_store.invalidate_scan_cache()
        available_repos = fastcode.vector_store.scan_available_indexes(use_cache=False)

        return {
            "status": "success",
            "message": "Index cache refreshed",
            "repository_count": len(available_repos),
        }
    except Exception as e:
        logger.error(f"Failed to refresh index cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/repository")
async def unload_repository():
    """Unload current repository"""
    global fastcode_instance

    if fastcode_instance:
        fastcode_instance.cleanup()
        fastcode_instance = FastCode()

    return {"status": "success", "message": "Repository unloaded"}


def _safe_jsonable(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable structures."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, dict):
        safe_dict = {}
        for k, v in obj.items():
            try:
                safe_dict[str(k)] = _safe_jsonable(v)
            except Exception:
                safe_dict[str(k)] = repr(v)
        return safe_dict
    if isinstance(obj, (list, tuple, set)):
        return [_safe_jsonable(v) for v in obj]
    if hasattr(obj, "to_dict"):
        try:
            return _safe_jsonable(obj.to_dict())
        except Exception:
            return {"repr": repr(obj)}
    if hasattr(obj, "__dict__"):
        try:
            return _safe_jsonable(vars(obj))
        except Exception:
            return {"repr": repr(obj)}
    return repr(obj)


def start_api(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """Start the API server"""
    logger.info(f"Starting FastCode API at http://{host}:{port}")
    logger.info(f"API documentation available at http://{host}:{port}/docs")
    uvicorn.run("api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="FastCode API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()
    start_api(host=args.host, port=args.port, reload=args.reload)
