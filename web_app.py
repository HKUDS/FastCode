#!/usr/bin/env python3
"""
FastCode 2.0 - Web Interface
Simple web interface for FastCode system
"""

import os
import platform

if platform.system() == 'Darwin':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

from fastapi import FastAPI, HTTPException, File, UploadFile, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import asyncio
import json as json_module
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
import uvicorn
import logging
from pathlib import Path
import tempfile
import zipfile
import shutil
import uuid

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



class StatusResponse(BaseModel):
    status: str
    repo_loaded: bool
    repo_indexed: bool
    repo_info: Dict[str, Any]
    available_repositories: List[Dict[str, Any]] = Field(default_factory=list)
    loaded_repositories: List[Dict[str, Any]] = Field(default_factory=list)


# Initialize FastAPI app
app = FastAPI(
    title="FastCode Web Interface",
    description="Repository-Level Code Understanding System",
    version="2.0.0",
)

# Mount static files for assets
assets_path = Path(__file__).parent / "assets"
if assets_path.exists():
    app.mount("/assets", StaticFiles(directory=str(assets_path)), name="assets")

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
# Create logs directory
log_dir = Path("./logs")
log_dir.mkdir(exist_ok=True)

# Configure logging with both file and console handlers
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(log_dir / "web_app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@app.on_event("startup")
async def startup_event():
    """Initialize FastCode on startup"""
    global fastcode_instance
    logger.info("Initializing FastCode system")
    fastcode_instance = FastCode()


@app.get("/", response_class=HTMLResponse)
async def get_web_interface():
    """Serve the main web interface"""
    html_file = Path(__file__).parent / "web_interface.html"
    if html_file.exists():
        return FileResponse(html_file)
    raise HTTPException(status_code=404, detail="Web interface not found")


@app.get("/api/status", response_model=StatusResponse)
async def get_status(full_scan: bool = False):
    """
    Get system status
    
    Args:
        full_scan: If True, force a full scan of available indexes (slower but fresh data)
    """
    if fastcode_instance is None:
        raise HTTPException(status_code=500, detail="FastCode not initialized")
    
    # Use cached scan by default for better performance
    available_repos = fastcode_instance.vector_store.scan_available_indexes(use_cache=not full_scan)
    loaded_repos = fastcode_instance.list_repositories()

    return StatusResponse(
        status="ready" if fastcode_instance.repo_indexed else "not_ready",
        repo_loaded=fastcode_instance.repo_loaded,
        repo_indexed=fastcode_instance.repo_indexed,
        repo_info=fastcode_instance.repo_info,
        available_repositories=available_repos,
        loaded_repositories=loaded_repos,
    )


@app.get("/api/health")
async def health_check():
    """Lightweight health check endpoint (no expensive operations)"""
    if fastcode_instance is None:
        raise HTTPException(status_code=500, detail="FastCode not initialized")
    
    return {
        "status": "healthy",
        "repo_loaded": fastcode_instance.repo_loaded,
        "repo_indexed": fastcode_instance.repo_indexed,
        "multi_repo_mode": fastcode_instance.multi_repo_mode,
    }


@app.get("/api/repositories")
async def list_repositories(full_scan: bool = False):
    """
    List available (indexed on disk) and loaded repositories
    
    Args:
        full_scan: If True, force a full scan of available indexes (slower but fresh data)
    """
    if fastcode_instance is None:
        raise HTTPException(status_code=500, detail="FastCode not initialized")
    
    try:
        # Use cached scan by default for better performance
        available_repos = fastcode_instance.vector_store.scan_available_indexes(use_cache=not full_scan)
        loaded_repos = fastcode_instance.list_repositories()
        
        return {
            "status": "success",
            "available": available_repos,
            "loaded": loaded_repos,
        }
    except Exception as e:
        logger.error(f"Failed to list repositories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/load")
async def load_repository(request: LoadRepositoryRequest):
    """Load a repository"""
    if fastcode_instance is None:
        raise HTTPException(status_code=500, detail="FastCode not initialized")
    
    try:
        logger.info(f"Loading repository: {request.source}")
        fastcode_instance.load_repository(request.source, request.is_url)
        
        return {
            "status": "success",
            "message": "Repository loaded successfully",
            "repo_info": fastcode_instance.repo_info,
        }
    
    except Exception as e:
        logger.error(f"Failed to load repository: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/index")
async def index_repository(force: bool = False):
    """Index the loaded repository"""
    if fastcode_instance is None:
        raise HTTPException(status_code=500, detail="FastCode not initialized")
    
    if not fastcode_instance.repo_loaded:
        raise HTTPException(status_code=400, detail="No repository loaded")
    
    try:
        logger.info("Indexing repository")
        fastcode_instance.index_repository(force=force)
        
        # Invalidate scan cache since we just added/updated an index
        fastcode_instance.vector_store.invalidate_scan_cache()
        
        return {
            "status": "success",
            "message": "Repository indexed successfully",
            "summary": fastcode_instance.get_repository_summary(),
        }
    
    except Exception as e:
        logger.error(f"Failed to index repository: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/index-multiple")
async def index_multiple(request: IndexMultipleRequest):
    """Load and index multiple repositories"""
    if fastcode_instance is None:
        raise HTTPException(status_code=500, detail="FastCode not initialized")
    
    if not request.sources:
        raise HTTPException(status_code=400, detail="No repositories provided")
    
    try:
        logger.info(f"Indexing {len(request.sources)} repositories")
        fastcode_instance.load_multiple_repositories([s.dict() for s in request.sources])
        
        # Invalidate scan cache since we just added/updated indexes
        fastcode_instance.vector_store.invalidate_scan_cache()
        
        return {
            "status": "success",
            "message": "Repositories indexed successfully",
            "stats": fastcode_instance.get_repository_stats(),
        }
    except Exception as e:
        logger.error(f"Failed to index multiple repositories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/load-and-index")
async def load_and_index(request: LoadRepositoryRequest, force: bool = False):
    """Load and index repository in one call"""
    if fastcode_instance is None:
        raise HTTPException(status_code=500, detail="FastCode not initialized")
    
    try:
        logger.info(f"Loading repository: {request.source}")
        fastcode_instance.load_repository(request.source, request.is_url)
        
        logger.info("Indexing repository")
        fastcode_instance.index_repository(force=force)
        
        # Invalidate scan cache since we just added/updated an index
        fastcode_instance.vector_store.invalidate_scan_cache()
        
        return {
            "status": "success",
            "message": "Repository loaded and indexed successfully",
            "summary": fastcode_instance.get_repository_summary(),
        }
    
    except Exception as e:
        logger.error(f"Failed to load and index: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/load-repositories")
async def load_repositories(request: LoadRepositoriesRequest):
    """Load existing indexed repositories from cache"""
    if fastcode_instance is None:
        raise HTTPException(status_code=500, detail="FastCode not initialized")
    
    if not request.repo_names:
        raise HTTPException(status_code=400, detail="No repository names provided")
    
    try:
        logger.info(f"Loading repositories from cache: {request.repo_names}")
        success = fastcode_instance._load_multi_repo_cache(repo_names=request.repo_names)
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to load repositories from cache")
        
        return {
            "status": "success",
            "loaded": fastcode_instance.list_repositories(),
            "stats": fastcode_instance.get_repository_stats(),
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to load repositories: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-zip")
async def upload_repository_zip(file: UploadFile = File(...)):
    """Upload and extract repository ZIP file to repos directory (same as URL download)"""
    if fastcode_instance is None:
        raise HTTPException(status_code=500, detail="FastCode not initialized")
    
    # Validate file type
    if not file.filename.endswith('.zip'):
        raise HTTPException(status_code=400, detail="Only ZIP files are supported")
    
    # Check file size (100MB limit)
    file.file.seek(0, 2)  # Seek to end
    file_size = file.file.tell()
    file.file.seek(0)  # Reset to beginning
    
    max_size = 100 * 1024 * 1024  # 100MB
    if file_size > max_size:
        raise HTTPException(status_code=400, detail=f"File too large. Maximum size is {max_size / (1024*1024)}MB")
    
    try:
        # Extract repository name from ZIP filename (remove .zip extension)
        repo_name = file.filename.rsplit('.', 1)[0]
        # Clean repository name (remove common suffixes like -main, -master)
        for suffix in ['-main', '-master', '_main', '_master']:
            if repo_name.endswith(suffix):
                repo_name = repo_name[:-len(suffix)]
                break
        
        # Use configured repository workspace directory
        repo_workspace = getattr(fastcode_instance.loader, "safe_repo_root", "./repos")
        repos_dir = Path(repo_workspace)
        repos_dir.mkdir(parents=True, exist_ok=True)
        
        # Target repository path in repos directory
        repo_path = repos_dir / repo_name
        
        # Backup existing repository before overwrite
        if repo_path.exists():
            fastcode_instance.loader._backup_existing_repo(str(repo_path))
        
        # Create temporary directory for ZIP extraction
        temp_dir = tempfile.mkdtemp(prefix="fastcode_upload_")
        zip_path = Path(temp_dir) / file.filename
        
        logger.info(f"Saving uploaded ZIP file: {file.filename} ({file_size} bytes)")
        
        # Save uploaded file to temp directory
        with open(zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract to temporary directory first
        extract_dir = Path(temp_dir) / "extracted"
        extract_dir.mkdir(exist_ok=True)
        
        logger.info(f"Extracting ZIP file to temporary directory: {extract_dir}")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)
        
        # Find the actual repository root (in case ZIP contains a single root folder)
        extracted_items = list(extract_dir.iterdir())
        if len(extracted_items) == 1 and extracted_items[0].is_dir():
            source_repo_path = extracted_items[0]
        else:
            source_repo_path = extract_dir
        
        # Move the extracted repository to repos directory
        logger.info(f"Moving repository to: {repo_path}")
        shutil.move(str(source_repo_path), str(repo_path))
        
        # Clean up temporary directory
        try:
            shutil.rmtree(temp_dir)
            logger.info(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as cleanup_error:
            logger.warning(f"Failed to clean up temp directory: {cleanup_error}")
        
        # Load repository from repos directory (same as URL download)
        logger.info(f"Loading repository from: {repo_path}")
        fastcode_instance.load_repository(str(repo_path), is_url=False)
        
        return {
            "status": "success",
            "message": f"ZIP file '{file.filename}' uploaded and extracted to repos/{repo_name}",
            "repo_info": fastcode_instance.repo_info,
            "repo_path": str(repo_path),
        }
    
    except zipfile.BadZipFile:
        logger.error("Invalid ZIP file")
        raise HTTPException(status_code=400, detail="Invalid ZIP file")
    except Exception as e:
        logger.error(f"Failed to upload and extract ZIP: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload-and-index")
async def upload_and_index(file: UploadFile = File(...), force: bool = False):
    """Upload ZIP and index in one call"""
    if fastcode_instance is None:
        raise HTTPException(status_code=500, detail="FastCode not initialized")
    
    # First upload and load
    upload_result = await upload_repository_zip(file)
    
    if upload_result["status"] != "success":
        return upload_result
    
    # Then index
    try:
        logger.info("Indexing uploaded repository")
        fastcode_instance.index_repository(force=force)
        
        # Invalidate scan cache since we just added an index
        fastcode_instance.vector_store.invalidate_scan_cache()
        
        return {
            "status": "success",
            "message": "Repository uploaded and indexed successfully",
            "summary": fastcode_instance.get_repository_summary(),
        }
    
    except Exception as e:
        logger.error(f"Failed to index uploaded repository: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/query", response_model=QueryResponse)
async def query_repository(request: QueryRequest):
    """Query the repository"""
    if fastcode_instance is None:
        raise HTTPException(status_code=500, detail="FastCode not initialized")
    
    if not fastcode_instance.repo_indexed:
        raise HTTPException(status_code=400, detail="Repository not indexed")
    
    try:
        # Derive session handling for both modes (single-turn keeps a session for history)
        session_id = request.session_id or str(uuid.uuid4())[:8]
        if request.multi_turn and not request.session_id:
            logger.info(f"Generated new multi-turn session: {session_id}")
        elif not request.session_id:
            logger.info(f"Generated session for single-turn request: {session_id}")

        logger.info(f"Processing query: {request.question}")
        result = fastcode_instance.query(
            request.question, 
            request.filters,
            repo_filter=request.repo_filter,
            session_id=session_id,
            enable_multi_turn=request.multi_turn,
        )
        
        # Extract token information
        prompt_tokens = result.get("prompt_tokens")
        completion_tokens = result.get("completion_tokens")
        total_tokens = result.get("total_tokens")
        
        # Calculate total if not provided
        if total_tokens is None and prompt_tokens and completion_tokens:
            total_tokens = prompt_tokens + completion_tokens
        
        # Ensure sources are JSON-serializable (query() already saves to cache)
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


@app.post("/api/query-stream")
async def query_repository_stream(request: QueryRequest):
    """Query the repository with streaming response (SSE)"""
    if fastcode_instance is None:
        raise HTTPException(status_code=500, detail="FastCode not initialized")

    if not fastcode_instance.repo_indexed:
        raise HTTPException(status_code=400, detail="Repository not indexed")

    # Derive session handling
    session_id = request.session_id or str(uuid.uuid4())[:8]
    if request.multi_turn and not request.session_id:
        logger.info(f"Generated new multi-turn session: {session_id}")
    elif not request.session_id:
        logger.info(f"Generated session for single-turn request: {session_id}")

    logger.info(f"Processing streaming query: {request.question}")

    async def event_generator():
        """Generate SSE events from query_stream"""
        try:
            for chunk, metadata in fastcode_instance.query_stream(
                request.question,
                request.filters,
                repo_filter=request.repo_filter,
                session_id=session_id,
                enable_multi_turn=request.multi_turn,
            ):
                if metadata:
                    # Send metadata as JSON event
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
                    # Send text chunk
                    event_data = {"type": "chunk", "content": chunk}
                    yield f"data: {json_module.dumps(event_data)}\n\n"

                # Small delay to allow browser to process
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


@app.get("/api/summary")
async def get_repository_summary():
    """Get repository summary"""
    if fastcode_instance is None:
        raise HTTPException(status_code=500, detail="FastCode not initialized")
    
    if not fastcode_instance.repo_loaded:
        raise HTTPException(status_code=400, detail="No repository loaded")
    
    summary_payload: Dict[str, Any] = {
        "status": "success",
    }
    
    try:
        if fastcode_instance.multi_repo_mode:
            summary_payload["summary"] = fastcode_instance.get_repository_stats()
        else:
            summary_payload["summary"] = fastcode_instance.get_repository_summary()
    except Exception as e:
        logger.error(f"Failed to build summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    return summary_payload


@app.post("/api/clear-cache")
async def clear_cache():
    """Clear cache"""
    if fastcode_instance is None:
        raise HTTPException(status_code=500, detail="FastCode not initialized")
    
    success = fastcode_instance.cache_manager.clear()
    
    if success:
        return {"status": "success", "message": "Cache cleared"}
    else:
        return {"status": "failed", "message": "Failed to clear cache or cache disabled"}


@app.post("/api/refresh-index-cache")
async def refresh_index_cache():
    """Force refresh the index scan cache"""
    if fastcode_instance is None:
        raise HTTPException(status_code=500, detail="FastCode not initialized")
    
    try:
        # Invalidate cache first
        fastcode_instance.vector_store.invalidate_scan_cache()
        
        # Perform fresh scan
        available_repos = fastcode_instance.vector_store.scan_available_indexes(use_cache=False)
        
        return {
            "status": "success",
            "message": "Index cache refreshed",
            "repository_count": len(available_repos),
        }
    except Exception as e:
        logger.error(f"Failed to refresh index cache: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/new-session", response_model=NewSessionResponse)
async def new_session(clear_session_id: Optional[str] = None):
    """Start a new conversation session"""
    if fastcode_instance is None:
        raise HTTPException(status_code=500, detail="FastCode not initialized")
    
    if clear_session_id:
        fastcode_instance.delete_session(clear_session_id)
    
    session_id = str(uuid.uuid4())[:8]
    return NewSessionResponse(session_id=session_id)


def _safe_jsonable(obj: Any) -> Any:
    """Recursively convert objects to JSON-serializable structures."""
    # Primitives
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    # Dict
    if isinstance(obj, dict):
        safe_dict = {}
        for k, v in obj.items():
            try:
                safe_dict[str(k)] = _safe_jsonable(v)
            except Exception:
                safe_dict[str(k)] = repr(v)
        return safe_dict
    # List / tuple / set
    if isinstance(obj, (list, tuple, set)):
        return [_safe_jsonable(v) for v in obj]
    # Has to_dict
    if hasattr(obj, "to_dict"):
        try:
            return _safe_jsonable(obj.to_dict())
        except Exception:
            return {"repr": repr(obj)}
    # Has __dict__
    if hasattr(obj, "__dict__"):
        try:
            return _safe_jsonable(vars(obj))
        except Exception:
            return {"repr": repr(obj)}
    # Fallback
    return repr(obj)


@app.get("/api/sessions")
async def list_sessions():
    """List all dialogue sessions with titles (sorted by last update time)"""
    if fastcode_instance is None:
        raise HTTPException(status_code=500, detail="FastCode not initialized")
    try:
        sessions = fastcode_instance.list_sessions()

        # Format sessions for better display
        formatted_sessions = []
        for session in sessions:
            formatted_session = {
                "session_id": session.get("session_id", ""),
                "title": session.get("title", f"Session {session.get('session_id', '')}"),
                "total_turns": session.get("total_turns", 0),
                "created": session.get("created", 0),
                "last_updated": session.get("last_updated", 0),
                "multi_turn": session.get("multi_turn", False),
            }
            formatted_sessions.append(formatted_session)

        return {"status": "success", "sessions": formatted_sessions}
    except Exception as e:
        logger.error(f"Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/session/{session_id}")
async def get_session(session_id: str):
    """Get full dialogue history for a session"""
    if fastcode_instance is None:
        raise HTTPException(status_code=500, detail="FastCode not initialized")
    try:
        history = fastcode_instance.get_session_history(session_id) or []

        # Ensure JSON-serializable history
        safe_history = [_safe_jsonable(turn) for turn in history]

        # Determine if this session was multi-turn from session index
        session_index = fastcode_instance.cache_manager._get_session_index(session_id)
        multi_turn = session_index.get("multi_turn", False) if session_index else False

        return {"status": "success", "session_id": session_id, "history": safe_history, "multi_turn": multi_turn}
    except Exception as e:
        logger.error(f"Failed to load session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


class DeleteReposRequest(BaseModel):
    repo_names: List[str] = Field(..., description="Repository names to delete")
    delete_source: bool = Field(True, description="Also delete cloned source code in repos/")


@app.post("/api/delete-repos")
async def delete_repositories(request: DeleteReposRequest):
    """Delete one or more repositories and all associated data"""
    if fastcode_instance is None:
        raise HTTPException(status_code=500, detail="FastCode not initialized")

    if not request.repo_names:
        raise HTTPException(status_code=400, detail="No repository names provided")

    try:
        results = []
        for repo_name in request.repo_names:
            result = fastcode_instance.remove_repository(
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


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a single dialogue session"""
    if fastcode_instance is None:
        raise HTTPException(status_code=500, detail="FastCode not initialized")

    try:
        history = fastcode_instance.get_session_history(session_id)
        if not history:
            raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")

        success = fastcode_instance.delete_session(session_id)
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


def start_web_app(host: str = "127.0.0.1", port: int = 5777, reload: bool = False):
    """Start the web application server"""
    logger.info(f"Starting FastCode Web Interface at http://{host}:{port}")
    uvicorn.run("web_app:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="FastCode Web Interface")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to (default: localhost)")
    parser.add_argument("--port", type=int, default=5777, help="Port to bind to (default: 5777)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    start_web_app(host=args.host, port=args.port, reload=args.reload)