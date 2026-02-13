#!/usr/bin/env python3
"""
FastCode 2.0 - Command Line Interface
"""

import os
import platform

if platform.system() == 'Darwin':
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'

import click
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from fastcode import FastCode


@click.group()
def cli():
    """FastCode - Repository-Level Code Understanding System"""
    pass


@cli.command()
@click.option('--repo-url', '-u', help='Repository URL to clone')
@click.option('--repo-path', '-p', help='Local repository path')
@click.option('--repo-zip', '-z', help='ZIP file containing repository')
@click.option('--query', '-q', required=True, help='Question to ask about the repository')
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--output', '-o', help='Output file (default: stdout)')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
@click.option('--load-cache', is_flag=True, help='Load from existing index cache (multi-repo mode)')
@click.option('--repos', '-r', multiple=True, help='Specific repositories to search in multi-repo mode')
def query(repo_url, repo_path, repo_zip, query, config, output, verbose, load_cache, repos):
    """Query a repository with a question (supports both single and multi-repo modes)"""
    
    # Initialize FastCode
    fastcode = FastCode(config_path=config)
    
    try:
        # Multi-repo mode with cache
        if load_cache:
            # Determine which repositories to load
            repo_names_to_load = list(repos) if repos else None
            if repo_names_to_load:
                click.echo(f"Loading repositories from cache: {', '.join(repo_names_to_load)}...")
            else:
                click.echo("Loading multi-repository index from cache...")
            
            if not fastcode._load_multi_repo_cache(repo_names=repo_names_to_load):
                click.echo("Error: Failed to load multi-repo cache", err=True)
                sys.exit(1)
            
            # Get available repositories
            available_repos = fastcode.vector_store.get_repository_names()
            
            if not available_repos:
                click.echo("Error: No repositories found in cache", err=True)
                sys.exit(1)
            
            # Determine which repositories to query
            repo_filter = None
            if repos:
                invalid_repos = [r for r in repos if r not in available_repos]
                if invalid_repos:
                    click.echo(f"Warning: Unknown repositories: {', '.join(invalid_repos)}")
                repo_filter = [r for r in repos if r in available_repos]
                if repo_filter:
                    click.echo(f"Searching in repositories: {', '.join(repo_filter)}")
            else:
                click.echo(f"Searching in all {len(available_repos)} repositories")
            
            # Query
            click.echo(f"\nProcessing query: {query}\n")
            result = fastcode.query(query, repo_filter=repo_filter)
        
        # Single repository mode
        else:
            # Validate input
            if not repo_url and not repo_path and not repo_zip:
                click.echo("Error: Either --repo-url, --repo-path, --repo-zip, or --load-cache must be provided", err=True)
                sys.exit(1)
            
            # Check for conflicting options
            provided_options = sum([bool(repo_url), bool(repo_path), bool(repo_zip)])
            if provided_options > 1:
                click.echo("Error: Only one of --repo-url, --repo-path, or --repo-zip can be specified", err=True)
                sys.exit(1)
            
            # Load repository
            if repo_zip:
                click.echo(f"Loading repository from ZIP: {repo_zip}")
                fastcode.load_repository(repo_zip, is_url=False, is_zip=True)
            elif repo_url:
                click.echo(f"Loading repository from URL: {repo_url}")
                fastcode.load_repository(repo_url, is_url=True)
            else:
                click.echo(f"Loading repository from path: {repo_path}")
                fastcode.load_repository(repo_path, is_url=False)
            
            # Index repository
            click.echo("Indexing repository...")
            fastcode.index_repository()
            
            # Get summary
            if verbose:
                summary = fastcode.get_repository_summary()
                click.echo(f"\n{summary}\n")
            
            # Query
            click.echo(f"Processing query: {query}\n")
            result = fastcode.query(query)
        
        # Format output
        formatted = fastcode.answer_generator.format_answer_with_sources(result)
        
        # Output result
        if output:
            with open(output, 'w') as f:
                f.write(formatted)
            click.echo(f"Result saved to {output}")
        else:
            click.echo(formatted)
        
        # Cleanup
        if not load_cache:
            fastcode.cleanup()
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--repo-url', '-u', help='Repository URL to clone')
@click.option('--repo-path', '-p', help='Local repository path')
@click.option('--repo-zip', '-z', help='ZIP file containing repository')
@click.option('--config', '-c', help='Path to configuration file')
def index(repo_url, repo_path, repo_zip, config):
    """Index a repository (without querying)"""
    
    if not repo_url and not repo_path and not repo_zip:
        click.echo("Error: Either --repo-url, --repo-path, or --repo-zip must be provided", err=True)
        sys.exit(1)
    
    # Check for conflicting options
    provided_options = sum([bool(repo_url), bool(repo_path), bool(repo_zip)])
    if provided_options > 1:
        click.echo("Error: Only one of --repo-url, --repo-path, or --repo-zip can be specified", err=True)
        sys.exit(1)
    
    fastcode = FastCode(config_path=config)
    
    try:
        # Load repository
        if repo_zip:
            click.echo(f"Loading repository from ZIP: {repo_zip}")
            fastcode.load_repository(repo_zip, is_url=False, is_zip=True)
        elif repo_url:
            click.echo(f"Loading repository from URL: {repo_url}")
            fastcode.load_repository(repo_url, is_url=True)
        else:
            click.echo(f"Loading repository from path: {repo_path}")
            fastcode.load_repository(repo_path, is_url=False)
        
        # Index
        click.echo("Indexing repository...")
        fastcode.index_repository()
        
        # Show summary
        summary = fastcode.get_repository_summary()
        click.echo(f"\n{summary}")
        click.echo("\nIndexing complete!")
        
        fastcode.cleanup()
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--repo-url', '-u', help='Repository URL to clone')
@click.option('--repo-path', '-p', help='Local repository path')
@click.option('--repo-zip', '-z', help='ZIP file containing repository')
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--load-cache', is_flag=True, help='Load from multi-repo cache for multi-repo mode')
@click.option('--repos', '-r', multiple=True, help='Specific repositories to load from cache (omit to load all)')
@click.option('--multi-turn', is_flag=True, help='Enable multi-turn dialogue mode')
@click.option('--session-id', '-s', help='Session ID for multi-turn dialogue (auto-generated if not provided)')
@click.option('--agency/--no-agency', default=None, help='Enable/disable agency mode (default: auto based on query intent)')
def interactive(repo_url, repo_path, repo_zip, config, load_cache, repos, multi_turn, session_id, agency):
    """Start interactive query session (supports single and multi-repo modes)"""
    
    fastcode = FastCode(config_path=config)
    
    # Generate session ID if multi-turn is enabled and no session_id provided
    if multi_turn and not session_id:
        import uuid
        session_id = str(uuid.uuid4())[:8]
        click.echo(f"Multi-turn mode enabled. Session ID: {session_id}")
    
    try:
        # Multi-repo mode
        if load_cache:
            if repos:
                click.echo(f"Loading repositories from cache: {', '.join(repos)}...")
            else:
                click.echo("Loading all available repositories from cache...")
            
            # Load specific repositories or all if none specified
            repo_filter = list(repos) if repos else None
            if not fastcode._load_multi_repo_cache(repo_names=repo_filter):
                click.echo("Error: Failed to load multi-repo cache", err=True)
                sys.exit(1)
            
            # Show available repositories
            repo_names = fastcode.vector_store.get_repository_names()
            click.echo(f"\nLoaded {len(repo_names)} repositories:")
            for repo in repo_names:
                click.echo(f"  - {repo}")
            click.echo("\nYou can query across all repositories or specify repositories with the format:")
            click.echo("  @repo1,repo2 your question here")
        
        # Single repository mode
        else:
            if not repo_url and not repo_path and not repo_zip:
                click.echo("Error: Either --repo-url, --repo-path, --repo-zip, or --load-cache must be provided", err=True)
                sys.exit(1)
            
            # Check for conflicting options
            provided_options = sum([bool(repo_url), bool(repo_path), bool(repo_zip)])
            if provided_options > 1:
                click.echo("Error: Only one of --repo-url, --repo-path, or --repo-zip can be specified", err=True)
                sys.exit(1)
            
            # Load and index repository
            if repo_zip:
                click.echo(f"Loading repository from ZIP: {repo_zip}")
                fastcode.load_repository(repo_zip, is_url=False, is_zip=True)
            elif repo_url:
                click.echo(f"Loading repository from URL: {repo_url}")
                fastcode.load_repository(repo_url, is_url=True)
            else:
                click.echo(f"Loading repository from path: {repo_path}")
                fastcode.load_repository(repo_path, is_url=False)
            
            click.echo("Indexing repository...")
            fastcode.index_repository()
            
            summary = fastcode.get_repository_summary()
            click.echo(f"\n{summary}\n")
        
        # Interactive loop
        click.echo("=" * 60)
        if multi_turn:
            click.echo("FastCode Interactive Mode (Multi-turn)")
            click.echo(f"Session ID: {session_id}")
            click.echo("Your conversations will be tracked across turns.")
        else:
            click.echo("FastCode Interactive Mode (Single-turn)")
        
        # Show agency mode status
        if agency is True:
            click.echo("Agency Mode: ENABLED (forced on)")
        elif agency is False:
            click.echo("Agency Mode: DISABLED (forced off)")
        else:
            click.echo("Agency Mode: AUTO (activates based on query intent)")
        
        click.echo("\nSpecial commands:")
        if multi_turn:
            click.echo("  'history' - Show dialogue history")
            click.echo("  'new-session' - Start a new session")
        click.echo("  'agency on' - Force enable agency mode")
        click.echo("  'agency off' - Force disable agency mode")
        click.echo("  'agency auto' - Use automatic intent-based activation")
        click.echo("  'agency status' - Show current agency mode setting")
        click.echo("  'quit' or 'exit' - Exit interactive mode")
        click.echo("=" * 60 + "\n")
        
        while True:
            try:
                query = click.prompt("\nYour question", type=str)
                
                if query.lower() in ['quit', 'exit', 'q']:
                    break
                
                # Handle special commands
                # Agency mode commands (available in all modes)
                if query.lower() == 'agency on':
                    agency = True
                    click.echo("✓ Agency mode: ENABLED (forced on)")
                    click.echo("  All queries will use agency mode for accurate and comprehensive retrieval")
                    continue
                
                elif query.lower() == 'agency off':
                    agency = False
                    click.echo("✓ Agency mode: DISABLED (forced off)")
                    click.echo("  All queries will use standard fast retrieval")
                    continue
                
                elif query.lower() == 'agency auto':
                    agency = None
                    click.echo("✓ Agency mode: AUTO")
                    click.echo("  Agency mode will activate automatically based on query intent")
                    click.echo("  Trigger intents: implement, debug, understand, trace, refactor, architecture")
                    continue
                
                elif query.lower() == 'agency status':
                    if agency is True:
                        status = "ENABLED (forced on)"
                        detail = "All queries use accurate + association agents"
                    elif agency is False:
                        status = "DISABLED (forced off)"
                        detail = "All queries use standard fast retrieval"
                    else:
                        status = "AUTO"
                        detail = "Activates for: implement, debug, understand, trace, refactor, architecture"
                    
                    click.echo(f"\n=== Agency Mode Status ===")
                    click.echo(f"Mode: {status}")
                    click.echo(f"Detail: {detail}")
                    
                    # Check if agents are initialized
                    if hasattr(fastcode.retriever, 'accurate_agent') and fastcode.retriever.accurate_agent:
                        click.echo("Agents: ✓ Initialized")
                    else:
                        click.echo("Agents: ✗ Not initialized (will initialize on first agency query)")
                    continue
                
                # Multi-turn specific commands
                if multi_turn:
                    if query.lower() == 'history':
                        history = fastcode.get_session_history(session_id)
                        if not history:
                            click.echo("No conversation history yet.")
                        else:
                            click.echo(f"\n=== Session History ({len(history)} turns) ===")
                            for turn in history:
                                turn_num = turn.get("turn_number")
                                turn_query = turn.get("query")
                                turn_answer = turn.get("answer", "")[:100] + "..."
                                click.echo(f"\nTurn {turn_num}:")
                                click.echo(f"  Q: {turn_query}")
                                click.echo(f"  A: {turn_answer}")
                        continue
                    
                    elif query.lower() == 'new-session':
                        import uuid
                        session_id = str(uuid.uuid4())[:8]
                        click.echo(f"Started new session: {session_id}")
                        continue
                
                if not query.strip():
                    continue
                
                # Parse repository filter from query (format: @repo1,repo2 question)
                repo_filter = None
                if load_cache and query.startswith('@'):
                    parts = query.split(' ', 1)
                    if len(parts) == 2:
                        repo_names_str = parts[0][1:]  # Remove @
                        query = parts[1]
                        repo_filter = [r.strip() for r in repo_names_str.split(',')]
                        click.echo(f"Searching in: {', '.join(repo_filter)}")
                
                # Show processing indicator with agency mode info
                if agency is True:
                    click.echo("\nProcessing with agency mode (accurate + association agents)...\n")
                elif agency is False:
                    click.echo("\nProcessing with standard retrieval...\n")
                else:
                    click.echo("\nProcessing...\n")
                
                # Call query with multi-turn and agency parameters
                result = fastcode.query(
                    query, 
                    repo_filter=repo_filter,
                    session_id=session_id if multi_turn else None,
                    enable_multi_turn=multi_turn,
                    use_agency_mode=agency
                )
                
                formatted = fastcode.answer_generator.format_answer_with_sources(result)
                click.echo(formatted)
                
                # Show turn number in multi-turn mode
                if multi_turn:
                    turn_num = fastcode._get_next_turn_number(session_id) - 1
                    click.echo(f"\n[Turn {turn_num} saved]")
                
            except (KeyboardInterrupt, EOFError):
                break
        
        click.echo("\nGoodbye!")
        if not load_cache:
            fastcode.cleanup()
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
def clear_cache():
    """Clear all cached data"""
    fastcode = FastCode()
    
    if fastcode.cache_manager.clear():
        click.echo("Cache cleared successfully")
    else:
        click.echo("Failed to clear cache or cache is disabled")


@cli.command()
def cache_stats():
    """Show cache statistics (query result cache, not repository indexes)"""
    fastcode = FastCode()
    stats = fastcode.cache_manager.get_stats()
    
    click.echo("=" * 60)
    click.echo("Query Result Cache Statistics")
    click.echo("=" * 60)
    click.echo(f"Enabled: {stats.get('enabled', False)}")
    
    if stats.get('enabled'):
        click.echo(f"Backend: {stats.get('backend', 'unknown')}")
        click.echo(f"Cached Queries: {stats.get('items', 0)}")
        size_mb = stats.get('size', 0) / (1024 * 1024)
        click.echo(f"Cache Size: {size_mb:.2f} MB")
        click.echo("\nNote: This shows query result cache, not repository indexes.")
        click.echo("Use 'repo-stats' to see repository index statistics.")


@cli.command()
@click.argument('repo_name')
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
@click.option('--keep-source', is_flag=True, help='Keep cloned source code in repos/')
def remove_repo(repo_name, config, confirm, keep_source):
    """Remove a repository and all its data (index, BM25, graphs, overview, source)"""

    fastcode = FastCode(config_path=config)
    persist_dir = fastcode.vector_store.persist_dir
    repo_root = fastcode.config.get("repo_root", "./repos")

    # Check if any data exists for this repository
    file_patterns = [
        f"{repo_name}.faiss",
        f"{repo_name}_metadata.pkl",
        f"{repo_name}_bm25.pkl",
        f"{repo_name}_graphs.pkl",
    ]
    existing_files = []
    total_size = 0
    for fname in file_patterns:
        fpath = os.path.join(persist_dir, fname)
        if os.path.exists(fpath):
            size = os.path.getsize(fpath)
            existing_files.append((fname, size))
            total_size += size

    repo_dir = os.path.join(repo_root, repo_name)
    has_source = os.path.isdir(repo_dir)

    if not existing_files and not has_source:
        click.echo(f"Error: Repository '{repo_name}' not found", err=True)
        sys.exit(1)

    # Confirmation
    if not confirm:
        click.echo(f"Repository: {repo_name}")
        click.echo(f"\nData to be deleted:")
        for fname, size in existing_files:
            click.echo(f"  - {fname} ({size / (1024*1024):.2f} MB)")

        # Check overview
        overviews = fastcode.vector_store.load_repo_overviews()
        if repo_name in overviews:
            click.echo(f"  - repo_overviews.pkl (entry for {repo_name})")

        if has_source and not keep_source:
            click.echo(f"  - repos/{repo_name}/ (source code)")

        click.echo(f"\nTotal index size: {total_size / (1024*1024):.2f} MB")

        if not click.confirm("\nAre you sure you want to remove this repository?"):
            click.echo("Cancelled.")
            return

    # Remove using the comprehensive method
    try:
        result = fastcode.remove_repository(repo_name, delete_source=not keep_source)
        click.echo(f"✓ Successfully removed repository '{repo_name}'")
        for f in result["deleted_files"]:
            click.echo(f"  - {f}")
        click.echo(f"  Freed {result['freed_mb']:.2f} MB of disk space")
    except Exception as e:
        click.echo(f"Error removing repository: {e}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', help='Path to configuration file')
def clean_indices(config):
    """Clean up orphaned index files"""
    
    fastcode = FastCode(config_path=config)
    # fastcode = FastCode()
    persist_dir = fastcode.vector_store.persist_dir
    
    if not os.path.exists(persist_dir):
        click.echo("No vector store directory found")
        return
    
    # Find all .faiss files
    faiss_files = [f for f in os.listdir(persist_dir) if f.endswith('.faiss')]
    
    orphaned = []
    valid = []
    
    for faiss_file in faiss_files:
        repo_name = faiss_file.replace('.faiss', '')
        metadata_file = f"{repo_name}_metadata.pkl"
        
        if metadata_file not in os.listdir(persist_dir):
            orphaned.append(faiss_file)
        else:
            valid.append(repo_name)
    
    # Find orphaned metadata files
    metadata_files = [f for f in os.listdir(persist_dir) if f.endswith('_metadata.pkl')]
    
    for metadata_file in metadata_files:
        repo_name = metadata_file.replace('_metadata.pkl', '')
        faiss_file = f"{repo_name}.faiss"
        
        if faiss_file not in os.listdir(persist_dir):
            if metadata_file not in orphaned:
                orphaned.append(metadata_file)
    
    if not orphaned:
        click.echo("✓ No orphaned files found")
        click.echo(f"  {len(valid)} valid repositories in index")
        return
    
    click.echo(f"Found {len(orphaned)} orphaned file(s):")
    total_size = 0
    for file in orphaned:
        file_path = os.path.join(persist_dir, file)
        size = os.path.getsize(file_path) / (1024 * 1024)
        total_size += size
        click.echo(f"  - {file} ({size:.2f} MB)")
    
    click.echo(f"\nTotal size: {total_size:.2f} MB")
    
    if click.confirm("\nRemove these orphaned files?"):
        for file in orphaned:
            try:
                os.remove(os.path.join(persist_dir, file))
                click.echo(f"  ✓ Removed {file}")
            except Exception as e:
                click.echo(f"  ✗ Failed to remove {file}: {e}")
        
        click.echo(f"\n✓ Cleanup complete. Freed {total_size:.2f} MB")
    else:
        click.echo("Cancelled.")


@cli.command()
@click.option('--repo-urls', '-u', multiple=True, help='Multiple repository URLs (can be used multiple times)')
@click.option('--repo-paths', '-p', multiple=True, help='Multiple local repository paths (can be used multiple times)')
@click.option('--repo-zips', '-z', multiple=True, help='Multiple ZIP files containing repositories (can be used multiple times)')
@click.option('--urls-file', '-f', help='File containing repository URLs (one per line)')
@click.option('--config', '-c', help='Path to configuration file')
def index_multiple(repo_urls, repo_paths, repo_zips, urls_file, config):
    """Index multiple repositories at once"""
    
    sources = []
    
    # Add URLs from command line
    for url in repo_urls:
        sources.append({'source': url, 'is_url': True, 'is_zip': False})
    
    # Add paths from command line
    for path in repo_paths:
        sources.append({'source': path, 'is_url': False, 'is_zip': False})
    
    # Add ZIP files from command line
    for zip_path in repo_zips:
        sources.append({'source': zip_path, 'is_url': False, 'is_zip': True})
    
    # Add URLs from file
    if urls_file:
        try:
            with open(urls_file, 'r') as f:
                for line in f:
                    url = line.strip()
                    if url and not url.startswith('#'):
                        sources.append({'source': url, 'is_url': True, 'is_zip': False})
        except Exception as e:
            click.echo(f"Error reading URLs file: {e}", err=True)
            sys.exit(1)
    
    if not sources:
        click.echo("Error: No repositories specified. Use --repo-urls, --repo-paths, --repo-zips, or --urls-file", err=True)
        sys.exit(1)
    
    fastcode = FastCode(config_path=config)
    
    try:
        click.echo(f"Loading and indexing {len(sources)} repositories...")
        fastcode.load_multiple_repositories(sources)
        
        # Show summary
        stats = fastcode.get_repository_stats()
        click.echo("\n" + "=" * 60)
        click.echo("Multi-Repository Indexing Complete!")
        click.echo("=" * 60)
        click.echo(f"Total Repositories: {stats['total_repositories']}")
        click.echo(f"Total Elements: {stats['total_elements']}")
        click.echo("\nRepository Details:")
        for repo in stats['repositories']:
            click.echo(f"  - {repo['name']}: {repo['elements']} elements, {repo['files']} files")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--query', '-q', required=True, help='Question to ask')
@click.option('--repos', '-r', multiple=True, help='Specific repositories to search (can be used multiple times, omit for all)')
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--output', '-o', help='Output file (default: stdout)')
@click.option('--load-cache', is_flag=True, help='Load from multi-repo cache')
def query_multiple(query, repos, config, output, load_cache):
    """Query across multiple indexed repositories"""
    
    fastcode = FastCode(config_path=config)
    
    try:
        # Try to load from cache
        if load_cache:
            click.echo("Loading multi-repository index from cache...")
            if not fastcode._load_multi_repo_cache():
                click.echo("Error: Failed to load multi-repo cache. Please index repositories first.", err=True)
                sys.exit(1)
        else:
            click.echo("Error: No repositories loaded. Use 'index-multiple' first or use --load-cache", err=True)
            sys.exit(1)
        
        # Get list of available repositories
        available_repos = fastcode.vector_store.get_repository_names()
        
        if not available_repos:
            click.echo("Error: No repositories found in index", err=True)
            sys.exit(1)
        
        # Determine which repositories to query
        repo_filter = None
        if repos:
            # Validate repository names
            invalid_repos = [r for r in repos if r not in available_repos]
            if invalid_repos:
                click.echo(f"Warning: Unknown repositories: {', '.join(invalid_repos)}")
            repo_filter = [r for r in repos if r in available_repos]
            click.echo(f"Searching in repositories: {', '.join(repo_filter)}")
        else:
            click.echo(f"Searching in all {len(available_repos)} repositories")
        
        # Query
        click.echo(f"\nProcessing query: {query}\n")
        result = fastcode.query(query, repo_filter=repo_filter)
        
        # Format output
        formatted = fastcode.answer_generator.format_answer_with_sources(result)
        
        # Output result
        if output:
            with open(output, 'w') as f:
                f.write(formatted)
            click.echo(f"Result saved to {output}")
        else:
            click.echo(formatted)
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--load-cache', is_flag=True, help='Load from multi-repo cache (for full metadata)')
def list_repos(config, load_cache):
    """List all indexed repositories"""
    
    fastcode = FastCode(config_path=config)
    
    try:
        # If load-cache is specified, load into memory for full details
        if load_cache:
            if not fastcode._load_multi_repo_cache():
                click.echo("Error: Failed to load multi-repo cache", err=True)
                sys.exit(1)
            
            repositories = fastcode.list_repositories()
            
            if not repositories:
                click.echo("No repositories found in index")
                return
            
            click.echo("=" * 80)
            click.echo("Indexed Repositories (Loaded)")
            click.echo("=" * 80)
            
            for i, repo in enumerate(repositories, 1):
                click.echo(f"\n{i}. {repo['name']}")
                click.echo(f"   Elements: {repo['element_count']}")
                click.echo(f"   Files: {repo['file_count']}")
                click.echo(f"   Size: {repo['size_mb']:.2f} MB")
                if repo['url'] != 'N/A':
                    click.echo(f"   URL: {repo['url']}")
            
            click.echo("\n" + "=" * 80)
            click.echo(f"Total: {len(repositories)} repositories")
        
        # Default: Scan available index files without loading
        else:
            available_repos = fastcode.vector_store.scan_available_indexes()
            
            if not available_repos:
                click.echo("No repository indexes found")
                return
            
            click.echo("=" * 80)
            click.echo("Available Repository Indexes")
            click.echo("=" * 80)
            
            for i, repo in enumerate(available_repos, 1):
                click.echo(f"\n{i}. {repo['name']}")
                click.echo(f"   Elements: {repo['element_count']}")
                click.echo(f"   Files: {repo['file_count']}")
                click.echo(f"   Index Size: {repo['size_mb']:.2f} MB")
                if repo['url'] != 'N/A':
                    click.echo(f"   URL: {repo['url']}")
            
            click.echo("\n" + "=" * 80)
            click.echo(f"Total: {len(available_repos)} repositories")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', help='Path to configuration file')
def repo_stats(config):
    """Show statistics for all indexed repositories"""
    
    fastcode = FastCode(config_path=config)
    
    try:
        # Scan available indexes from disk
        available_repos = fastcode.vector_store.scan_available_indexes()
        
        if not available_repos:
            click.echo("=" * 60)
            click.echo("Repository Statistics")
            click.echo("=" * 60)
            click.echo("Total Repositories: 0")
            click.echo("Total Indexed Elements: 0")
            click.echo("\nPer-Repository Breakdown:")
            return
        
        # Calculate totals
        total_repos = len(available_repos)
        total_elements = sum(repo['element_count'] for repo in available_repos)
        total_size = sum(repo['size_mb'] for repo in available_repos)
        
        click.echo("=" * 60)
        click.echo("Repository Statistics")
        click.echo("=" * 60)
        click.echo(f"Total Repositories: {total_repos}")
        click.echo(f"Total Indexed Elements: {total_elements}")
        click.echo(f"Total Index Size: {total_size:.2f} MB")
        click.echo("\nPer-Repository Breakdown:")
        
        for repo in available_repos:
            click.echo(f"\n  {repo['name']}:")
            click.echo(f"    Elements: {repo['element_count']}")
            click.echo(f"    Files: {repo['file_count']}")
            click.echo(f"    Size: {repo['size_mb']:.2f} MB")
            if repo['url'] != 'N/A':
                click.echo(f"    URL: {repo['url']}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.option('--config', '-c', help='Path to configuration file')
def list_sessions(config):
    """List all dialogue sessions"""
    
    fastcode = FastCode(config_path=config)
    
    try:
        sessions = fastcode.list_sessions()
        
        if not sessions:
            click.echo("No dialogue sessions found")
            return
        
        click.echo("=" * 80)
        click.echo("Dialogue Sessions")
        click.echo("=" * 80)
        
        import time
        from datetime import datetime
        
        for i, session in enumerate(sessions, 1):
            session_id = session.get("session_id", "Unknown")
            total_turns = session.get("total_turns", 0)
            created_at = session.get("created_at", 0)
            last_updated = session.get("last_updated", 0)
            
            created_str = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d %H:%M:%S")
            updated_str = datetime.fromtimestamp(last_updated).strftime("%Y-%m-%d %H:%M:%S")
            
            click.echo(f"\n{i}. Session ID: {session_id}")
            click.echo(f"   Total Turns: {total_turns}")
            click.echo(f"   Created: {created_str}")
            click.echo(f"   Last Updated: {updated_str}")
        
        click.echo("\n" + "=" * 80)
        click.echo(f"Total Sessions: {len(sessions)}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('session_id')
@click.option('--config', '-c', help='Path to configuration file')
def show_session(session_id, config):
    """Show dialogue history for a session"""
    
    fastcode = FastCode(config_path=config)
    
    try:
        history = fastcode.get_session_history(session_id)
        
        if not history:
            click.echo(f"No history found for session: {session_id}")
            return
        
        click.echo("=" * 80)
        click.echo(f"Session: {session_id}")
        click.echo("=" * 80)
        
        for turn in history:
            turn_num = turn.get("turn_number", 0)
            query = turn.get("query", "")
            answer = turn.get("answer", "")
            summary = turn.get("summary", "")
            
            import time
            from datetime import datetime
            timestamp = turn.get("timestamp", 0)
            time_str = datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")
            
            click.echo(f"\n{'='*80}")
            click.echo(f"Turn {turn_num} ({time_str})")
            click.echo(f"{'='*80}")
            click.echo(f"\nQuestion: {query}")
            click.echo(f"\nAnswer:\n{answer}")
            
            if summary:
                click.echo(f"\n[Summary]:\n{summary}")
        
        click.echo("\n" + "=" * 80)
        click.echo(f"Total Turns: {len(history)}")
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


@cli.command()
@click.argument('session_id')
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--confirm', is_flag=True, help='Skip confirmation prompt')
def delete_session(session_id, config, confirm):
    """Delete a dialogue session"""
    
    fastcode = FastCode(config_path=config)
    
    try:
        # Check if session exists
        history = fastcode.get_session_history(session_id)
        if not history:
            click.echo(f"Session not found: {session_id}", err=True)
            sys.exit(1)
        
        # Confirmation
        if not confirm:
            click.echo(f"Session ID: {session_id}")
            click.echo(f"Total turns: {len(history)}")
            
            if not click.confirm("\nAre you sure you want to delete this session?"):
                click.echo("Cancelled.")
                return
        
        # Delete session
        if fastcode.delete_session(session_id):
            click.echo(f"✓ Successfully deleted session: {session_id}")
        else:
            click.echo(f"Failed to delete session: {session_id}", err=True)
            sys.exit(1)
        
    except Exception as e:
        click.echo(f"Error: {str(e)}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    cli()

