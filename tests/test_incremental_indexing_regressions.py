import ast
import copy
import os
import pickle
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
FASTCODE_MAIN = ROOT / "fastcode" / "main.py"
MCP_SERVER = ROOT / "mcp_server.py"


def _null_logger():
    return SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: None,
        debug=lambda *args, **kwargs: None,
    )


def _load_functions(path, names, *, class_name=None, global_ns=None):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    if class_name:
        class_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        lookup = {
            node.name: node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
    else:
        lookup = {
            node.name: node
            for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }

    selected = []
    for name in names:
        node = copy.deepcopy(lookup[name])
        node.decorator_list = []
        selected.append(node)

    future_import = ast.parse("from __future__ import annotations").body
    module = ast.Module(body=future_import + selected, type_ignores=[])
    ast.fix_missing_locations(module)

    namespace = {}
    if global_ns:
        namespace.update(global_ns)
    exec(compile(module, str(path), "exec"), namespace)
    return [namespace[name] for name in names]


class StubCodeElement:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def to_dict(self):
        return dict(self.__dict__)


def _element_meta(element_id, file_path, relative_path, embedding):
    return {
        "id": element_id,
        "type": "file",
        "name": relative_path,
        "file_path": file_path,
        "relative_path": relative_path,
        "language": "python",
        "start_line": 1,
        "end_line": 10,
        "code": "print('hello')",
        "signature": None,
        "docstring": None,
        "summary": None,
        "metadata": {"embedding": embedding},
        "repo_name": "repo",
        "repo_url": None,
    }


def _make_incremental_reindex(globals_override=None):
    base_globals = {
        "os": os,
        "np": np,
        "CodeElement": StubCodeElement,
    }
    if globals_override:
        base_globals.update(globals_override)
    return _load_functions(
        FASTCODE_MAIN,
        ["incremental_reindex"],
        class_name="FastCode",
        global_ns=base_globals,
    )[0]


def _get_function_node(path, name, *, class_name=None):
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))

    if class_name:
        class_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == class_name
        )
        return next(
            node
            for node in class_node.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
        )

    return next(
        node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name
    )


def _assert_no_print_calls(path, name, *, class_name=None):
    node = _get_function_node(path, name, class_name=class_name)
    print_calls = [
        child
        for child in ast.walk(node)
        if isinstance(child, ast.Call)
        and isinstance(child.func, ast.Name)
        and child.func.id == "print"
    ]
    assert not print_calls, f"{path}:{name} writes to stdout via print()"


def test_incremental_reindex_uses_loader_repo_path_when_rebuilding_graphs(tmp_path):
    captured = {}

    class FakeTempStore:
        def __init__(self, config):
            self.config = config

        def initialize(self, dimension):
            self.dimension = dimension

        def add_vectors(self, vectors, metadata_list):
            self.vectors = vectors
            self.metadata_list = metadata_list

    class FakeRetriever:
        def __init__(self, config, vector_store, embedder, graph_builder, repo_root=None):
            self.repo_root = repo_root

        def index_for_bm25(self, elements):
            self.elements = elements

    class FakeGraphBuilder:
        def __init__(self, config):
            self.config = config

        def build_graphs(self, elements, module_resolver, symbol_resolver):
            self.elements = elements

    class FakeGlobalIndexBuilder:
        def __init__(self, config):
            self.config = config

        def build_maps(self, elements, repo_root):
            captured["repo_root"] = repo_root

    incremental_reindex = _make_incremental_reindex(
        {
            "VectorStore": FakeTempStore,
            "HybridRetriever": FakeRetriever,
            "CodeGraphBuilder": FakeGraphBuilder,
            "GlobalIndexBuilder": FakeGlobalIndexBuilder,
            "ModuleResolver": lambda gib: ("module_resolver", gib),
            "SymbolResolver": lambda gib, module_resolver: (
                "symbol_resolver",
                gib,
                module_resolver,
            ),
        }
    )

    original_repo = tmp_path / "source-repo"
    copied_repo = tmp_path / "workspace-copy" / "repo"
    original_repo.mkdir()
    copied_repo.mkdir(parents=True)

    file_path = copied_repo / "a.py"
    meta = _element_meta("elem-1", str(file_path), "a.py", [0.1, 0.2, 0.3])

    class FakeLoader:
        def __init__(self):
            self.repo_path = None

        def load_from_path(self, path):
            self.repo_path = str(copied_repo)

        def scan_files(self):
            return []

    fc = SimpleNamespace(
        logger=_null_logger(),
        loader=FakeLoader(),
        config={},
        embedder=SimpleNamespace(embedding_dim=3),
        loaded_repositories={},
        indexer=SimpleNamespace(index_files=lambda file_infos, repo_name, repo_url=None: []),
        _load_file_manifest=lambda repo_name: {"files": {"a.py": {"element_ids": ["elem-1"]}}},
        _detect_file_changes=lambda repo_name, current_files: {
            "added": [],
            "modified": [],
            "deleted": ["deleted.py"],
            "unchanged": ["a.py"],
            "manifest": {"files": {"a.py": {"element_ids": ["elem-1"]}}},
            "current_lookup": {},
        },
        _load_existing_metadata=lambda repo_name: [meta],
        _collect_unchanged_elements=lambda manifest, unchanged_files, existing_metadata: (
            existing_metadata,
            ["elem-1"],
        ),
        _should_persist_indexes=lambda: False,
    )

    incremental_reindex(fc, "repo", repo_path=str(original_repo))

    assert captured["repo_root"] == str(copied_repo)


def test_incremental_reindex_regenerates_repository_overview_after_changes(tmp_path):
    class FakeTempStore:
        def __init__(self, config):
            self.config = config

        def initialize(self, dimension):
            self.dimension = dimension

        def add_vectors(self, vectors, metadata_list):
            self.vectors = vectors
            self.metadata_list = metadata_list

        def save(self, repo_name):
            self.saved_repo = repo_name

    class FakeRetriever:
        def __init__(self, config, vector_store, embedder, graph_builder, repo_root=None):
            self.repo_root = repo_root

        def index_for_bm25(self, elements):
            self.elements = elements

        def save_bm25(self, repo_name):
            self.saved_repo = repo_name

    class FakeGraphBuilder:
        def __init__(self, config):
            self.config = config

        def build_graphs(self, elements, module_resolver, symbol_resolver):
            self.elements = elements

        def save(self, repo_name):
            self.saved_repo = repo_name

    class FakeGlobalIndexBuilder:
        def __init__(self, config):
            self.config = config

        def build_maps(self, elements, repo_root):
            self.elements = elements
            self.repo_root = repo_root

    incremental_reindex = _make_incremental_reindex(
        {
            "VectorStore": FakeTempStore,
            "HybridRetriever": FakeRetriever,
            "CodeGraphBuilder": FakeGraphBuilder,
            "GlobalIndexBuilder": FakeGlobalIndexBuilder,
            "ModuleResolver": lambda gib: ("module_resolver", gib),
            "SymbolResolver": lambda gib, module_resolver: (
                "symbol_resolver",
                gib,
                module_resolver,
            ),
        }
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()
    changed_file = repo_root / "a.py"

    current_file_info = {
        "path": str(changed_file),
        "relative_path": "a.py",
        "size": 10,
        "extension": ".py",
    }
    existing_meta = _element_meta(
        "old-elem",
        str(repo_root / "old.py"),
        "old.py",
        [0.1, 0.2, 0.3],
    )
    new_element = StubCodeElement(
        id="new-elem",
        type="file",
        name="a.py",
        file_path=str(changed_file),
        relative_path="a.py",
        language="python",
        start_line=1,
        end_line=20,
        code="print('updated')",
        signature=None,
        docstring=None,
        summary=None,
        metadata={"embedding": [0.4, 0.5, 0.6]},
        repo_name="repo",
        repo_url="https://example.com/repo.git",
    )

    class FakeLoader:
        def __init__(self):
            self.repo_path = None

        def load_from_path(self, path):
            self.repo_path = str(repo_root)

        def scan_files(self):
            return [current_file_info]

    vector_store = SimpleNamespace(
        persist_dir=str(tmp_path / "persist"),
        save_repo_overview=Mock(),
    )

    class FakeIndexer:
        def __init__(self):
            self.overview_generator = SimpleNamespace(
                parse_file_structure=lambda repo_path, files: {"languages": {"python": len(files)}},
                generate_overview=lambda repo_path, repo_name, file_structure: {
                    "repo_name": repo_name,
                    "summary": "updated summary",
                    "structure_text": "a.py",
                    "file_structure": file_structure,
                    "readme_content": "",
                    "has_readme": False,
                },
            )

        def index_files(self, file_infos, repo_name, repo_url=None):
            return [new_element]

        def _save_repository_overview(self, overview):
            vector_store.save_repo_overview(
                overview["repo_name"],
                overview["summary"],
                np.array([0.1, 0.2, 0.3], dtype=np.float32),
                {
                    "summary": overview["summary"],
                    "structure_text": overview["structure_text"],
                    "file_structure": overview["file_structure"],
                },
            )

    fc = SimpleNamespace(
        logger=_null_logger(),
        loader=FakeLoader(),
        vector_store=vector_store,
        indexer=FakeIndexer(),
        config={},
        embedder=SimpleNamespace(embedding_dim=3),
        loaded_repositories={"repo": {"url": "https://example.com/repo.git"}},
        _load_file_manifest=lambda repo_name: {"files": {"old.py": {"element_ids": ["old-elem"]}}},
        _detect_file_changes=lambda repo_name, current_files: {
            "added": ["a.py"],
            "modified": [],
            "deleted": [],
            "unchanged": [],
            "manifest": {"files": {"old.py": {"element_ids": ["old-elem"]}}},
            "current_lookup": {"a.py": {"file_info": current_file_info}},
        },
        _load_existing_metadata=lambda repo_name: [existing_meta],
        _collect_unchanged_elements=lambda manifest, unchanged_files, existing_metadata: ([], []),
        _should_persist_indexes=lambda: True,
        _build_file_manifest=lambda elements, repo_root: {"files": {}},
        _save_file_manifest=lambda repo_name, manifest: None,
    )

    incremental_reindex(fc, "repo", repo_path=str(repo_root))

    assert vector_store.save_repo_overview.called


def test_incremental_reindex_rejects_incompatible_preserved_embeddings(tmp_path):
    class GuardedTempStore:
        def __init__(self, config):
            self.config = config
            self.dimension = None

        def initialize(self, dimension):
            self.dimension = dimension

        def add_vectors(self, vectors, metadata_list):
            if vectors.shape[1] != self.dimension:
                raise AssertionError(
                    "incremental_reindex attempted to rebuild with incompatible "
                    "preserved embeddings"
                )

    class FakeRetriever:
        def __init__(self, config, vector_store, embedder, graph_builder, repo_root=None):
            self.repo_root = repo_root

        def index_for_bm25(self, elements):
            self.elements = elements

    class FakeGraphBuilder:
        def __init__(self, config):
            self.config = config

        def build_graphs(self, elements, module_resolver, symbol_resolver):
            self.elements = elements

    class FakeGlobalIndexBuilder:
        def __init__(self, config):
            self.config = config

        def build_maps(self, elements, repo_root):
            self.elements = elements

    incremental_reindex = _make_incremental_reindex(
        {
            "VectorStore": GuardedTempStore,
            "HybridRetriever": FakeRetriever,
            "CodeGraphBuilder": FakeGraphBuilder,
            "GlobalIndexBuilder": FakeGlobalIndexBuilder,
            "ModuleResolver": lambda gib: ("module_resolver", gib),
            "SymbolResolver": lambda gib, module_resolver: (
                "symbol_resolver",
                gib,
                module_resolver,
            ),
        }
    )

    repo_root = tmp_path / "repo"
    repo_root.mkdir()

    existing_meta = _element_meta(
        "elem-1",
        str(repo_root / "a.py"),
        "a.py",
        [0.1, 0.2],
    )

    class FakeLoader:
        def __init__(self):
            self.repo_path = None

        def load_from_path(self, path):
            self.repo_path = str(repo_root)

        def scan_files(self):
            return []

    fc = SimpleNamespace(
        logger=_null_logger(),
        loader=FakeLoader(),
        config={},
        embedder=SimpleNamespace(embedding_dim=3),
        loaded_repositories={},
        indexer=SimpleNamespace(index_files=lambda file_infos, repo_name, repo_url=None: []),
        _load_file_manifest=lambda repo_name: {"files": {"a.py": {"element_ids": ["elem-1"]}}},
        _detect_file_changes=lambda repo_name, current_files: {
            "added": [],
            "modified": [],
            "deleted": ["deleted.py"],
            "unchanged": ["a.py"],
            "manifest": {"files": {"a.py": {"element_ids": ["elem-1"]}}},
            "current_lookup": {},
        },
        _load_existing_metadata=lambda repo_name: [existing_meta],
        _collect_unchanged_elements=lambda manifest, unchanged_files, existing_metadata: (
            existing_metadata,
            ["elem-1"],
        ),
        _should_persist_indexes=lambda: False,
    )

    incremental_reindex(fc, "repo", repo_path=str(repo_root))


def test_ensure_repos_ready_falls_back_to_full_reindex_when_manifest_is_missing(tmp_path):
    ensure_repos_ready = _load_functions(
        MCP_SERVER,
        ["_ensure_repos_ready"],
        global_ns={"os": os, "logger": _null_logger()},
    )[0]

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    fc = SimpleNamespace(
        _infer_is_url=lambda source: False,
        incremental_reindex=Mock(return_value={"status": "no_manifest", "changes": 0}),
        load_repository=Mock(),
        index_repository=Mock(),
        repo_indexed=True,
        loaded_repositories={},
    )
    full_reindex = Mock(return_value={"status": "success", "count": 12})

    ensure_repos_ready.__globals__.update(
        {
            "_get_fastcode": lambda: fc,
            "_apply_forced_env_excludes": lambda fc: None,
            "_repo_name_from_source": lambda source, is_url: "repo",
            "_is_repo_indexed": lambda repo_name: True,
            "_run_full_reindex": full_reindex,
            "_invalidate_loaded_state": lambda fc: None,
        }
    )

    ensure_repos_ready([str(repo_dir)])

    assert full_reindex.called
    assert not fc.load_repository.called
    assert not fc.index_repository.called


def test_ensure_repos_ready_falls_back_to_full_reindex_on_embedding_mismatch(tmp_path):
    ensure_repos_ready = _load_functions(
        MCP_SERVER,
        ["_ensure_repos_ready"],
        global_ns={"os": os, "logger": _null_logger()},
    )[0]

    repo_dir = tmp_path / "repo"
    repo_dir.mkdir()

    fc = SimpleNamespace(
        _infer_is_url=lambda source: False,
        incremental_reindex=Mock(
            return_value={"status": "embedding_dimension_mismatch", "changes": 3}
        ),
        load_repository=Mock(),
        index_repository=Mock(),
        repo_indexed=True,
        loaded_repositories={},
    )
    full_reindex = Mock(return_value={"status": "success", "count": 9})

    ensure_repos_ready.__globals__.update(
        {
            "_get_fastcode": lambda: fc,
            "_apply_forced_env_excludes": lambda fc: None,
            "_repo_name_from_source": lambda source, is_url: "repo",
            "_is_repo_indexed": lambda repo_name: True,
            "_run_full_reindex": full_reindex,
            "_invalidate_loaded_state": lambda fc: None,
        }
    )

    ensure_repos_ready([str(repo_dir)])

    assert full_reindex.called


def test_lookup_tools_do_not_disable_incremental_refresh():
    search_symbol, get_file_summary, get_call_chain = _load_functions(
        MCP_SERVER,
        ["search_symbol", "get_file_summary", "get_call_chain"],
        global_ns={},
    )

    calls = []

    def fake_ensure_repos_ready(repos, allow_incremental=True, ctx=None):
        calls.append(allow_incremental)
        return ["repo"]

    fake_fc = SimpleNamespace(
        vector_store=SimpleNamespace(metadata=[]),
        graph_builder=SimpleNamespace(
            element_by_name={},
            element_by_id={},
            get_callers=lambda element_id: [],
            get_callees=lambda element_id: [],
        ),
    )

    shared_globals = {
        "_get_fastcode": lambda: fake_fc,
        "_ensure_repos_ready": fake_ensure_repos_ready,
        "_ensure_loaded": lambda fc, ready_names: True,
    }

    search_symbol.__globals__.update(shared_globals)
    get_file_summary.__globals__.update(shared_globals)
    get_call_chain.__globals__.update(shared_globals)

    search_symbol("FastCode", ["/tmp/repo"])
    get_file_summary("fastcode/main.py", ["/tmp/repo"])
    get_call_chain("query", ["/tmp/repo"])

    assert calls == [True, True, True]


def test_reindex_repo_uses_clean_full_reindex_helper():
    reindex_repo = _load_functions(
        MCP_SERVER,
        ["reindex_repo"],
        global_ns={"os": os, "logger": _null_logger()},
    )[0]

    fc = SimpleNamespace(_infer_is_url=lambda source: False)
    full_reindex = Mock(return_value={"status": "success", "count": 42})

    reindex_repo.__globals__.update(
        {
            "_get_fastcode": lambda: fc,
            "_repo_name_from_source": lambda source, is_url: "repo",
            "_run_full_reindex": full_reindex,
        }
    )

    message = reindex_repo("/tmp/repo")

    assert "42 elements indexed" in message
    assert full_reindex.called


def test_load_multi_repo_cache_replaces_loaded_repository_set(tmp_path):
    load_multi_repo_cache = _load_functions(
        FASTCODE_MAIN,
        ["_load_multi_repo_cache"],
        class_name="FastCode",
        global_ns={
            "os": os,
            "pickle": __import__("pickle"),
            "CodeGraphBuilder": lambda config: SimpleNamespace(
                load=lambda repo_name: False,
                merge_from_file=lambda repo_name: False,
                build_graphs=lambda elements: None,
            ),
        },
    )[0]

    persist_dir = tmp_path / "persist"
    persist_dir.mkdir()
    for repo_name in ("repo_a", "repo_b"):
        (persist_dir / f"{repo_name}.faiss").write_bytes(b"index")
        (persist_dir / f"{repo_name}_metadata.pkl").write_bytes(b"meta")

    class FakeVectorStore:
        def __init__(self):
            self.persist_dir = str(persist_dir)
            self.merged = []

        def initialize(self, dimension):
            self.dimension = dimension

        def merge_from_index(self, repo_name):
            self.merged.append(repo_name)
            return True

        def get_count(self):
            return len(self.merged)

    fake_fc = SimpleNamespace(
        logger=_null_logger(),
        config={},
        vector_store=FakeVectorStore(),
        embedder=SimpleNamespace(embedding_dim=3),
        loaded_repositories={"repo_a": {"name": "repo_a"}, "repo_b": {"name": "repo_b"}},
        retriever=SimpleNamespace(
            persist_dir=str(persist_dir),
            build_repo_overview_bm25=lambda: None,
            index_for_bm25=lambda elements: None,
            full_bm25_elements=[],
            full_bm25_corpus=[],
            full_bm25=None,
        ),
        graph_builder=SimpleNamespace(load=lambda repo_name: False, merge_from_file=lambda repo_name: False),
        _reconstruct_elements_from_metadata=lambda: [],
    )

    ok = load_multi_repo_cache(fake_fc, repo_names=["repo_a"])

    assert ok is True
    assert set(fake_fc.loaded_repositories) == {"repo_a"}


def test_load_multi_repo_cache_fails_when_any_requested_repo_cannot_be_merged(tmp_path):
    load_multi_repo_cache = _load_functions(
        FASTCODE_MAIN,
        ["_load_multi_repo_cache"],
        class_name="FastCode",
        global_ns={"os": os, "pickle": pickle},
    )[0]

    persist_dir = tmp_path / "persist"
    persist_dir.mkdir()
    for repo_name in ("repo_a", "repo_b"):
        (persist_dir / f"{repo_name}.faiss").write_bytes(b"index")
        (persist_dir / f"{repo_name}_metadata.pkl").write_bytes(b"meta")

    class FakeVectorStore:
        def __init__(self):
            self.persist_dir = str(persist_dir)
            self.merged = []

        def initialize(self, dimension):
            self.dimension = dimension

        def merge_from_index(self, repo_name):
            if repo_name == "repo_a":
                self.merged.append(repo_name)
                return True
            return False

        def get_count(self):
            return len(self.merged)

    fake_fc = SimpleNamespace(
        logger=_null_logger(),
        vector_store=FakeVectorStore(),
        embedder=SimpleNamespace(embedding_dim=3),
        loaded_repositories={"repo_a": {"name": "repo_a"}, "repo_b": {"name": "repo_b"}},
        retriever=SimpleNamespace(
            persist_dir=str(persist_dir),
            build_repo_overview_bm25=lambda: None,
        ),
        graph_builder=SimpleNamespace(load=lambda repo_name: False, merge_from_file=lambda repo_name: False),
        _reconstruct_elements_from_metadata=lambda: [],
    )

    ok = load_multi_repo_cache(fake_fc, repo_names=["repo_a", "repo_b"])

    assert ok is False


def test_partial_multi_repo_load_does_not_keep_failed_repo_marked_as_loaded(tmp_path):
    load_multi_repo_cache = _load_functions(
        FASTCODE_MAIN,
        ["_load_multi_repo_cache"],
        class_name="FastCode",
        global_ns={"os": os, "pickle": pickle},
    )[0]

    persist_dir = tmp_path / "persist"
    persist_dir.mkdir()
    for repo_name in ("repo_a", "repo_b"):
        (persist_dir / f"{repo_name}.faiss").write_bytes(b"index")
        (persist_dir / f"{repo_name}_metadata.pkl").write_bytes(b"meta")

    class FakeVectorStore:
        def __init__(self):
            self.persist_dir = str(persist_dir)
            self.merged = []

        def initialize(self, dimension):
            self.dimension = dimension

        def merge_from_index(self, repo_name):
            if repo_name == "repo_a":
                self.merged.append(repo_name)
                return True
            return False

        def get_count(self):
            return len(self.merged)

    fake_fc = SimpleNamespace(
        logger=_null_logger(),
        vector_store=FakeVectorStore(),
        embedder=SimpleNamespace(embedding_dim=3),
        loaded_repositories={"repo_a": {"name": "repo_a"}, "repo_b": {"name": "repo_b"}},
        retriever=SimpleNamespace(
            persist_dir=str(persist_dir),
            build_repo_overview_bm25=lambda: None,
        ),
        graph_builder=SimpleNamespace(load=lambda repo_name: False, merge_from_file=lambda repo_name: False),
        _reconstruct_elements_from_metadata=lambda: [],
    )

    ok = load_multi_repo_cache(fake_fc, repo_names=["repo_a", "repo_b"])

    assert ok is False
    assert "repo_b" not in fake_fc.loaded_repositories


def test_ensure_loaded_rejects_partial_multi_repo_load():
    ensure_loaded = _load_functions(
        MCP_SERVER,
        ["_ensure_loaded"],
        global_ns={"logger": _null_logger()},
    )[0]

    class FakeFC:
        def __init__(self):
            self.repo_indexed = False
            self.loaded_repositories = {}

        def _load_multi_repo_cache(self, repo_names=None):
            self.loaded_repositories = {"repo_a": {"name": "repo_a"}}
            return True

    fc = FakeFC()

    ok = ensure_loaded(fc, ["repo_a", "repo_b"])

    assert ok is False


def test_repo_name_from_source_disambiguates_local_paths_with_same_basename(monkeypatch):
    repo_name_from_source = _load_functions(
        MCP_SERVER,
        ["_repo_name_from_source"],
        global_ns={"os": os},
    )[0]

    fake_fastcode = types.ModuleType("fastcode")
    fake_fastcode.__path__ = []
    fake_utils = types.ModuleType("fastcode.utils")
    fake_utils.get_repo_name_from_url = lambda source: "repo-from-url"
    fake_utils.get_repo_name_from_path = (
        lambda source, workspace_root=None: f"derived::{source}"
    )
    fake_fastcode.utils = fake_utils

    monkeypatch.setitem(sys.modules, "fastcode", fake_fastcode)
    monkeypatch.setitem(sys.modules, "fastcode.utils", fake_utils)
    repo_name_from_source.__globals__["_get_fastcode"] = lambda: SimpleNamespace(
        loader=SimpleNamespace(safe_repo_root="/repos")
    )

    repo_a = repo_name_from_source("/tmp/team-a/service", False)
    repo_b = repo_name_from_source("/var/team-b/service", False)

    assert repo_a != repo_b


def test_repo_name_from_source_is_stable_for_workspace_copy(monkeypatch):
    repo_name_from_source = _load_functions(
        MCP_SERVER,
        ["_repo_name_from_source"],
        global_ns={"os": os},
    )[0]
    get_repo_name_from_path = _load_functions(
        ROOT / "fastcode" / "utils.py",
        ["get_repo_name_from_path"],
        global_ns={
            "os": os,
            "hashlib": __import__("hashlib"),
            "normalize_path": lambda path: os.path.normpath(path).replace("\\", "/"),
        },
    )[0]

    fake_fastcode = types.ModuleType("fastcode")
    fake_fastcode.__path__ = []
    fake_utils = types.ModuleType("fastcode.utils")
    fake_utils.get_repo_name_from_url = lambda source: "repo-from-url"
    fake_utils.get_repo_name_from_path = get_repo_name_from_path
    fake_fastcode.utils = fake_utils

    monkeypatch.setitem(sys.modules, "fastcode", fake_fastcode)
    monkeypatch.setitem(sys.modules, "fastcode.utils", fake_utils)
    repo_name_from_source.__globals__["_get_fastcode"] = lambda: SimpleNamespace(
        loader=SimpleNamespace(safe_repo_root="/repos")
    )

    original_path = "/tmp/team/service"
    workspace_path = f"/repos/{get_repo_name_from_path(original_path)}"

    assert repo_name_from_source(workspace_path, False) == repo_name_from_source(
        original_path, False
    )


def test_get_repo_name_from_path_is_stable_for_workspace_paths():
    get_repo_name_from_path = _load_functions(
        ROOT / "fastcode" / "utils.py",
        ["get_repo_name_from_path"],
        global_ns={
            "os": os,
            "hashlib": __import__("hashlib"),
            "normalize_path": lambda path: os.path.normpath(path).replace("\\", "/"),
        },
    )[0]

    original_path = "/tmp/team/service"
    derived_name = get_repo_name_from_path(original_path)
    workspace_path = f"/repos/{derived_name}"

    assert get_repo_name_from_path(workspace_path, workspace_root="/repos") == derived_name


def test_get_repo_name_from_path_keeps_disambiguation_for_hex_suffixed_names():
    get_repo_name_from_path = _load_functions(
        ROOT / "fastcode" / "utils.py",
        ["get_repo_name_from_path"],
        global_ns={
            "os": os,
            "hashlib": __import__("hashlib"),
            "normalize_path": lambda path: os.path.normpath(path).replace("\\", "/"),
        },
    )[0]

    repo_a = get_repo_name_from_path("/tmp/team-a/service-deadbeef", workspace_root="/repos")
    repo_b = get_repo_name_from_path("/var/team-b/service-deadbeef", workspace_root="/repos")

    assert repo_a != repo_b


def test_load_from_path_reuses_existing_workspace_copy(tmp_path):
    get_repo_name_from_path = _load_functions(
        ROOT / "fastcode" / "utils.py",
        ["get_repo_name_from_path"],
        global_ns={
            "os": os,
            "hashlib": __import__("hashlib"),
            "normalize_path": lambda path: os.path.normpath(path).replace("\\", "/"),
        },
    )[0]
    load_from_path = _load_functions(
        ROOT / "fastcode" / "loader.py",
        ["load_from_path"],
        class_name="RepositoryLoader",
        global_ns={
            "os": os,
            "shutil": SimpleNamespace(copytree=Mock()),
            "get_repo_name_from_path": get_repo_name_from_path,
        },
    )[0]

    safe_repo_root = tmp_path / "repos"
    source_repo = tmp_path / "source" / "service"
    source_repo.mkdir(parents=True)

    derived_name = get_repo_name_from_path(str(source_repo))
    workspace_repo = safe_repo_root / derived_name
    workspace_repo.mkdir(parents=True)

    copytree = load_from_path.__globals__["shutil"].copytree
    prepare_repo_path = Mock(return_value=str(tmp_path / "unexpected-copy"))
    loader = SimpleNamespace(
        logger=_null_logger(),
        safe_repo_root=str(safe_repo_root),
        _prepare_repo_path=prepare_repo_path,
        repo_name=None,
        repo_path=None,
    )

    repo_path = load_from_path(loader, str(workspace_repo))

    assert repo_path == str(workspace_repo)
    assert loader.repo_path == str(workspace_repo)
    assert not prepare_repo_path.called
    assert not copytree.called


def test_load_multi_repo_cache_rebuilds_graph_when_graph_cache_is_missing(tmp_path):
    load_multi_repo_cache = _load_functions(
        FASTCODE_MAIN,
        ["_load_multi_repo_cache"],
        class_name="FastCode",
        global_ns={
            "os": os,
            "pickle": pickle,
            "BM25Okapi": lambda corpus: ("bm25", len(corpus)),
            "CodeElement": StubCodeElement,
            "CodeGraphBuilder": lambda config: FakeGraphBuilder(),
        },
    )[0]

    persist_dir = tmp_path / "persist"
    persist_dir.mkdir()
    (persist_dir / "repo.faiss").write_bytes(b"index")
    (persist_dir / "repo_metadata.pkl").write_bytes(b"meta")
    with open(persist_dir / "repo_bm25.pkl", "wb") as f:
        pickle.dump(
            {
                "bm25_corpus": [["query"]],
                "bm25_elements": [
                    {
                        "id": "file_repo_main",
                        "type": "file",
                        "name": "main.py",
                        "file_path": "main.py",
                        "relative_path": "main.py",
                        "language": "python",
                        "start_line": 1,
                        "end_line": 10,
                        "code": "print('hello')",
                        "signature": None,
                        "docstring": None,
                        "summary": None,
                        "metadata": {},
                        "repo_name": "repo",
                        "repo_url": None,
                    }
                ],
            },
            f,
        )

    class FakeVectorStore:
        def __init__(self):
            self.persist_dir = str(persist_dir)
            self.merged = []

        def initialize(self, dimension):
            self.dimension = dimension

        def merge_from_index(self, repo_name):
            self.merged.append(repo_name)
            return True

        def get_count(self):
            return len(self.merged)

    class FakeGraphBuilder:
        def __init__(self):
            self.rebuilt = False

        def load(self, repo_name):
            return False

        def merge_from_file(self, repo_name):
            return False

        def build_graphs(self, elements):
            self.rebuilt = True

    graph_builder = FakeGraphBuilder()
    load_multi_repo_cache.__globals__["CodeGraphBuilder"] = lambda config: graph_builder
    fake_fc = SimpleNamespace(
        logger=_null_logger(),
        config={},
        vector_store=FakeVectorStore(),
        embedder=SimpleNamespace(embedding_dim=3),
        loaded_repositories={"repo": {"name": "repo"}},
        retriever=SimpleNamespace(
            persist_dir=str(persist_dir),
            build_repo_overview_bm25=lambda: None,
            full_bm25_elements=[],
            full_bm25_corpus=[],
            full_bm25=None,
        ),
        graph_builder=graph_builder,
        _reconstruct_elements_from_metadata=lambda: [],
    )

    ok = load_multi_repo_cache(fake_fc, repo_names=["repo"])

    assert ok is True
    assert graph_builder.rebuilt is True


def test_failed_partial_reload_cannot_leave_ensure_loaded_with_stale_memory(tmp_path):
    load_multi_repo_cache = _load_functions(
        FASTCODE_MAIN,
        ["_load_multi_repo_cache"],
        class_name="FastCode",
        global_ns={"os": os, "pickle": pickle},
    )[0]
    ensure_loaded = _load_functions(
        MCP_SERVER,
        ["_ensure_loaded"],
        global_ns={"logger": _null_logger()},
    )[0]

    persist_dir = tmp_path / "persist"
    persist_dir.mkdir()
    for repo_name in ("repo_a", "repo_b"):
        (persist_dir / f"{repo_name}.faiss").write_bytes(b"index")
        (persist_dir / f"{repo_name}_metadata.pkl").write_bytes(b"meta")

    class FakeVectorStore:
        def __init__(self):
            self.persist_dir = str(persist_dir)
            self.merged = ["repo_a", "repo_b"]

        def initialize(self, dimension):
            self.dimension = dimension
            self.merged = []

        def merge_from_index(self, repo_name):
            if repo_name == "repo_a":
                self.merged.append(repo_name)
                return True
            return False

        def get_count(self):
            return len(self.merged)

    fake_fc = SimpleNamespace(
        logger=_null_logger(),
        config={},
        vector_store=FakeVectorStore(),
        embedder=SimpleNamespace(embedding_dim=3),
        retriever=SimpleNamespace(
            persist_dir=str(persist_dir),
            build_repo_overview_bm25=lambda: None,
        ),
        graph_builder=SimpleNamespace(load=lambda repo_name: False, merge_from_file=lambda repo_name: False),
        loaded_repositories={"repo_a": {"name": "repo_a"}, "repo_b": {"name": "repo_b"}},
        repo_indexed=True,
        _reconstruct_elements_from_metadata=lambda: [],
    )

    reload_calls = {"count": 0}

    def wrapped_load_multi_repo_cache(repo_names=None):
        reload_calls["count"] += 1
        return load_multi_repo_cache(fake_fc, repo_names=repo_names)

    fake_fc._load_multi_repo_cache = wrapped_load_multi_repo_cache

    assert fake_fc._load_multi_repo_cache(repo_names=["repo_a", "repo_b"]) is False
    assert ensure_loaded(fake_fc, ["repo_a", "repo_b"]) is False
    assert reload_calls["count"] == 2


def test_ensure_repos_ready_rejects_missing_local_path_even_if_index_exists():
    ensure_repos_ready = _load_functions(
        MCP_SERVER,
        ["_ensure_repos_ready"],
        global_ns={"os": os, "logger": _null_logger()},
    )[0]

    fc = SimpleNamespace(
        _infer_is_url=lambda source: False,
        incremental_reindex=Mock(),
        load_repository=Mock(),
        index_repository=Mock(),
        repo_indexed=True,
        loaded_repositories={},
    )

    ensure_repos_ready.__globals__.update(
        {
            "_get_fastcode": lambda: fc,
            "_apply_forced_env_excludes": lambda fc: None,
            "_repo_name_from_source": lambda source, is_url: "repo",
            "_is_repo_indexed": lambda repo_name: True,
            "_run_full_reindex": Mock(return_value={"status": "success", "count": 3}),
            "_invalidate_loaded_state": lambda fc: None,
        }
    )

    ready = ensure_repos_ready(["/tmp/path-that-does-not-exist"])

    assert ready == []


def test_sequential_index_repository_calls_do_not_save_mixed_repo_indexes():
    class FakeGlobalIndexBuilder:
        def __init__(self, config):
            self.config = config
            self.file_map = {}
            self.module_map = {}

        def build_maps(self, elements, repo_root):
            self.elements = elements
            self.repo_root = repo_root

    index_repository = _load_functions(
        FASTCODE_MAIN,
        ["index_repository"],
        class_name="FastCode",
        global_ns={
            "np": np,
            "CodeGraphBuilder": lambda config: SimpleNamespace(
                build_graphs=lambda elements, module_resolver, symbol_resolver: None,
                save=lambda repo_name: None,
            ),
            "GlobalIndexBuilder": FakeGlobalIndexBuilder,
            "ModuleResolver": lambda gib: ("module_resolver", gib),
            "SymbolResolver": lambda gib, module_resolver: (
                "symbol_resolver",
                gib,
                module_resolver,
            ),
        },
    )[0]

    saved_indexes = {}

    def make_element(repo_name: str) -> StubCodeElement:
        return StubCodeElement(
            id=f"{repo_name}-file",
            type="file",
            name=f"{repo_name}.py",
            file_path=f"/workspace/{repo_name}/{repo_name}.py",
            relative_path=f"{repo_name}.py",
            language="python",
            start_line=1,
            end_line=10,
            code=f"print('{repo_name}')",
            signature=None,
            docstring=None,
            summary=None,
            metadata={"embedding": [0.1, 0.2, 0.3]},
            repo_name=repo_name,
            repo_url=f"https://example.com/{repo_name}.git",
        )

    class FakeIndexer:
        def index_repository(self, repo_name=None, repo_url=None):
            return [make_element(repo_name)]

    class FakeVectorStore:
        def __init__(self):
            self.dimension = None
            self.metadata = []

        def initialize(self, dimension):
            self.dimension = dimension
            self.metadata = []

        def add_vectors(self, vectors, metadata):
            self.metadata.extend(metadata)

    fake_fc = SimpleNamespace(
        logger=_null_logger(),
        eval_config={},
        repo_loaded=True,
        repo_indexed=False,
        repo_info={"name": "repo_a", "url": "https://example.com/repo_a.git"},
        indexer=FakeIndexer(),
        embedder=SimpleNamespace(embedding_dim=3),
        vector_store=FakeVectorStore(),
        retriever=SimpleNamespace(
            index_for_bm25=lambda elements: None,
            build_repo_overview_bm25=lambda: None,
            save_bm25=lambda repo_name: None,
        ),
        graph_builder=SimpleNamespace(
            build_graphs=lambda elements, module_resolver, symbol_resolver: None,
            save=lambda repo_name: None,
        ),
        loader=SimpleNamespace(repo_path="/workspace/repo_a"),
        config={},
        _should_use_cache=lambda: False,
        _should_persist_indexes=lambda: True,
        _build_file_manifest=lambda elements, repo_root: {"files": {}},
        _save_file_manifest=lambda repo_name, manifest: None,
        _log_statistics=lambda: None,
    )

    def save_snapshot(cache_name=None):
        saved_indexes[cache_name] = [dict(meta) for meta in fake_fc.vector_store.metadata]

    fake_fc._save_to_cache = save_snapshot

    index_repository(fake_fc)

    fake_fc.repo_info = {"name": "repo_b", "url": "https://example.com/repo_b.git"}
    fake_fc.loader.repo_path = "/workspace/repo_b"
    index_repository(fake_fc)

    assert {meta["repo_name"] for meta in saved_indexes["repo_a"]} == {"repo_a"}
    assert {meta["repo_name"] for meta in saved_indexes["repo_b"]} == {"repo_b"}


def test_query_processor_llm_enhancement_does_not_print_to_stdout():
    _assert_no_print_calls(
        ROOT / "fastcode" / "query_processor.py",
        "_enhance_with_llm",
        class_name="QueryProcessor",
    )


def test_vector_store_repo_overview_search_does_not_print_to_stdout():
    _assert_no_print_calls(
        ROOT / "fastcode" / "vector_store.py",
        "search_repository_overviews",
        class_name="VectorStore",
    )


def test_retriever_repo_selection_does_not_print_to_stdout():
    _assert_no_print_calls(
        ROOT / "fastcode" / "retriever.py",
        "_select_relevant_repositories",
        class_name="HybridRetriever",
    )


def test_load_multiple_repositories_persists_manifest_for_each_repo():
    class FakeTempVectorStore:
        def __init__(self, config):
            self.config = config
            self.dimension = None
            self.metadata = []

        def initialize(self, dimension):
            self.dimension = dimension
            self.metadata = []

        def add_vectors(self, vectors, metadata):
            self.metadata.extend(metadata)

        def save(self, repo_name):
            self.saved_repo = repo_name

    class FakeCodeIndexer:
        def __init__(self, config, loader, parser, embedder, vector_store):
            self.loader = loader

        def index_repository(self, repo_name=None, repo_url=None):
            return [
                StubCodeElement(
                    id=f"{repo_name}-file",
                    type="file",
                    name=f"{repo_name}.py",
                    file_path=f"/workspace/{repo_name}/{repo_name}.py",
                    relative_path=f"{repo_name}.py",
                    language="python",
                    start_line=1,
                    end_line=10,
                    code=f"print('{repo_name}')",
                    signature=None,
                    docstring=None,
                    summary=None,
                    metadata={"embedding": [0.1, 0.2, 0.3]},
                    repo_name=repo_name,
                    repo_url=repo_url,
                )
            ]

    class FakeRetriever:
        def __init__(self, config, vector_store, embedder, graph_builder, repo_root=None):
            self.repo_root = repo_root

        def index_for_bm25(self, elements):
            self.elements = elements

        def save_bm25(self, repo_name):
            self.saved_repo = repo_name

        def build_repo_overview_bm25(self):
            self.repo_overview_built = True

    class FakeGraphBuilder:
        def __init__(self, config=None):
            self.config = config

        def build_graphs(self, elements, module_resolver=None, symbol_resolver=None):
            self.elements = elements

        def save(self, repo_name):
            self.saved_repo = repo_name

    class FakeGlobalIndexBuilder:
        def __init__(self, config):
            self.config = config

        def build_maps(self, elements, repo_root):
            self.elements = elements
            self.repo_root = repo_root

    load_multiple_repositories = _load_functions(
        FASTCODE_MAIN,
        ["load_multiple_repositories"],
        class_name="FastCode",
        global_ns={
            "np": np,
            "VectorStore": FakeTempVectorStore,
            "CodeIndexer": FakeCodeIndexer,
            "HybridRetriever": FakeRetriever,
            "CodeGraphBuilder": FakeGraphBuilder,
            "GlobalIndexBuilder": FakeGlobalIndexBuilder,
            "ModuleResolver": lambda gib: ("module_resolver", gib),
            "SymbolResolver": lambda gib, module_resolver: (
                "symbol_resolver",
                gib,
                module_resolver,
            ),
        },
    )[0]

    class FakeLoader:
        def __init__(self):
            self.repo_path = None
            self.current_name = None

        def load_from_path(self, source):
            self.current_name = os.path.basename(source)
            self.repo_path = f"/workspace/{self.current_name}"

        def get_repository_info(self):
            return {
                "name": self.current_name,
                "file_count": 1,
                "total_size_mb": 0.01,
            }

    class FakeMainVectorStore:
        def __init__(self):
            self.dimension = None
            self.merged = []

        def initialize(self, dimension):
            self.dimension = dimension

        def merge_from_index(self, repo_name):
            self.merged.append(repo_name)
            return True

    saved_manifests = []

    fake_fc = SimpleNamespace(
        logger=_null_logger(),
        config={},
        loader=FakeLoader(),
        parser=None,
        embedder=SimpleNamespace(embedding_dim=3),
        vector_store=FakeMainVectorStore(),
        graph_builder=FakeGraphBuilder(),
        loaded_repositories={},
        multi_repo_mode=False,
        repo_indexed=False,
        repo_loaded=False,
        _save_file_manifest=lambda repo_name, manifest: saved_manifests.append(repo_name),
        _build_file_manifest=lambda elements, repo_root: {"files": {}},
    )

    load_multiple_repositories(
        fake_fc,
        [
            {"source": "/tmp/repo_a", "is_url": False},
            {"source": "/tmp/repo_b", "is_url": False},
        ],
    )

    assert fake_fc.vector_store.merged == ["repo_a", "repo_b"]
    assert saved_manifests == ["repo_a", "repo_b"]


def test_sequential_index_repository_calls_do_not_save_mixed_repo_graphs():
    class FakeGlobalIndexBuilder:
        def __init__(self, config):
            self.config = config
            self.file_map = {}
            self.module_map = {}

        def build_maps(self, elements, repo_root):
            self.elements = elements
            self.repo_root = repo_root

    saved_graphs = {}

    class FakeGraphBuilder:
        def __init__(self):
            self.repo_names = []

        def build_graphs(self, elements, module_resolver, symbol_resolver):
            self.repo_names.extend(elem.repo_name for elem in elements)

        def save(self, repo_name):
            saved_graphs[repo_name] = list(self.repo_names)

    index_repository = _load_functions(
        FASTCODE_MAIN,
        ["index_repository"],
        class_name="FastCode",
        global_ns={
            "np": np,
            "CodeGraphBuilder": lambda config: FakeGraphBuilder(),
            "GlobalIndexBuilder": FakeGlobalIndexBuilder,
            "ModuleResolver": lambda gib: ("module_resolver", gib),
            "SymbolResolver": lambda gib, module_resolver: (
                "symbol_resolver",
                gib,
                module_resolver,
            ),
        },
    )[0]

    def make_element(repo_name: str) -> StubCodeElement:
        return StubCodeElement(
            id=f"{repo_name}-file",
            type="file",
            name=f"{repo_name}.py",
            file_path=f"/workspace/{repo_name}/{repo_name}.py",
            relative_path=f"{repo_name}.py",
            language="python",
            start_line=1,
            end_line=10,
            code=f"print('{repo_name}')",
            signature=None,
            docstring=None,
            summary=None,
            metadata={"embedding": [0.1, 0.2, 0.3]},
            repo_name=repo_name,
            repo_url=f"https://example.com/{repo_name}.git",
        )

    class FakeIndexer:
        def index_repository(self, repo_name=None, repo_url=None):
            return [make_element(repo_name)]

    class FakeVectorStore:
        def __init__(self):
            self.dimension = None
            self.metadata = []

        def initialize(self, dimension):
            self.dimension = dimension
            self.metadata = []

        def add_vectors(self, vectors, metadata):
            self.metadata.extend(metadata)

    fake_fc = SimpleNamespace(
        logger=_null_logger(),
        eval_config={},
        repo_loaded=True,
        repo_indexed=False,
        repo_info={"name": "repo_a", "url": "https://example.com/repo_a.git"},
        indexer=FakeIndexer(),
        embedder=SimpleNamespace(embedding_dim=3),
        vector_store=FakeVectorStore(),
        retriever=SimpleNamespace(
            index_for_bm25=lambda elements: None,
            build_repo_overview_bm25=lambda: None,
            save_bm25=lambda repo_name: None,
        ),
        graph_builder=FakeGraphBuilder(),
        loader=SimpleNamespace(repo_path="/workspace/repo_a"),
        config={},
        _should_use_cache=lambda: False,
        _should_persist_indexes=lambda: True,
        _build_file_manifest=lambda elements, repo_root: {"files": {}},
        _save_file_manifest=lambda repo_name, manifest: None,
        _save_to_cache=lambda cache_name=None: None,
        _log_statistics=lambda: None,
    )

    index_repository(fake_fc)

    fake_fc.repo_info = {"name": "repo_b", "url": "https://example.com/repo_b.git"}
    fake_fc.loader.repo_path = "/workspace/repo_b"
    index_repository(fake_fc)

    assert set(saved_graphs["repo_a"]) == {"repo_a"}
    assert set(saved_graphs["repo_b"]) == {"repo_b"}


def test_ensure_repos_ready_reuses_legacy_local_basename_indexes(monkeypatch, tmp_path):
    ensure_repos_ready = _load_functions(
        MCP_SERVER,
        ["_ensure_repos_ready"],
        global_ns={"os": os, "logger": _null_logger()},
    )[0]
    repo_name_from_source = _load_functions(
        MCP_SERVER,
        ["_repo_name_from_source"],
        global_ns={"os": os},
    )[0]
    is_repo_indexed = _load_functions(
        MCP_SERVER,
        ["_is_repo_indexed"],
        global_ns={"os": os},
    )[0]
    get_repo_name_from_path = _load_functions(
        ROOT / "fastcode" / "utils.py",
        ["get_repo_name_from_path"],
        global_ns={
            "os": os,
            "hashlib": __import__("hashlib"),
            "normalize_path": lambda path: os.path.normpath(path).replace("\\", "/"),
        },
    )[0]

    fake_fastcode = types.ModuleType("fastcode")
    fake_fastcode.__path__ = []
    fake_utils = types.ModuleType("fastcode.utils")
    fake_utils.get_repo_name_from_url = lambda source: "repo-from-url"
    fake_utils.get_repo_name_from_path = get_repo_name_from_path
    fake_fastcode.utils = fake_utils

    monkeypatch.setitem(sys.modules, "fastcode", fake_fastcode)
    monkeypatch.setitem(sys.modules, "fastcode.utils", fake_utils)

    persist_dir = tmp_path / "persist"
    persist_dir.mkdir()
    (persist_dir / "service.faiss").write_bytes(b"index")
    (persist_dir / "service_metadata.pkl").write_bytes(b"meta")

    source_repo = tmp_path / "team" / "service"
    source_repo.mkdir(parents=True)

    fake_fc = SimpleNamespace(
        _infer_is_url=lambda source: False,
        load_repository=Mock(),
        index_repository=Mock(),
        vector_store=SimpleNamespace(persist_dir=str(persist_dir)),
        loader=SimpleNamespace(safe_repo_root="/repos"),
    )

    repo_name_from_source.__globals__["_get_fastcode"] = lambda: fake_fc
    is_repo_indexed.__globals__["_get_fastcode"] = lambda: fake_fc
    ensure_repos_ready.__globals__.update(
        {
            "_get_fastcode": lambda: fake_fc,
            "_apply_forced_env_excludes": lambda fc: None,
            "_repo_name_from_source": repo_name_from_source,
            "_is_repo_indexed": is_repo_indexed,
            "_run_full_reindex": Mock(return_value={"status": "success", "count": 1}),
            "_invalidate_loaded_state": lambda fc: None,
        }
    )

    ready = ensure_repos_ready([str(source_repo)], allow_incremental=False)

    assert ready
    assert not fake_fc.load_repository.called
    assert not fake_fc.index_repository.called


def test_ensure_repos_ready_errors_on_legacy_basename_collision(monkeypatch, tmp_path):
    logged_errors = []
    logger = SimpleNamespace(
        info=lambda *args, **kwargs: None,
        warning=lambda *args, **kwargs: None,
        error=lambda *args, **kwargs: logged_errors.append((args, kwargs)),
        debug=lambda *args, **kwargs: None,
    )

    ensure_repos_ready = _load_functions(
        MCP_SERVER,
        ["_ensure_repos_ready"],
        global_ns={"os": os, "logger": logger},
    )[0]
    repo_name_from_source = _load_functions(
        MCP_SERVER,
        ["_repo_name_from_source"],
        global_ns={"os": os},
    )[0]
    is_repo_indexed = _load_functions(
        MCP_SERVER,
        ["_is_repo_indexed"],
        global_ns={"os": os},
    )[0]
    get_repo_name_from_path = _load_functions(
        ROOT / "fastcode" / "utils.py",
        ["get_repo_name_from_path"],
        global_ns={
            "os": os,
            "hashlib": __import__("hashlib"),
            "normalize_path": lambda path: os.path.normpath(path).replace("\\", "/"),
        },
    )[0]

    fake_fastcode = types.ModuleType("fastcode")
    fake_fastcode.__path__ = []
    fake_utils = types.ModuleType("fastcode.utils")
    fake_utils.get_repo_name_from_url = lambda source: "repo-from-url"
    fake_utils.get_repo_name_from_path = get_repo_name_from_path
    fake_fastcode.utils = fake_utils

    monkeypatch.setitem(sys.modules, "fastcode", fake_fastcode)
    monkeypatch.setitem(sys.modules, "fastcode.utils", fake_utils)

    persist_dir = tmp_path / "persist"
    persist_dir.mkdir()
    (persist_dir / "service.faiss").write_bytes(b"index")
    (persist_dir / "service_metadata.pkl").write_bytes(b"meta")

    original_repo = tmp_path / "team-a" / "service"
    colliding_repo = tmp_path / "team-b" / "service"
    original_repo.mkdir(parents=True)
    colliding_repo.mkdir(parents=True)

    assert get_repo_name_from_path(str(original_repo)) != get_repo_name_from_path(
        str(colliding_repo)
    )

    fake_fc = SimpleNamespace(
        _infer_is_url=lambda source: False,
        load_repository=Mock(),
        index_repository=Mock(),
        vector_store=SimpleNamespace(persist_dir=str(persist_dir)),
        loader=SimpleNamespace(safe_repo_root="/repos"),
    )

    repo_name_from_source.__globals__["_get_fastcode"] = lambda: fake_fc
    is_repo_indexed.__globals__["_get_fastcode"] = lambda: fake_fc
    ensure_repos_ready.__globals__.update(
        {
            "_get_fastcode": lambda: fake_fc,
            "_apply_forced_env_excludes": lambda fc: None,
            "_repo_name_from_source": repo_name_from_source,
            "_is_repo_indexed": is_repo_indexed,
            "_run_full_reindex": Mock(return_value={"status": "success", "count": 1}),
            "_invalidate_loaded_state": lambda fc: None,
        }
    )

    ready = ensure_repos_ready([str(colliding_repo)], allow_incremental=False)

    assert ready == []
    assert logged_errors
    assert not fake_fc.load_repository.called
    assert not fake_fc.index_repository.called


def test_load_multi_repo_cache_rebuilds_graph_from_all_loaded_repositories_when_bm25_is_partial(
    tmp_path,
):
    load_multi_repo_cache = _load_functions(
        FASTCODE_MAIN,
        ["_load_multi_repo_cache"],
        class_name="FastCode",
        global_ns={
            "os": os,
            "pickle": pickle,
            "BM25Okapi": lambda corpus: ("bm25", len(corpus)),
            "CodeElement": StubCodeElement,
            "CodeGraphBuilder": lambda config: None,
        },
    )[0]

    persist_dir = tmp_path / "persist"
    persist_dir.mkdir()
    for repo_name in ("repo_a", "repo_b"):
        (persist_dir / f"{repo_name}.faiss").write_bytes(b"index")
        (persist_dir / f"{repo_name}_metadata.pkl").write_bytes(b"meta")

    with open(persist_dir / "repo_a_bm25.pkl", "wb") as f:
        pickle.dump(
            {
                "bm25_corpus": [["repo", "a"]],
                "bm25_elements": [
                    {
                        "id": "repo_a-file",
                        "type": "file",
                        "name": "repo_a.py",
                        "file_path": "/workspace/repo_a/repo_a.py",
                        "relative_path": "repo_a.py",
                        "language": "python",
                        "start_line": 1,
                        "end_line": 10,
                        "code": "print('repo_a')",
                        "signature": None,
                        "docstring": None,
                        "summary": None,
                        "metadata": {},
                        "repo_name": "repo_a",
                        "repo_url": None,
                    }
                ],
            },
            f,
        )

    def make_element(repo_name: str) -> StubCodeElement:
        return StubCodeElement(
            id=f"{repo_name}-file",
            type="file",
            name=f"{repo_name}.py",
            file_path=f"/workspace/{repo_name}/{repo_name}.py",
            relative_path=f"{repo_name}.py",
            language="python",
            start_line=1,
            end_line=10,
            code=f"print('{repo_name}')",
            signature=None,
            docstring=None,
            summary=None,
            metadata={},
            repo_name=repo_name,
            repo_url=None,
        )

    class FakeVectorStore:
        def __init__(self):
            self.persist_dir = str(persist_dir)
            self.merged = []

        def initialize(self, dimension):
            self.dimension = dimension

        def merge_from_index(self, repo_name):
            self.merged.append(repo_name)
            return True

        def get_count(self):
            return len(self.merged)

    class FakeGraphBuilder:
        def __init__(self):
            self.rebuilt_repo_names = []

        def load(self, repo_name):
            return False

        def merge_from_file(self, repo_name):
            return False

        def build_graphs(self, elements):
            self.rebuilt_repo_names = [elem.repo_name for elem in elements]

    graph_builder = FakeGraphBuilder()
    load_multi_repo_cache.__globals__["CodeGraphBuilder"] = lambda config: graph_builder

    fake_fc = SimpleNamespace(
        logger=_null_logger(),
        config={},
        vector_store=FakeVectorStore(),
        embedder=SimpleNamespace(embedding_dim=3),
        loaded_repositories={"repo_a": {"name": "repo_a"}, "repo_b": {"name": "repo_b"}},
        retriever=SimpleNamespace(
            persist_dir=str(persist_dir),
            build_repo_overview_bm25=lambda: None,
            index_for_bm25=lambda elements: None,
            full_bm25_elements=[],
            full_bm25_corpus=[],
            full_bm25=None,
        ),
        graph_builder=graph_builder,
        _reconstruct_elements_from_metadata=lambda: [
            make_element("repo_a"),
            make_element("repo_b"),
        ],
    )

    ok = load_multi_repo_cache(fake_fc, repo_names=["repo_a", "repo_b"])

    assert ok is True
    assert set(graph_builder.rebuilt_repo_names) == {"repo_a", "repo_b"}
