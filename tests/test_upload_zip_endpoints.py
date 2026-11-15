"""
The upload endpoints must not use the client supplied file name to read, write or
move anything outside of the directories they are meant to use.

`api.py` (/upload-zip) and `web_app.py` (/api/upload-zip) have the same handler.
The repos and temporary directories are nested inside tmp_path so that, even if the
code under test did not confine its writes, a "../" hop or the absolute path used
below would still end up inside tmp_path.
"""

import importlib.util
import io
import itertools
import os
import tempfile
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from conftest import ROOT

APPS = [("api", "/upload-zip"), ("web_app", "/api/upload-zip")]
_counter = itertools.count()


def make_zip(name="project/README.md", content=b"hello"):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr(name, content)
    return buffer.getvalue()


class Loader:
    def __init__(self, repos):
        self.safe_repo_root = str(repos)
        self.backups = []

    def _backup_existing_repo(self, path):
        self.backups.append(path)


class Instance:
    repo_info = {}

    def __init__(self, repos):
        self.loader = Loader(repos)
        self.loaded = []

    def load_repository(self, path, is_url=False):
        self.loaded.append(path)


class Env:
    def __init__(self, client, url, root, repos, instance):
        self.client = client
        self.url = url
        self.root = root
        self.repos = repos
        self.instance = instance

    def upload(self, filename, data=None):
        data = make_zip() if data is None else data
        return self.client.post(self.url, files={"file": (filename, data, "application/zip")})

    def unexpected_entries(self):
        """Everything under root that is not in a place the endpoints may use."""
        allowed = (self.root / "cwd", self.repos, self.root / "a" / "b" / "tmp")
        found = []
        for dirpath, dirnames, filenames in os.walk(self.root):
            for name in dirnames + filenames:
                path = Path(dirpath, name)
                if path in allowed or any(a in path.parents for a in allowed):
                    continue
                if path in (self.root / "a", self.root / "a" / "b"):
                    continue
                found.append(str(path.relative_to(self.root)))
        return sorted(found)


@pytest.fixture(params=APPS, ids=[a[0] for a in APPS])
def env(request, tmp_path, monkeypatch):
    module_name, url = request.param
    cwd = tmp_path / "cwd"
    repos = tmp_path / "a" / "b" / "repos"
    temp = tmp_path / "a" / "b" / "tmp"
    for directory in (cwd, repos, temp):
        directory.mkdir(parents=True)
    monkeypatch.chdir(cwd)  # both modules create ./logs when imported
    monkeypatch.setattr(tempfile, "tempdir", str(temp))

    spec = importlib.util.spec_from_file_location(
        f"{module_name}_under_test_{next(_counter)}", ROOT / f"{module_name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    instance = Instance(repos)
    module.fastcode_instance = instance
    return Env(TestClient(module.app), url, tmp_path, repos, instance)


def test_a_regular_upload_is_extracted(env):
    response = env.upload("project-main.zip")

    assert response.status_code == 200, response.text
    assert (env.repos / "project" / "README.md").read_bytes() == b"hello"
    assert response.json()["repo_path"] == str(env.repos / "project")
    assert env.instance.loaded == [str(env.repos / "project")]
    assert env.instance.loader.backups == []
    assert env.unexpected_entries() == []


@pytest.mark.parametrize(
    "filename, repo",
    [
        ("../../outside.zip", "outside"),
        ("../../../outside-main.zip", "outside"),
        ("a/../../../outside.zip", "outside"),
        ("..\\..\\win-style.zip", "win-style"),
        ("C:\\Users\\me\\drive.zip", "drive"),
    ],
)
def test_directories_in_the_file_name_are_dropped(env, filename, repo):
    response = env.upload(filename)

    assert response.status_code == 200, response.text
    assert (env.repos / repo / "README.md").exists()
    assert response.json()["repo_path"] == str(env.repos / repo)
    assert env.unexpected_entries() == []


def test_an_absolute_file_name_is_reduced_to_its_last_component(env):
    target = env.root / "absolute-target.zip"

    response = env.upload(str(target))

    assert response.status_code == 200, response.text
    assert not target.exists()
    assert (env.repos / "absolute-target" / "README.md").exists()
    assert env.unexpected_entries() == []


@pytest.mark.parametrize("filename", [".zip", "-main.zip", "_master.zip", "...zip", "..zip", "x/.zip"])
def test_a_name_that_is_not_a_child_of_the_repos_directory_is_rejected(env, filename):
    (env.repos / "keep").mkdir()

    response = env.upload(filename)

    assert response.status_code == 400, response.text
    # Nothing was moved away (the repos directory itself would be backed up
    # for an empty name) and nothing was written.
    assert env.instance.loader.backups == []
    assert env.instance.loaded == []
    assert sorted(p.name for p in env.repos.iterdir()) == ["keep"]
    assert env.unexpected_entries() == []


def test_an_existing_repository_is_backed_up_only_within_the_repos_directory(env):
    existing = env.repos / "existing"
    existing.mkdir()

    response = env.upload("../../existing.zip")

    assert response.status_code == 200, response.text
    assert env.instance.loader.backups == [str(existing)]


def test_an_invalid_zip_with_a_traversal_name_does_not_move_directories_outside(env):
    victim = env.root / "a" / "victim"
    victim.mkdir()
    (victim / "important.txt").write_text("keep me")

    response = env.upload("../../victim.zip", data=b"this is not a zip file")

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid ZIP file"
    assert env.instance.loader.backups == []
    assert (victim / "important.txt").read_text() == "keep me"


def test_an_empty_file_name_is_rejected(env):
    response = env.upload("dir/")

    assert response.status_code == 400
    assert env.instance.loader.backups == []
    assert env.unexpected_entries() == []
