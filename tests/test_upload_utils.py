import os

import pytest

from fastcode.upload_utils import ensure_within_directory, sanitize_upload_filename


@pytest.mark.parametrize(
    "filename, expected",
    [
        ("project-main.zip", "project-main.zip"),
        ("my project (1).zip", "my project (1).zip"),
        ("../../x.zip", "x.zip"),
        ("a/b/c.zip", "c.zip"),
        ("/etc/cron.d/evil.zip", "evil.zip"),
        ("..\\..\\x.zip", "x.zip"),
        ("C:\\Users\\me\\x.zip", "x.zip"),
        ("C:x.zip", "x.zip"),
        ("a/..\\b/c.zip", "c.zip"),
        ("x\x00.zip", "x.zip"),
        ("..zip", "..zip"),
    ],
)
def test_sanitize_keeps_only_the_last_component(filename, expected):
    assert sanitize_upload_filename(filename) == expected


@pytest.mark.parametrize("filename", [None, "", ".", "..", "dir/", "a/b/", "a/..", "..\\", "\x00"])
def test_sanitize_rejects_names_with_nothing_usable(filename):
    with pytest.raises(ValueError):
        sanitize_upload_filename(filename)


def test_ensure_within_directory_accepts_children(tmp_path):
    repos = tmp_path / "repos"
    repos.mkdir()
    assert ensure_within_directory(repos, repos / "project") == repos / "project"
    assert ensure_within_directory(repos, repos / "a" / "b") == repos / "a" / "b"
    assert ensure_within_directory(str(repos), str(repos / "x.y")) == repos / "x.y"


@pytest.mark.parametrize("name", ["", ".", "..", "../x", "a/../../x", "a/../.."])
def test_ensure_within_directory_rejects_the_directory_and_outside_paths(tmp_path, name):
    repos = tmp_path / "repos"
    repos.mkdir()
    with pytest.raises(ValueError):
        ensure_within_directory(repos, repos / name)


def test_ensure_within_directory_rejects_a_sibling_sharing_the_prefix(tmp_path):
    repos = tmp_path / "repos"
    repos.mkdir()
    with pytest.raises(ValueError):
        ensure_within_directory(repos, tmp_path / "repos2" / "x")


def test_ensure_within_directory_rejects_absolute_paths_outside(tmp_path):
    repos = tmp_path / "repos"
    repos.mkdir()
    with pytest.raises(ValueError):
        ensure_within_directory(repos, os.path.join(str(tmp_path), "elsewhere"))


@pytest.mark.skipif(not hasattr(os, "symlink"), reason="needs symlink support")
def test_ensure_within_directory_rejects_a_symlink_to_outside(tmp_path):
    repos = tmp_path / "repos"
    repos.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    try:
        os.symlink(outside, repos / "link")
    except OSError:
        pytest.skip("cannot create symlinks")
    with pytest.raises(ValueError):
        ensure_within_directory(repos, repos / "link" / "x")
