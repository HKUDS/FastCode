"""
Helpers for file names and paths that come from uploads.

The name of an uploaded file is chosen by the client, so it must never be used
as part of a filesystem path as-is: it can contain directory components
("../../x.zip"), be an absolute path ("/etc/x.zip") or carry a drive
("C:\\x.zip"), any of which would make the server read, write or move files
outside of the directory it means to use.

This module deliberately has no dependencies besides the standard library.
"""

import ntpath
from pathlib import Path
from typing import Optional, Union


def sanitize_upload_filename(filename: Optional[str]) -> str:
    """
    Return only the final component of a client supplied file name.

    Both "/" and "\\" are treated as separators (browsers used to send the full
    client path, and an attacker can send whatever they like), and drive
    letters are dropped.

    Raises:
        ValueError: if nothing usable is left (empty, ".", "..").
    """
    name = ntpath.basename((filename or "").replace("\x00", ""))
    if name in ("", ".", ".."):
        raise ValueError("Invalid file name")
    return name


def ensure_within_directory(directory: Union[str, Path], path: Union[str, Path]) -> Path:
    """
    Return `path` unchanged if it resolves to a location strictly inside
    `directory`.

    Symlinks are resolved on both sides. `directory` itself is not accepted: an
    empty or "." name would otherwise make the caller operate on the whole
    directory.

    Raises:
        ValueError: if `path` is `directory` or is outside of it.
    """
    base = Path(directory).resolve()
    target = Path(path).resolve()
    if target == base or base not in target.parents:
        raise ValueError(f"'{path}' is not inside '{directory}'")
    return Path(path)
