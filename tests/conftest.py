"""
Test configuration.

`api.py` and `web_app.py` import `fastcode`, whose package `__init__` pulls in the
whole indexing/ML stack. The upload tests only need the HTTP layer and the small
dependency-free helpers in `fastcode/upload_utils.py`, so `fastcode` is replaced by
a stub package that still resolves its submodules from the source tree.
"""

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class FastCode:
    """Stand-in for fastcode.FastCode (never instantiated by the tests)."""


_stub = types.ModuleType("fastcode")
_stub.__path__ = [str(ROOT / "fastcode")]
_stub.FastCode = FastCode
sys.modules["fastcode"] = _stub
