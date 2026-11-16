"""``python -m praxis.serve`` sets the compile cache before it imports torch.

The check is the one of ``tests/test_compile_cache.py`` for the benchmark
script. It lives here because only the server tests may import the server.
"""

import sys
from pathlib import Path

import pytest

# The server needs the [serve] extra.
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
pytest.importorskip("msgpack")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tests"))  # test_compile_cache, as tests/ imports it

from test_compile_cache import check_entry_point  # noqa: E402


def test_server_sets_the_cache_before_torch(tmp_path: Path) -> None:
    check_entry_point("praxis.serve.__main__", tmp_path)
