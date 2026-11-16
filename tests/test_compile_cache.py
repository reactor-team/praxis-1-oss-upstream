"""The compile cache defaults and the threaded shard load (CPU only, no weights).

``configure_compile_cache`` sets the cache variables under
``PRAXIS_CACHE_DIR`` and keeps the values that are already set. The serve
entry point and the benchmark script set them before torch is imported: a
probe in a child process records the environment at the first import of
torch (the server entry point in ``tests/serve/test_serve_compile_cache.py``).
The shards of a checkpoint load to the same bytes, in the same order,
with one thread or several.

The tests run without the cache and load variables of the caller's
environment and restore them after each test.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as installed_version
from pathlib import Path
from typing import Any

import pytest
import torch

from praxis import checkpoint, compile_cache
from praxis.checkpoint import (
    LOAD_WORKERS,
    CheckpointError,
    load_state_dict_from_dir,
    save_checkpoint,
)
from praxis.compile_cache import (
    CACHE_DIR_ENV,
    cache_env,
    cache_version,
    configure_compile_cache,
)

ROOT = Path(__file__).resolve().parents[1]
KEYS = list(cache_env("."))
# The seconds that a child process may take (it imports torch).
TIMEOUT = 600


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the cache and load variables; restore them after the test."""
    for key in (*KEYS, CACHE_DIR_ENV, "PRAXIS_LOAD_WORKERS"):
        # setenv records the value to restore, also for a variable that a
        # test sets later through os.environ.
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)


def test_no_cache_dir_changes_nothing() -> None:
    env: dict[str, str] = {}
    assert configure_compile_cache(environ=env) == {}
    assert env == {}
    env = {CACHE_DIR_ENV: ""}
    assert configure_compile_cache(environ=env) == {}
    assert env == {CACHE_DIR_ENV: ""}


def test_defaults_under_the_cache_dir(tmp_path: Path) -> None:
    root = tmp_path / "cache"
    base = root / cache_version()
    env = {CACHE_DIR_ENV: str(root)}
    applied = configure_compile_cache(environ=env)
    assert root.is_dir()
    assert applied == {
        "TORCHINDUCTOR_CACHE_DIR": str(base / "inductor"),
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
        "TRITON_CACHE_DIR": str(base / "triton"),
        "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
        "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": str(base / "flash_attn_cute"),
    }
    assert {k: env[k] for k in applied} == applied


def test_the_versions_name_the_cache_subdirectory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An upgrade of torch, Triton, flash-attn-4 or the CuTe DSL changes the dir."""
    assert cache_version().startswith(f"torch-{installed_version('torch')}_")
    versions = {"torch": "2.9.1+cu130", "triton": "3.5.0", "flash-attn-4": "4.0.0b37"}

    def fake_version(name: str) -> str:
        if name not in versions:
            raise PackageNotFoundError(name)
        return versions[name]

    monkeypatch.setattr(compile_cache, "version", fake_version)
    name = cache_version()
    assert name == (
        "torch-2.9.1+cu130_triton-3.5.0_flash-attn-4-4.0.0b37_nvidia-cutlass-dsl-none"
    )
    for package in ("torch", "triton", "flash-attn-4", "nvidia-cutlass-dsl"):
        old = versions.get(package)
        versions[package] = "9.9/9"
        assert cache_version() != name
        assert "/" not in cache_version()
        if old is None:
            del versions[package]
        else:
            versions[package] = old
    assert cache_version() == name


def test_home_and_relative_cache_dirs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    applied = configure_compile_cache("~/c", environ={})
    assert (tmp_path / "home" / "c").is_dir()
    assert applied["TRITON_CACHE_DIR"] == str(
        tmp_path / "home" / "c" / cache_version() / "triton"
    )
    monkeypatch.chdir(tmp_path)
    applied = configure_compile_cache(environ={CACHE_DIR_ENV: "rel"})
    assert (tmp_path / "rel").is_dir()
    assert applied["TRITON_CACHE_DIR"] == str(
        tmp_path / "rel" / cache_version() / "triton"
    )


def test_warns_when_torch_is_already_imported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.WARNING, logger=compile_cache.__name__)
    configure_compile_cache(tmp_path / "c", environ={})
    warned = [r for r in caplog.records if r.name == compile_cache.__name__]
    assert len(warned) == 1
    assert "torch" in warned[0].getMessage()
    assert CACHE_DIR_ENV in warned[0].getMessage()
    caplog.clear()
    monkeypatch.setattr(compile_cache, "_READERS", ("not_a_module_of_praxis",))
    configure_compile_cache(tmp_path / "c", environ={})
    assert not [r for r in caplog.records if r.name == compile_cache.__name__]


def test_user_values_win(tmp_path: Path) -> None:
    env = {
        "TRITON_CACHE_DIR": "/mine/triton",
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "0",
        "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "0",
    }
    applied = configure_compile_cache(tmp_path / "c", environ=env)
    assert env["TRITON_CACHE_DIR"] == "/mine/triton"
    assert env["TORCHINDUCTOR_FX_GRAPH_CACHE"] == "0"
    assert env["FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED"] == "0"
    assert applied["TRITON_CACHE_DIR"] == "/mine/triton"
    assert env["TORCHINDUCTOR_CACHE_DIR"] == str(
        tmp_path / "c" / cache_version() / "inductor"
    )


def test_argument_before_environment(tmp_path: Path) -> None:
    env = {CACHE_DIR_ENV: str(tmp_path / "env")}
    configure_compile_cache(tmp_path / "arg", environ=env)
    assert env["TRITON_CACHE_DIR"] == str(tmp_path / "arg" / cache_version() / "triton")


# Records the cache variables at the first import of a module (torch), then
# runs the top level of the target (a module name or a script path) and prints
# them.
# The target can fail after torch (an extra that is not installed): the
# environment at the import is what counts.
PROBE = """
import importlib, importlib.abc, json, os, runpy, sys
keys, target, module = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3]
seen = {}
class Probe(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == module and module not in seen:
            seen[module] = {k: os.environ.get(k) for k in keys}
        return None
sys.meta_path.insert(0, Probe())
try:
    if target.endswith(".py"):
        runpy.run_path(target, run_name="probe")
    else:
        importlib.import_module(target)
except Exception as e:
    seen["error"] = repr(e)
print("PROBE" + json.dumps(seen))
"""


def env_at_torch_import(
    target: str, env: dict[str, str], module: str = "torch"
) -> dict:
    """Return the cache variables when ``target`` first imports ``module``."""
    base = {k: v for k, v in os.environ.items() if k not in (*KEYS, CACHE_DIR_ENV)}
    base["PYTHONPATH"] = os.pathsep.join(
        p for p in (str(ROOT), os.environ.get("PYTHONPATH")) if p
    )
    out = subprocess.run(
        [sys.executable, "-c", PROBE, json.dumps(KEYS), target, module],
        env={**base, **env},
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=TIMEOUT,
    ).stdout
    line = next(x for x in out.splitlines() if x.startswith("PROBE"))
    seen = json.loads(line.removeprefix("PROBE"))
    assert module in seen, seen
    return seen[module]


def check_entry_point(target: str, tmp_path: Path) -> None:
    """Check that ``target`` sets the cache variables before it imports torch."""
    root = tmp_path / "cache"
    seen = env_at_torch_import(
        target, {CACHE_DIR_ENV: str(root), "TRITON_CACHE_DIR": "/mine/triton"}
    )
    expected = cache_env(root)
    expected["TRITON_CACHE_DIR"] = "/mine/triton"
    assert seen == expected
    # Without PRAXIS_CACHE_DIR the libraries keep their own defaults.
    assert env_at_torch_import(target, {}) == dict.fromkeys(KEYS)


def test_benchmark_sets_the_cache_before_torch(tmp_path: Path) -> None:
    check_entry_point(str(ROOT / "scripts" / "benchmark_inference.py"), tmp_path)


def test_compile_cache_imports_no_torch() -> None:
    code = (
        "import sys; from praxis.compile_cache import configure_compile_cache; "
        "configure_compile_cache(); assert 'torch' not in sys.modules"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        env={**os.environ, CACHE_DIR_ENV: ""},
        check=True,
        timeout=TIMEOUT,
    )


def _state() -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    state = {}
    for i in range(12):
        state[f"backbone.layers.{i}.w"] = torch.randn(64, 32, generator=g).to(
            torch.bfloat16
        )
        state[f"backbone.layers.{i}.bias"] = torch.randn(32, generator=g)
        state[f"expert.layers.{i}.w"] = torch.randn(16, 8, generator=g).half()
    return state


@pytest.mark.parametrize("prefix", ["", "backbone."])
def test_threaded_load_is_the_sequential_load(tmp_path: Path, prefix: str) -> None:
    state = _state()
    save_checkpoint(tmp_path / "ckpt", state, {"x": 1}, max_shard_bytes=6000)
    shards = list((tmp_path / "ckpt").glob("model-*.safetensors"))
    assert len(shards) > 4
    one = load_state_dict_from_dir(tmp_path / "ckpt", prefix=prefix, workers=1)
    assert set(one) == {n[len(prefix) :] for n in state if n.startswith(prefix)}
    for workers in (2, 4, 16, None):
        many = load_state_dict_from_dir(
            tmp_path / "ckpt", prefix=prefix, workers=workers
        )
        assert list(many) == list(one)
        for name, t in one.items():
            assert many[name].dtype == t.dtype
            assert many[name].shape == t.shape
            assert (
                many[name].view(torch.uint8).numpy().tobytes()
                == t.view(torch.uint8).numpy().tobytes()
            )
    for name, t in one.items():
        assert torch.equal(t, state[prefix + name])


def _index(path: Path) -> tuple[Path, dict]:
    index = path / "model.safetensors.index.json"
    return index, json.loads(index.read_text())


def test_load_raises_on_a_tensor_that_the_index_does_not_list(
    tmp_path: Path,
) -> None:
    """A shard holds a tensor more than the index lists."""
    save_checkpoint(tmp_path / "ckpt", _state(), {"x": 1}, max_shard_bytes=6000)
    index, data = _index(tmp_path / "ckpt")
    del data["weight_map"][next(iter(data["weight_map"]))]
    index.write_text(json.dumps(data))
    for workers in (1, 4):
        with pytest.raises(CheckpointError, match="does not hold.*1 extra, 0 missing"):
            load_state_dict_from_dir(tmp_path / "ckpt", workers=workers)


def test_load_raises_on_a_missing_tensor_outside_the_prefix(tmp_path: Path) -> None:
    """The check covers the whole shard, not only the tensors of the prefix."""
    save_checkpoint(tmp_path / "ckpt", _state(), {"x": 1}, max_shard_bytes=6000)
    index, data = _index(tmp_path / "ckpt")
    wmap = data["weight_map"]
    shard = next(f for n, f in wmap.items() if n.startswith("backbone."))
    wmap["expert.not.in.the.shard"] = shard
    index.write_text(json.dumps(data))
    for workers in (1, 4):
        with pytest.raises(CheckpointError, match="does not hold.*0 extra, 1 missing"):
            load_state_dict_from_dir(
                tmp_path / "ckpt", prefix="backbone.", workers=workers
            )


class _Pools:
    """Record the thread pools of the loader and the threads that read shards."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.sizes: list[int] = []
        self.threads: list[str] = []
        pools = self

        class Pool(checkpoint.ThreadPoolExecutor):
            def __init__(self, max_workers: int, **kwargs: Any) -> None:
                pools.sizes.append(max_workers)
                super().__init__(max_workers, **kwargs)

        read = checkpoint._read_shard

        def record(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
            self.threads.append(threading.current_thread().name)
            return read(*args, **kwargs)

        monkeypatch.setattr(checkpoint, "ThreadPoolExecutor", Pool)
        monkeypatch.setattr(checkpoint, "_read_shard", record)


def test_the_shards_load_in_parallel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two shards are read at the same time, on the threads of the pool."""
    save_checkpoint(tmp_path / "ckpt", _state(), {"x": 1}, max_shard_bytes=6000)
    pools = _Pools(monkeypatch)
    # The first two reads wait for each other: a sequential load times out.
    barrier = threading.Barrier(2, timeout=30)
    read = checkpoint._read_shard
    calls: list[int] = []
    lock = threading.Lock()

    def meet(*args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        with lock:
            calls.append(1)
            first = len(calls) <= 2
        if first:
            barrier.wait()
        return read(*args, **kwargs)

    monkeypatch.setattr(checkpoint, "_read_shard", meet)
    load_state_dict_from_dir(tmp_path / "ckpt", workers=2)
    assert pools.sizes == [2]
    assert len(set(pools.threads)) == 2
    assert all(name.startswith("praxis-load") for name in pools.threads)


@pytest.mark.parametrize(
    ("value", "count"),
    [
        (None, LOAD_WORKERS),
        ("", LOAD_WORKERS),
        ("3", 3),
        (" 2 ", 2),
        ("1", 1),
        ("0", 1),
        ("-2", 1),
    ],
)
def test_load_workers_from_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str | None, count: int
) -> None:
    """PRAXIS_LOAD_WORKERS is read at each load; below 1 is 1 thread."""
    state = _state()
    save_checkpoint(tmp_path / "ckpt", state, {"x": 1}, max_shard_bytes=6000)
    pools = _Pools(monkeypatch)
    if value is not None:
        monkeypatch.setenv("PRAXIS_LOAD_WORKERS", value)
    out = load_state_dict_from_dir(tmp_path / "ckpt")
    assert set(out) == set(state)
    if count == 1:
        assert pools.sizes == []
        assert set(pools.threads) == {threading.current_thread().name}
    else:
        assert pools.sizes == [count]
        assert all(name.startswith("praxis-load") for name in pools.threads)
    # The argument wins over the environment.
    pools.sizes.clear()
    load_state_dict_from_dir(tmp_path / "ckpt", workers=5)
    assert pools.sizes == [5]


@pytest.mark.parametrize("value", ["four", "2.5"])
def test_a_bad_load_workers_value_fails_the_load_not_the_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    save_checkpoint(tmp_path / "ckpt", _state(), {"x": 1}, max_shard_bytes=6000)
    monkeypatch.setenv("PRAXIS_LOAD_WORKERS", value)
    with pytest.raises(ValueError, match="PRAXIS_LOAD_WORKERS must be an integer"):
        load_state_dict_from_dir(tmp_path / "ckpt")
    subprocess.run(
        [sys.executable, "-c", "import praxis.checkpoint"],
        cwd=ROOT,
        env={**os.environ, "PRAXIS_LOAD_WORKERS": value},
        check=True,
        timeout=TIMEOUT,
    )


def test_threaded_load_raises_on_a_bad_shard(tmp_path: Path) -> None:
    save_checkpoint(tmp_path / "ckpt", _state(), {"x": 1}, max_shard_bytes=6000)
    index = tmp_path / "ckpt" / "model.safetensors.index.json"
    data = json.loads(index.read_text())
    name = next(iter(data["weight_map"]))
    data["weight_map"]["missing.tensor"] = data["weight_map"][name]
    index.write_text(json.dumps(data))
    with pytest.raises(CheckpointError, match="does not hold"):
        load_state_dict_from_dir(tmp_path / "ckpt", workers=4)
