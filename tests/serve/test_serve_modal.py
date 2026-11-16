"""Tests of the Modal app (CPU only, no weights, no Modal credentials).

The tests import praxis/serve/modal_app.py and run the boot of the app in
this process with ``web.local()``. A stand-in replaces the networks of the
pipeline. Nothing connects to Modal: a stand-in replaces the compile cache
volume, and the tests run without the cache variables of the caller.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path

import pytest

# The Modal tests need the [serve] extra.
pytest.importorskip("modal")
pytest.importorskip("fastapi")
pytest.importorskip("uvicorn")
pytest.importorskip("websockets")

import uvicorn
from fastapi import FastAPI
from test_serve_protocol import (
    SINGLE_ARM,
    FakePipeline,
    _droid_obs,
    _free_port,
    _open,
    _recv,
)
from websockets.sync.client import connect

from praxis.compile_cache import CACHE_DIR_ENV, cache_env, cache_version
from praxis.serve import modal_app, server, wire

ROOT = Path(__file__).resolve().parents[2]
CACHE_KEYS = (*cache_env("."), CACHE_DIR_ENV, "PRAXIS_MODAL_CACHE_VOLUME")


class FakeVolume:
    """The ``commit`` of a Modal volume; it can fail."""

    def __init__(self, fail: bool = False) -> None:
        self.commits = 0
        self.fail = fail

    def commit(self) -> None:
        if self.fail:
            raise RuntimeError("commit failed")
        self.commits += 1


@pytest.fixture(autouse=True)
def _no_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No cache volume and no cache variables; restore them after the test."""
    for key in CACHE_KEYS:
        # setenv records the value to restore, also for a variable that the
        # boot sets later through os.environ.
        monkeypatch.setenv(key, "")
        monkeypatch.delenv(key)
    monkeypatch.setattr(modal_app, "cache_volume", None)
    monkeypatch.setattr(modal_app, "CACHE_MOUNT", tmp_path / "mount")
    monkeypatch.setattr(modal_app, "CACHE_LOCAL", tmp_path / "local")


def _boot_setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, on_build: object = None
) -> dict[str, str]:
    """Set up a boot of the app on the CPU; return the model paths it loads.

    ``on_build`` runs when the server builds the pipeline (the warmup).
    """
    paths: dict[str, str] = {}

    def pipeline(**kwargs: object) -> FakePipeline:
        paths.update(
            {k: str(kwargs[k]) for k in ("checkpoint", "vae_weights", "text_encoder")}
        )
        if callable(on_build):
            on_build()
        sampler = kwargs["sampler"]
        return FakePipeline(SINGLE_ARM, sampler)  # ty: ignore[invalid-argument-type]

    # The serve config of the image, on the CPU.
    cfg = server.load_config(
        ROOT / "configs" / "serve_droid.yaml", ["model.device=cpu"]
    )
    config = tmp_path / "configs" / "serve_test.yaml"
    config.parent.mkdir(exist_ok=True)
    config.write_text(json.dumps(cfg))
    monkeypatch.setattr(server, "PraxisPipeline", pipeline)
    monkeypatch.setattr(modal_app, "IMAGE_REPO", tmp_path)
    monkeypatch.setenv("PRAXIS_CONFIG", "configs/serve_test.yaml")
    # A leading "/" is the root of the volume, as in "modal volume put".
    monkeypatch.setenv("PRAXIS_CHECKPOINT", "/checkpoints/other")
    monkeypatch.setenv("PRAXIS_SKIP_WARMUP", "1")
    return paths


def test_import_defines_the_app_and_needs_no_credentials() -> None:
    assert modal_app.app.name == "praxis-serve"
    assert modal_app.CONTAINER_ENV["PRAXIS_SERVED_COMMIT"]
    assert all(k.startswith("PRAXIS_") for k in modal_app.CONTAINER_ENV)


def test_the_image_is_the_example_dockerfile() -> None:
    dockerfile = (ROOT / "docker" / "Dockerfile").read_text()
    assert "WORKDIR " + str(modal_app.IMAGE_REPO) in dockerfile
    assert '"python", "-m", "praxis.serve"' in dockerfile
    rules = [
        line
        for line in (ROOT / ".dockerignore").read_text().splitlines()
        if line and not line.startswith("#")
    ]
    assert rules[0] == "*"
    assert {"!pyproject.toml", "!README.md", "!praxis", "!configs"} <= set(rules)


def test_the_image_has_the_flash_attention_package() -> None:
    """The single-arm checkpoint asks for the flash attention backend.

    Its first policy call fails without ``flash-attn-4``. The install resolves
    the torch dependencies again, so it must come before the cuDNN pin.
    """
    lines = (ROOT / "docker" / "Dockerfile").read_text().splitlines()
    runs = [i for i, line in enumerate(lines) if line.startswith("RUN ")]
    flash = [i for i in runs if "flash-attn-4" in lines[i]]
    cudnn = [i for i in runs if "nvidia-cudnn-cu13==" in lines[i]]
    assert len(flash) == 1 and len(cudnn) == 1
    assert "flash-attn-4[cu13]==" in lines[flash[0]]
    assert flash[0] < cudnn[0]


def test_web_local_serves_the_server_app(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``web.local()`` runs the boot and returns the app of ``python -m praxis.serve``."""
    paths = _boot_setup(monkeypatch, tmp_path)
    app = modal_app.PolicyServer().web.local()
    assert type(app) is FastAPI
    assert paths == {
        "checkpoint": "/data/checkpoints/other",
        "vae_weights": "/data/weights/Wan2.2_VAE.pth",
        "text_encoder": "/data/weights/Qwen3.5-2B",
    }

    port = _free_port()
    srv = uvicorn.Server(
        uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    )
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not srv.started:
        assert time.monotonic() < deadline, "the server did not start"
        time.sleep(0.02)
    try:
        addr = f"127.0.0.1:{port}"
        ws, ready = _open(addr, "/api/v1/session?sample_seed=0&schedule=chunked")
        with ws:
            assert ready["meta"]["checkpoint"] == "/data/checkpoints/other"
            ws.send(wire.serialise(_droid_obs(1)))
            assert len(_recv(ws)["result"]) == 21
        with connect(f"ws://{addr}/api/v1/session/nope", open_timeout=10) as ws:
            assert _recv(ws)["status"] == "error"
    finally:
        srv.should_exit = True
        thread.join(timeout=10)


def test_boot_uses_the_cache_volume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The boot loads the volume, compiles, copies the new files back, commits."""
    mount, local = tmp_path / "mount", tmp_path / "local"
    ver = cache_version()
    old = mount / ver / "triton" / "old" / "old.cubin"
    old.parent.mkdir(parents=True)
    old.write_bytes(b"old")

    def compile_kernel() -> None:
        # The warmup writes a kernel to the cache of the process.
        new = Path(os.environ["TRITON_CACHE_DIR"]) / "new" / "new.cubin"
        new.parent.mkdir(parents=True, exist_ok=True)
        new.write_bytes(b"new")

    _boot_setup(monkeypatch, tmp_path, compile_kernel)
    volume = FakeVolume()
    monkeypatch.setattr(modal_app, "cache_volume", volume)
    assert type(modal_app.PolicyServer().web.local()) is FastAPI
    assert os.environ["TRITON_CACHE_DIR"] == str(local / ver / "triton")
    assert (local / ver / "triton" / "old" / "old.cubin").read_bytes() == b"old"
    assert (mount / ver / "triton" / "new" / "new.cubin").read_bytes() == b"new"
    assert volume.commits == 1
    # A second boot finds no new file: no commit.
    modal_app.PolicyServer().web.local()
    assert volume.commits == 1


def test_the_cache_volume_is_best_effort(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A failed copy or commit logs a warning; the boot goes on."""
    # The boot resets the root handlers, which would drop those of caplog.
    monkeypatch.setattr(logging, "basicConfig", lambda **kwargs: None)
    caplog.set_level(logging.WARNING, logger=modal_app.__name__)
    local = tmp_path / "local"
    entry = tmp_path / "mount" / "triton" / "k" / "k.cubin"
    entry.parent.mkdir(parents=True)
    entry.write_bytes(b"k")

    def write_new() -> None:
        new = Path(os.environ["TRITON_CACHE_DIR"]) / "n" / "n.cubin"
        new.parent.mkdir(parents=True, exist_ok=True)
        new.write_bytes(b"n")

    _boot_setup(monkeypatch, tmp_path, write_new)
    volume = FakeVolume(fail=True)
    monkeypatch.setattr(modal_app, "cache_volume", volume)
    copy_new = modal_app.copy_new

    def copy_in_fails(src: Path, dst: Path) -> int:
        if src == modal_app.CACHE_MOUNT:
            # A partial copy, then an error: the boot starts empty.
            (dst / "partial").mkdir(parents=True)
            raise OSError("volume read failed")
        return copy_new(src, dst)

    monkeypatch.setattr(modal_app, "copy_new", copy_in_fails)
    assert type(modal_app.PolicyServer().web.local()) is FastAPI
    messages = [r.getMessage() for r in caplog.records]
    assert any("copy from the volume failed" in m for m in messages)
    assert any("copy to the volume failed" in m for m in messages)
    # The local cache started empty and holds the new kernel only.
    assert not (local / "partial").exists()
    assert sorted(p.name for p in local.rglob("*.cubin")) == ["n.cubin"]
    assert os.environ["TRITON_CACHE_DIR"].startswith(str(local))
    assert volume.commits == 0


def test_boot_sets_the_cache_before_the_server_import(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The boot sets the cache variables before it imports torch and the server."""
    monkeypatch.syspath_prepend(str(ROOT / "tests"))
    from test_compile_cache import env_at_torch_import

    local = tmp_path / "local"
    script = tmp_path / "boot.py"
    script.write_text(
        "from pathlib import Path\n"
        "from praxis.serve import modal_app\n"
        "class Volume:\n"
        "    def commit(self):\n"
        "        pass\n"
        "modal_app.cache_volume = Volume()\n"
        f"modal_app.CACHE_MOUNT = Path({str(tmp_path / 'mount')!r})\n"
        f"modal_app.CACHE_LOCAL = Path({str(local)!r})\n"
        # The boot fails after the import, at the config: the probe has seen it.
        "modal_app.PolicyServer().web.local()\n"
    )
    for module in ("torch", "praxis.serve.server"):
        seen = env_at_torch_import(str(script), {}, module=module)
        assert seen == cache_env(local)


def test_copy_new_copies_only_the_missing_cache_files(tmp_path: Path) -> None:
    """The cache volume gets the new entries; locks and partial writes stay."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "triton" / "abc").mkdir(parents=True)
    (src / "triton" / "abc" / "k.cubin").write_bytes(b"new")
    (src / "inductor").mkdir()
    (src / "inductor" / "old.py").write_text("src")
    (src / "inductor" / "x.lock").write_text("")
    # The temporary files of Inductor, Triton and of copy_new itself.
    (src / "inductor" / ".123.456.tmp").write_text("")
    (src / "triton" / "abc" / "tmp.pid_1_2").mkdir()
    (src / "triton" / "abc" / "tmp.pid_1_2" / "k.json").write_text("")
    (src / "inductor" / f"{modal_app.PARTIAL}9-z.py").write_text("")
    (dst / "inductor").mkdir(parents=True)
    (dst / "inductor" / "old.py").write_text("dst")
    assert modal_app.copy_new(src, dst) == 1
    assert (dst / "triton" / "abc" / "k.cubin").read_bytes() == b"new"
    # An entry that exists is the same entry: it is not written again.
    assert (dst / "inductor" / "old.py").read_text() == "dst"
    assert sorted(p.name for p in (dst / "inductor").iterdir()) == ["old.py"]
    assert modal_app.copy_new(src, dst) == 0
    assert modal_app.copy_new(tmp_path / "missing", dst) == 0


def test_copy_new_keeps_names_that_only_look_temporary(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "a").mkdir(parents=True)
    names = ["foo.tmpl.py", "x.tmp1", "tmp.py", "empty.bin"]
    for name in names[:-1]:
        (src / "a" / name).write_text(name)
    # An empty file is an entry too.
    (src / "a" / "empty.bin").write_bytes(b"")
    assert modal_app.copy_new(src, dst) == len(names)
    assert sorted(p.name for p in (dst / "a").iterdir()) == sorted(names)
    assert (dst / "a" / "empty.bin").read_bytes() == b""


def test_a_failed_copy_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copy cut short leaves nothing under the real name; the next copy does it."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    (src / "k").mkdir(parents=True)
    (src / "k" / "k.cubin").write_bytes(b"cubin" * 100)
    copy2 = modal_app.shutil.copy2

    def cut_short(a: Path, b: Path) -> None:
        Path(b).write_bytes(Path(a).read_bytes()[:3])
        raise OSError("no space left")

    monkeypatch.setattr(modal_app.shutil, "copy2", cut_short)
    with pytest.raises(OSError, match="no space"):
        modal_app.copy_new(src, dst)
    assert list((dst / "k").iterdir()) == []
    monkeypatch.setattr(modal_app.shutil, "copy2", copy2)
    assert modal_app.copy_new(src, dst) == 1
    assert (dst / "k" / "k.cubin").read_bytes() == b"cubin" * 100


def _triton_entry(d: Path, *, cubin: bool) -> None:
    """Write a Triton cache entry: the files and the group that lists them."""
    d.mkdir(parents=True)
    children = {f"k.{ext}": str(d / f"k.{ext}") for ext in ("json", "cubin", "ttir")}
    for name in children:
        if name != "k.cubin" or cubin:
            (d / name).write_text(name)
    (d / "__grp__k.json").write_text(json.dumps({"child_paths": children}))


def test_copy_new_skips_an_incomplete_triton_group(tmp_path: Path) -> None:
    """A group without its cubin would load as a hit and fail; it is not copied."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    _triton_entry(src / "triton" / "good", cubin=True)
    _triton_entry(src / "triton" / "bad", cubin=False)
    (src / "triton" / "broken").mkdir()
    (src / "triton" / "broken" / "__grp__x.json").write_text("{not json")
    modal_app.copy_new(src, dst)
    assert sorted(p.name for p in (dst / "triton" / "good").iterdir()) == [
        "__grp__k.json",
        "k.cubin",
        "k.json",
        "k.ttir",
    ]
    assert sorted(p.name for p in (dst / "triton" / "bad").iterdir()) == [
        "k.json",
        "k.ttir",
    ]
    assert not (dst / "triton" / "broken").exists()


def test_copy_new_writes_the_groups_last(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A copy cut short leaves files without a group, never a group without files."""
    src, dst = tmp_path / "src", tmp_path / "dst"
    for name in ("a", "b"):
        _triton_entry(src / "triton" / name, cubin=True)
    order: list[str] = []
    copy2 = modal_app.shutil.copy2

    def record(a: Path, b: Path) -> None:
        order.append(Path(a).name)
        copy2(a, b)

    monkeypatch.setattr(modal_app.shutil, "copy2", record)
    assert modal_app.copy_new(src, dst) == 8
    assert order[-2:] == ["__grp__k.json", "__grp__k.json"]
    assert not any(n.startswith("__grp__") for n in order[:-2])
