"""Modal app of the praxis policy server.

The app serves the FastAPI app of ``python -m praxis.serve`` on one Modal GPU,
with the single-arm checkpoint. The Modal edge requires a proxy auth
token on each connection. A client sends the token in the headers
``Modal-Key`` and ``Modal-Secret``.

Deploy with ``modal>=1.2`` and your own Modal credentials (``modal token
set``), from the root of the repository, after the weights are in their
local directories ("Get the weights" in the top README: ``hf download
runwayml/praxis-1-droid --local-dir checkpoints/praxis-1-droid``):

    modal volume create praxis-weights
    modal volume put praxis-weights checkpoints/praxis-1-droid /checkpoints/praxis-1-droid
    modal volume put praxis-weights weights /weights
    modal deploy praxis/serve/modal_app.py

The serve config of the single-arm checkpoint is ``configs/serve_droid.yaml``
(the default). The bimanual checkpoint uses ``configs/serve_yam.yaml``
(``PRAXIS_CONFIG=configs/serve_yam.yaml``) and an so101 checkpoint
``configs/serve_so101.yaml`` (its ``routed_precision: nvfp4`` needs the
``nvfp4`` extra and an SM120 or SM100 GPU; ``bf16`` runs on the default
H100). The checkpoints use the ``flash`` attention backend of the trunk; the
image installs ``flash-attn-4`` for it.

Modal builds the image from ``docker/Dockerfile``. The volume holds the
``checkpoints/`` and ``weights/`` directories of the repository layout. The
model paths of the config and ``PRAXIS_CHECKPOINT`` are relative to the root
of the volume. A leading ``/`` also means the root of the volume.

These environment variables of ``modal deploy`` change the deployment:

* ``PRAXIS_CONFIG``: the serve config, a path in the repository. The default
  is ``configs/serve_droid.yaml``.
* ``PRAXIS_CHECKPOINT``: the checkpoint directory in the volume. The default
  is ``model.checkpoint`` of the config, for example a fine-tuned copy of
  the checkpoint. A checkpoint of another rig than the config fails the boot.
* ``PRAXIS_MODAL_APP_NAME``: the app name. The default is ``praxis-serve``.
  Deploy each checkpoint as a separate app.
* ``PRAXIS_MODAL_GPU``: the GPU type. The default is ``H100``. Modal can run
  an ``H100`` request on an ``H200``. ``H100!`` prevents this.
* ``PRAXIS_MODAL_VOLUME``: the volume with the weights. The default is
  ``praxis-weights``.
* ``PRAXIS_MODAL_SCALEDOWN_S``: the idle time in seconds before Modal stops
  the container. The default is 1800.
* ``PRAXIS_MODAL_CACHE_VOLUME``: a volume for the compiled kernels
  (:mod:`praxis.compile_cache`), for example ``praxis-compile-cache``. Modal
  creates it if it is missing. Unset (the default): each container compiles
  the attention kernels again in its warmup. Set: the boot copies the volume
  to a local cache directory, sets the cache variables of
  :mod:`praxis.compile_cache` under it before torch is imported (the app does
  not read ``PRAXIS_CACHE_DIR``), and after the warmup copies the new files back and commits the
  volume. The next container then loads the kernels. The local copy keeps the
  file locks of ``flash-attn-4`` off the network file system of the volume.
  The copies are best effort: an error logs a warning and the boot goes on,
  with an empty cache when the copy from the volume fails.

The app copies all ``PRAXIS_*`` variables into the container, for example
``PRAXIS_SAMPLE_SEED`` and ``PRAXIS_SKIP_WARMUP`` of the server.

The app has no ``@modal.concurrent`` setting. Modal then gives a container
one connection at a time.

The Modal edge accepts WebSocket messages of up to 2 MiB. With the JPEG
quality 90 of the Positronic client, the largest of 128 recorded single-arm
requests was 1.74 MiB (see ``praxis/serve/README.md``).
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import modal

from praxis.serve.params import git_commit

if TYPE_CHECKING:
    from fastapi import FastAPI

log = logging.getLogger(__name__)

# The repository root on the deploy machine.
REPO = Path(__file__).resolve().parent.parent.parent
# The repository in the image (docker/Dockerfile) and the mount point of the
# weights volume.
IMAGE_REPO = Path("/opt/praxis")
WEIGHTS = Path("/data")

# The environment of the container: the PRAXIS_* variables of the deploy. The
# image holds no git history, so the deploy machine sets the commit.
CONTAINER_ENV: dict[str, str | None] = {
    k: v for k, v in os.environ.items() if k.startswith("PRAXIS_")
}
if modal.is_local():
    CONTAINER_ENV.setdefault("PRAXIS_SERVED_COMMIT", git_commit(REPO))

image = modal.Image.from_dockerfile(REPO / "docker" / "Dockerfile", context_dir=REPO)
volume = modal.Volume.from_name(
    os.environ.get("PRAXIS_MODAL_VOLUME", "praxis-weights")
).read_only()
volumes: dict[str, modal.Volume] = {str(WEIGHTS): volume}
# The compile cache volume (optional) and the local copy of it.
CACHE_VOLUME_NAME = os.environ.get("PRAXIS_MODAL_CACHE_VOLUME")
CACHE_MOUNT = Path("/compile-cache")
CACHE_LOCAL = Path("/tmp/praxis-compile-cache")
cache_volume = None
if CACHE_VOLUME_NAME:
    cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
    volumes[str(CACHE_MOUNT)] = cache_volume
app = modal.App(os.environ.get("PRAXIS_MODAL_APP_NAME", "praxis-serve"), image=image)


# The name prefix of a file that copy_new is writing.
PARTIAL = ".praxis-partial-"


def _skipped(rel: Path) -> bool:
    """Return whether ``rel`` (a path in a cache) is a lock or temporary file."""
    name = rel.name
    return (
        rel.suffix in (".lock", ".tmp")
        or name.startswith(PARTIAL)
        # Triton writes each file in a tmp.pid_<pid>_<id> directory first.
        or any(part.startswith("tmp.pid_") for part in rel.parts[:-1])
    )


def _complete_group(path: Path) -> bool:
    """Return whether the files of a Triton group ``__grp__*.json`` are all there.

    A group lists its files (the kernel binary, its metadata, the IRs) by
    absolute path, in the directory of the group. A group whose files are
    missing is a miss for some Triton versions, and for others a hit that
    fails in each later process.
    """
    try:
        children = json.loads(path.read_text())["child_paths"]
        return all((path.parent / Path(p).name).is_file() for p in children.values())
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return False


def copy_new(src: Path, dst: Path) -> int:
    """Copy the files of ``src`` that ``dst`` does not hold; return the count.

    The cache entries are written once under content keys, so a file that
    exists is the same file. Lock and temporary files are skipped, and so are
    the Triton groups (``__grp__*``) whose files are not all in ``src``. Each
    file is written under a temporary name and renamed, so a copy cut short
    leaves no partial file under the real name, and the groups are copied
    last, so a copy cut short leaves no group without its files.
    """
    files = [
        path
        for path in sorted(src.rglob("*"))
        if path.is_file() and not _skipped(path.relative_to(src))
    ]
    files.sort(key=lambda path: path.name.startswith("__grp__"))
    count = 0
    for path in files:
        target = dst / path.relative_to(src)
        if target.exists():
            continue
        if path.name.startswith("__grp__") and not _complete_group(path):
            log.warning("compile cache: skipped the incomplete group %s", path)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_name(f"{PARTIAL}{os.getpid()}-{target.name}")
        try:
            shutil.copy2(path, partial)
            os.replace(partial, target)
        finally:
            partial.unlink(missing_ok=True)
        count += 1
    return count


def load_cache() -> int:
    """Copy the cache volume to ``CACHE_LOCAL`` and use it; return the file count.

    Best effort: when the copy fails, the boot starts with an empty local
    directory, and when the cache cannot be set up, without a cache.
    """
    from praxis.compile_cache import configure_compile_cache

    loaded = 0
    try:
        loaded = copy_new(CACHE_MOUNT, CACHE_LOCAL)
    except Exception:
        log.warning(
            "compile cache: the copy from the volume failed; starting empty",
            exc_info=True,
        )
        shutil.rmtree(CACHE_LOCAL, ignore_errors=True)
    try:
        configure_compile_cache(CACHE_LOCAL)
    except Exception:
        log.warning("compile cache: not set up", exc_info=True)
    return loaded


def save_cache(cache: modal.Volume) -> int:
    """Copy the new files of ``CACHE_LOCAL`` to the volume and commit it.

    Returns the file count. Best effort: an error logs a warning.
    """
    try:
        saved = copy_new(CACHE_LOCAL, CACHE_MOUNT)
        if saved:
            cache.commit()
    except Exception:
        log.warning("compile cache: the copy to the volume failed", exc_info=True)
        return 0
    log.info("compile cache: %d new files to the volume", saved)
    return saved


@app.cls(
    gpu=os.environ.get("PRAXIS_MODAL_GPU", "H100"),
    # One WebSocket connection is one input, so a session can last 1 hour.
    timeout=3600,
    # The boot (weights from the volume, warmup) has a separate limit.
    startup_timeout=7200,
    scaledown_window=int(os.environ.get("PRAXIS_MODAL_SCALEDOWN_S", "1800")),
    # A client tries to connect again during a cold start. Without this
    # limit, Modal starts a new GPU container for each try.
    max_containers=1,
    volumes=volumes,
    env=CONTAINER_ENV,
)
class PolicyServer:
    @modal.enter()
    def boot(self) -> None:
        """Load the model once and run the warmup."""
        # Before the server imports torch.
        loaded = load_cache() if cache_volume is not None else None
        from praxis.serve.server import build_server, load_config

        logging.basicConfig(level=logging.INFO, force=True)
        if loaded is not None:
            log.info("compile cache: %d files from the volume", loaded)
        config = os.environ.get("PRAXIS_CONFIG", "configs/serve_droid.yaml")
        cfg = load_config(IMAGE_REPO / config)
        model = cfg["model"]
        model["checkpoint"] = os.environ.get("PRAXIS_CHECKPOINT") or model["checkpoint"]
        for key in ("checkpoint", "vae_weights", "text_encoder"):
            # A path in the volume: a leading "/" is the root of the volume.
            model[key] = str(WEIGHTS / str(model[key]).lstrip("/"))
        self.server = build_server(cfg)
        if cache_volume is not None:
            save_cache(cache_volume)

    @modal.asgi_app(requires_proxy_auth=True)
    def web(self) -> FastAPI:
        return self.server.app
