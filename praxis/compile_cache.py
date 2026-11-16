"""Keep the compiled kernels of praxis on disk across processes.

The first call of a process compiles kernels at run time: the flex attention
graphs (``torch.compile``, Inductor), the Triton kernels that Inductor
generates and the CuTe kernels of ``flash-attn-4`` (the ``flash`` attention
backend). Each library can keep its compiled kernels in a directory and load
them in the next process. The default directories of PyTorch and Triton are
under ``/tmp`` and ``~/.triton``, which a container loses at exit, and
``flash-attn-4`` keeps its kernels only in memory unless
``FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED=1``.

``PRAXIS_CACHE_DIR`` names one directory for all of them.
``configure_compile_cache`` sets these defaults under it:

| Variable | Value | Read by |
|---|---|---|
| ``TORCHINDUCTOR_CACHE_DIR`` | ``<dir>/<versions>/inductor`` | PyTorch Inductor: generated code, FX graph cache |
| ``TORCHINDUCTOR_FX_GRAPH_CACHE`` | ``1`` | PyTorch Inductor |
| ``TORCHINDUCTOR_AUTOGRAD_CACHE`` | ``1`` | PyTorch AOTAutograd |
| ``TRITON_CACHE_DIR`` | ``<dir>/<versions>/triton`` | Triton: compiled kernels (cubins) |
| ``FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED`` | ``1`` | ``flash_attn.cute.cache_utils`` |
| ``FLASH_ATTENTION_CUTE_DSL_CACHE_DIR`` | ``<dir>/<versions>/flash_attn_cute`` | ``flash_attn.cute.cache_utils`` |

``<versions>`` names the installed versions of ``torch``, ``triton``,
``flash-attn-4`` and ``nvidia-cutlass-dsl`` (``cache_version``), for example
``torch-2.9.1_triton-3.5.0_flash-attn-4-none_nvidia-cutlass-dsl-none`` (``none``:
not installed).

A variable that is already set keeps its value. ``flash-attn-4`` reads its
two variables, and PyTorch the two ``1`` flags, when they are imported;
``TORCHINDUCTOR_CACHE_DIR`` and ``TRITON_CACHE_DIR`` are read later, at the
first compile. The entry points call ``configure_compile_cache`` before
``import torch``, so all of them apply.

The caches hold the compiled code of the same kernels; they do not change the
code of a kernel or the order of its operations. A kernel that Inductor
autotunes at run time is the exception to "same as a run without the cache":
Inductor stores the launch setting that the run picked (``<kernel>.best_config``
under ``TORCHINDUCTOR_CACHE_DIR``), and a warm process reuses the choice of the
process that wrote the cache instead of benchmarking again. A run without the
cache is not reproducible across processes on such a path either, since each
process benchmarks again; ``praxis/layers.py`` fixes the setting by a rule
where the choice matters (the dynamic flex decoding path).

Each library keys its entries on its own inputs: Inductor hashes the graph, the
config and the PyTorch version, Triton the kernel source and the compiler, and
``flash-attn-4`` puts its entries under a fingerprint of its CuTe sources, the
Python version and the ``cutlass`` and ``tvm_ffi`` versions. That fingerprint
does not hold the PyTorch version, but a ``flash`` kernel of flex attention
holds code that PyTorch generates. The ``<versions>`` directory keeps the
entries of each set of installed versions apart, so an upgrade of one of these
packages starts in a new, empty directory and compiles again instead of loading
a kernel of the old versions.

This module imports no torch, so it can run first.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import MutableMapping
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

log = logging.getLogger(__name__)

# The environment variable with the cache directory. Unset: no change.
CACHE_DIR_ENV = "PRAXIS_CACHE_DIR"

# Modules that read the variables at import. A call after one of them is
# imported is too late for that library.
_READERS = ("torch", "triton", "flash_attn")

# The distributions whose versions name the cache subdirectory.
VERSIONED = ("torch", "triton", "flash-attn-4", "nvidia-cutlass-dsl")


def cache_version() -> str:
    """Return the name of the subdirectory for the installed versions.

    The name holds the version of each distribution of ``VERSIONED``, or
    ``none`` for one that is not installed. It reads the package metadata and
    imports none of the packages.
    """
    parts = []
    for name in VERSIONED:
        try:
            ver = version(name)
        except PackageNotFoundError:
            ver = "none"
        parts.append(f"{name}-{re.sub(r'[^A-Za-z0-9.+-]', '_', ver)}")
    return "_".join(parts)


def cache_env(root: str | Path) -> dict[str, str]:
    """Return the cache variables for the cache directory ``root``."""
    base = Path(root).expanduser().absolute() / cache_version()
    return {
        "TORCHINDUCTOR_CACHE_DIR": str(base / "inductor"),
        "TORCHINDUCTOR_FX_GRAPH_CACHE": "1",
        "TORCHINDUCTOR_AUTOGRAD_CACHE": "1",
        "TRITON_CACHE_DIR": str(base / "triton"),
        "FLASH_ATTENTION_CUTE_DSL_CACHE_ENABLED": "1",
        "FLASH_ATTENTION_CUTE_DSL_CACHE_DIR": str(base / "flash_attn_cute"),
    }


def configure_compile_cache(
    root: str | Path | None = None,
    environ: MutableMapping[str, str] | None = None,
) -> dict[str, str]:
    """Set the defaults of the compile cache variables under one directory.

    ``root`` is the cache directory; ``None`` reads ``PRAXIS_CACHE_DIR``.
    Without a directory the function changes nothing and returns ``{}``.
    A variable that is already set keeps its value. The function creates the
    directory and returns the value of each cache variable after the call.
    Call it before ``import torch``: it logs a warning when torch, Triton or
    ``flash_attn`` is already imported.
    """
    env = os.environ if environ is None else environ
    if root is None:
        root = env.get(CACHE_DIR_ENV) or None
    if root is None:
        return {}
    defaults = cache_env(root)
    Path(root).expanduser().mkdir(parents=True, exist_ok=True)
    for key, value in defaults.items():
        env.setdefault(key, value)
    late = [m for m in _READERS if m in sys.modules]
    if late:
        log.warning(
            "compile cache: %s already imported; set %s before the import",
            ", ".join(late),
            CACHE_DIR_ENV,
        )
    return {key: env[key] for key in defaults}
