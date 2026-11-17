"""The CUDA graphs of ``Autoencoder.encode`` (VAE_ENCODE_GRAPH).

The CPU tests replace the CUDA pieces of ``EncodeGraphs`` (the side stream,
the capture) with a fake graph that reruns the captured function into its
output buffer, the contract of a replay. They check the bookkeeping: one
graph per key, the static input, outputs that later calls do not change, the
cap, the eager fallback of a failed capture or a capture that changes
bits, and the scope of :meth:`Autoencoder._encode_graph_eligible` (each
condition alone turns the graphs off; training and gradient encodes never
reach them). The fused encoder runs on the CPU with torch stand-ins for its
Triton kernels (as in ``test_vae_stream_encode.py``).

The CUDA tests compare graph replays with eager encodes bit for bit on the
GPU: a small encoder, and the full encoder at the serving canvas.
"""

from __future__ import annotations

from contextlib import contextmanager, nullcontext
from threading import Thread

import pytest
import torch
from test_vae_stream_encode import _add_rmsnorm_silu, _avg_down, _rmsnorm_silu
from torch import Tensor, nn

from praxis import vae
from praxis.kernels import vae_encoder_triton as kernels


@pytest.fixture
def cpu_kernels(monkeypatch: pytest.MonkeyPatch) -> None:
    """Torch stand-ins for the Triton kernels; the float path on the CPU."""
    monkeypatch.setattr(kernels, "rmsnorm_silu", _rmsnorm_silu)
    monkeypatch.setattr(kernels, "add_rmsnorm_silu", _add_rmsnorm_silu)
    monkeypatch.setattr(kernels, "avg_down", _avg_down)
    monkeypatch.setattr(vae, "VAE_HOIST_CASTS", False)


class _FakeGraph:
    """Replay reruns ``fn`` on the static input into the captured output."""

    def __init__(self, fn, static: Tensor, output: Tensor, perturb: bool) -> None:
        self.fn, self.static, self.output, self.perturb = fn, static, output, perturb

    def replay(self) -> None:
        y = self.fn(self.static)
        if self.perturb:
            y = y.clone()
            y.view(-1)[0] += 1
        self.output.copy_(y)


class _CpuGraphs(vae.EncodeGraphs):
    """EncodeGraphs with the CUDA stream and capture replaced."""

    def __init__(self, *args, fail: bool = False, perturb: bool = False, **kw):
        super().__init__(*args, **kw)
        self.fail, self.perturb = fail, perturb

    def _side(self):
        return nullcontext()

    def _sync(self) -> None:
        pass

    def _capture(self, fn, static):
        if self.fail:
            raise RuntimeError("operation not permitted when stream is capturing")
        output = torch.empty_like(fn(static))  # capture does not run kernels
        return _FakeGraph(fn, static, output, self.perturb), output


def _counting(fn):
    calls = []

    def wrapped(x: Tensor) -> Tensor:
        calls.append(tuple(x.shape))
        return fn(x)

    return wrapped, calls


def _toy(x: Tensor) -> Tensor:
    return (x.float() * 0.5 + 1).sum(dim=1, keepdim=True)


def _graphs(max_graphs: int = 4, **kw) -> _CpuGraphs:
    return _CpuGraphs(None, torch.device("cpu"), max_graphs, **kw)


def test_same_bits_compares_bytes_and_layout() -> None:
    a = torch.tensor([0.0, float("nan"), 1.0])
    assert vae._same_bits(a, a.clone())  # NaN with the same bits is equal
    assert not vae._same_bits(a, torch.tensor([-0.0, float("nan"), 1.0]))
    assert not vae._same_bits(a, a.to(torch.float64))
    b = torch.zeros(2, 3)
    assert not vae._same_bits(b, b.t().contiguous().t())


def test_one_graph_per_shape_and_independent_outputs() -> None:
    graphs = _graphs()
    fn, calls = _counting(_toy)
    kept = []
    for t in (1, 5, 9, 9, 5, 1):
        # A time slice of a larger tensor: the static input is a contiguous copy.
        frames = torch.randint(0, 256, (1, 3, t + 2, 4, 4), dtype=torch.uint8)[
            :, :, 1:-1
        ]
        out = graphs(fn, frames)
        assert vae._same_bits(out, _toy(frames))
        for old, snapshot in kept:
            assert torch.equal(old, snapshot)
        kept.append((out, out.clone()))
    assert graphs.stats == {"captures": 3, "replays": 6, "eager": 0, "rejected": 0}
    assert graphs.n_graphs == 3
    # Each capture runs two eager warmups and the capture; replays rerun fn
    # only through the fake graph.
    assert calls.count((1, 3, 9, 4, 4)) == 3 + 1 + 2


def test_returned_latent_is_not_the_graph_output() -> None:
    graphs = _graphs()
    frames = torch.zeros(1, 3, 1, 4, 4, dtype=torch.uint8)
    out = graphs(_toy, frames)
    static, _, output = graphs.variants[graphs.key(frames)]
    assert out.data_ptr() != output.data_ptr()
    assert static.is_contiguous() and static.data_ptr() != frames.data_ptr()


def test_shapes_past_the_cap_run_eagerly() -> None:
    graphs = _graphs(max_graphs=2)
    for t in (1, 5, 9, 13):
        frames = torch.randint(0, 256, (1, 3, t, 4, 4), dtype=torch.uint8)
        assert vae._same_bits(graphs(_toy, frames), _toy(frames))
    assert graphs.n_graphs == 2
    assert graphs.stats["eager"] == 2 and graphs.stats["captures"] == 2


def test_failed_capture_keeps_the_shape_eager() -> None:
    graphs = _graphs(fail=True)
    fn, calls = _counting(_toy)
    frames = torch.randint(0, 256, (1, 3, 5, 4, 4), dtype=torch.uint8)
    for _ in range(3):
        assert vae._same_bits(graphs(fn, frames), _toy(frames))
    assert graphs.stats == {"captures": 0, "replays": 0, "eager": 3, "rejected": 1}
    assert graphs.variants[graphs.key(frames)] is None
    assert graphs.n_graphs == 0
    # Two warmups, then one eager call per call: no second capture attempt.
    assert len(calls) == 2 + 3


def test_a_graph_that_changes_bits_is_rejected() -> None:
    graphs = _graphs(perturb=True)
    frames = torch.randint(0, 256, (1, 3, 5, 4, 4), dtype=torch.uint8)
    for _ in range(2):
        assert vae._same_bits(graphs(_toy, frames), _toy(frames))
    assert graphs.stats["rejected"] == 1 and graphs.stats["replays"] == 0


def test_module_flags_and_modes_are_part_of_the_key(monkeypatch) -> None:
    frames = torch.zeros(1, 3, 5, 4, 4, dtype=torch.uint8)
    base = vae.EncodeGraphs.key(frames)
    for flag in ("VAE_HOIST_CASTS", "VAE_FUSED_ADD_NORM", "ENCODER_CACHE_VIEWS"):
        with monkeypatch.context() as m:
            m.setattr(vae, flag, not getattr(vae, flag))
            assert vae.EncodeGraphs.key(frames) != base
    with torch.backends.cudnn.flags(enabled=True, benchmark=True):
        assert vae.EncodeGraphs.key(frames) != base
    assert vae.EncodeGraphs.key(frames.float()) != base
    assert vae.EncodeGraphs.key(frames[:, :, :1]) != base


def test_a_second_thread_during_a_call_runs_eagerly() -> None:
    graphs = _graphs()
    frames = torch.zeros(1, 3, 5, 4, 4, dtype=torch.uint8)
    seen = {}

    def other() -> None:
        seen["out"] = graphs(_toy, frames)

    def fn(x: Tensor) -> Tensor:
        if "out" not in seen:
            t = Thread(target=other)
            t.start()
            t.join()
        return _toy(x)

    graphs(fn, frames)
    assert vae._same_bits(seen["out"], _toy(frames))
    assert graphs.stats["eager"] == 1


def _small_autoencoder() -> vae.Autoencoder:
    """An Autoencoder with a small encoder and no decoder (encode only)."""
    torch.manual_seed(0)
    model = vae.Autoencoder.__new__(vae.Autoencoder)
    nn.Module.__init__(model)
    model.norm_block_c = {8: 64, 16: 64, 32: 64}
    model.encoder = vae.Encoder(dim=8)
    model.conv1 = vae.CausalConv3d(2 * vae.Z_DIM, 2 * vae.Z_DIM, 1)
    for p in model.parameters():
        p.data.normal_(0, 0.1)
    model.register_buffer("latent_mean", torch.randn(vae.Z_DIM), persistent=False)
    model.register_buffer(
        "latent_inv_std", torch.rand(vae.Z_DIM) + 0.5, persistent=False
    )
    model._fused = None
    model._encode_graphs = None
    model._decoder_controller = None
    return model.eval().requires_grad_(False)


def test_encode_is_eager_on_cpu_and_with_the_flag_off(monkeypatch) -> None:
    model = _small_autoencoder().enable_encode_graphs()
    frames = torch.zeros(1, 3, 5, 64, 64, dtype=torch.uint8)
    with torch.inference_mode():
        assert not model._encode_graph_eligible(frames)
    monkeypatch.setattr(vae, "VAE_ENCODE_GRAPH", False)
    with torch.inference_mode():
        assert not model._encode_graph_eligible(frames)
    assert model.encode_graph_stats() is None


class _CudaLike(Tensor):
    """A CPU tensor that reports ``is_cuda``: the device test of the eligibility."""

    @property
    def is_cuda(self) -> bool:  # ty: ignore[invalid-method-override]
        return True


def _cuda_like(t: Tensor) -> Tensor:
    return t.as_subclass(_CudaLike)


@pytest.fixture
def no_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    """No CUDA stream is capturing (the CPU build cannot ask)."""
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)


def test_encode_graphs_are_opt_in() -> None:
    model = _small_autoencoder()
    assert not model._encode_graphs_enabled
    assert model.enable_encode_graphs() is model
    assert model._encode_graphs_enabled
    assert not _small_autoencoder()._encode_graphs_enabled


def test_each_eligibility_condition_turns_the_graphs_off(
    monkeypatch, no_capture
) -> None:
    """Every condition of the serving-path scope, one at a time."""
    model = _small_autoencoder().enable_encode_graphs()
    frames = _cuda_like(torch.zeros(1, 3, 5, 64, 64, dtype=torch.uint8))
    with torch.inference_mode():
        assert model._encode_graph_eligible(frames)
    # Outside inference mode (fine-tuning, the serve precompute): eager.
    with torch.no_grad():
        assert not model._encode_graph_eligible(frames)
    assert not model._encode_graph_eligible(frames)
    with torch.inference_mode():
        # Not opted in (an evaluation script's own autoencoder).
        assert not _small_autoencoder()._encode_graph_eligible(frames)
        with monkeypatch.context() as m:
            m.setattr(vae, "VAE_ENCODE_GRAPH", False)
            assert not model._encode_graph_eligible(frames)
        assert not model._encode_graph_eligible(frames.as_subclass(Tensor))
        assert not model._encode_graph_eligible(frames[0])
        model.train()
        assert not model._encode_graph_eligible(frames)
        model.eval()
        with torch.enable_grad():
            assert not model._encode_graph_eligible(frames)
        with monkeypatch.context() as m:
            m.setattr(torch.compiler, "is_compiling", lambda: True)
            assert not model._encode_graph_eligible(frames)
        with monkeypatch.context() as m:
            m.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
            assert not model._encode_graph_eligible(frames)
        assert model._encode_graph_eligible(frames)
    wants_grad = _cuda_like(torch.zeros(1, 3, 5, 64, 64, requires_grad=True))
    with torch.inference_mode():
        assert not model._encode_graph_eligible(wants_grad)
        assert model._encode_graph_eligible(wants_grad.detach())


def test_training_and_grad_encodes_never_replay(
    monkeypatch, cpu_kernels, no_capture
) -> None:
    """Fine-tuning encodes (training mode, gradients on, frames that require
    grad) skip the graphs and keep the eager bits; the serving encode of the
    same frames replays. ``encode`` runs under ``no_grad`` (as on main), so
    no latent carries a grad_fn either way."""
    model = _small_autoencoder().enable_encode_graphs()
    monkeypatch.setattr(vae, "EncodeGraphs", _CpuGraphs)
    raw = torch.randint(0, 256, (1, 3, 5, 64, 64)).float()
    want = model._encode_fused(raw)
    frames = _cuda_like(raw)
    model.train().requires_grad_(True)
    with torch.enable_grad():
        out = model.encode(frames)  # fine-tuning: outside inference mode
    assert vae._same_bits(out, want) and out.grad_fn is None
    with torch.inference_mode():
        out = model.encode(frames)  # training mode alone
    assert vae._same_bits(out, want)
    model.eval()
    wants_grad = _cuda_like(raw.clone().requires_grad_(True))
    with torch.inference_mode():
        out = model.encode(wants_grad)  # frames that require grad alone
    assert vae._same_bits(out, want) and out.grad_fn is None
    with torch.no_grad():
        out = model.encode(frames)  # outside inference mode alone
    assert vae._same_bits(out, want)
    assert model._encode_graphs is None
    with torch.inference_mode():
        for _ in range(2):
            assert vae._same_bits(model.encode(frames), want)
    assert model.encode_graph_stats() == {
        "captures": 1,
        "replays": 2,
        "eager": 0,
        "rejected": 0,
        "graphs": 1,
    }
    model.train()
    with torch.enable_grad():
        out = model.encode(frames)
    assert vae._same_bits(out, want)
    assert model.encode_graph_stats()["replays"] == 2


def test_capture_checks_this_thread_only(monkeypatch) -> None:
    """The capture uses capture_error_mode="thread_local" on the side stream."""
    seen = {}

    class _Graph:
        pass

    @contextmanager
    def graph(cuda_graph, **kw):
        seen["graph"], seen["kw"] = cuda_graph, kw
        yield

    monkeypatch.setattr(torch.cuda, "CUDAGraph", _Graph)
    monkeypatch.setattr(torch.cuda, "graph", graph)
    graphs = vae.EncodeGraphs(None, torch.device("cpu"), 1)
    graphs._stream, graphs._pool = "stream", "pool"
    frames = torch.zeros(1, 3, 1, 4, 4, dtype=torch.uint8)
    g, out = graphs._capture(_toy, frames)
    assert g is seen["graph"] and vae._same_bits(out, _toy(frames))
    assert seen["kw"] == {
        "pool": "pool",
        "stream": "stream",
        "capture_error_mode": "thread_local",
    }


def test_encode_through_the_graph_wrapper_keeps_the_bits(
    monkeypatch, cpu_kernels
) -> None:
    """The real fused encode (CPU stand-in kernels) under the graph bookkeeping."""
    model = _small_autoencoder()
    monkeypatch.setattr(vae, "EncodeGraphs", _CpuGraphs)
    monkeypatch.setattr(
        vae.Autoencoder, "_encode_graph_eligible", lambda self, frames: True
    )
    kept = []
    for t in (1, 5, 9, 22, 9, 1):
        frames = torch.randint(0, 256, (1, 3, t + 1, 64, 64), dtype=torch.uint8)[
            :, :, 1:
        ]
        want = model._encode_fused(frames)
        got = model.encode(frames)
        assert vae._same_bits(got, want), t
        for old, snapshot in kept:
            assert torch.equal(old, snapshot)
        kept.append((got, got.clone()))
    assert model.encode_graph_stats() == {
        "captures": 4,
        "replays": 6,
        "eager": 0,
        "rejected": 0,
        "graphs": 4,
    }
    # A new fused encoder (weights moved) gets new graphs.
    graphs = model._encode_graphs
    model._fused = None
    model.encode(frames)
    assert model._encode_graphs is not graphs


def _cuda_autoencoder(dim: int, block_c: dict[int, int]) -> vae.Autoencoder:
    torch.manual_seed(0)
    model = vae.Autoencoder.__new__(vae.Autoencoder)
    nn.Module.__init__(model)
    model.norm_block_c = dict(block_c)
    with torch.device("cuda"):
        model.encoder = vae.Encoder(dim=dim).to(torch.bfloat16)
        model.conv1 = vae.CausalConv3d(2 * vae.Z_DIM, 2 * vae.Z_DIM, 1).to(
            torch.bfloat16
        )
        model.register_buffer("latent_mean", torch.tensor(vae.LATENT_MEAN))
        model.register_buffer("latent_inv_std", 1 / torch.tensor(vae.LATENT_STD))
    model._fused = None
    model._encode_graphs = None
    model._decoder_controller = None
    return model.eval().requires_grad_(False).enable_encode_graphs()


def _cuda_parity(model: vae.Autoencoder, h: int, w: int, lengths, monkeypatch):
    kept = []
    for t in lengths:
        # A time slice of a larger tensor checks the input strides.
        frames = torch.randint(
            0, 256, (1, 3, t + 2, h, w), dtype=torch.uint8, device="cuda"
        )[:, :, 1:-1]
        with monkeypatch.context() as m:
            m.setattr(vae, "VAE_ENCODE_GRAPH", False)
            want = model.encode(frames)
        got = model.encode(frames)
        assert vae._same_bits(got, want), f"T={t}"
        for old, snapshot in kept:
            assert torch.equal(old, snapshot)
        kept.append((got, got.clone()))
    torch.cuda.synchronize()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_cuda_graph_matches_eager_small(monkeypatch) -> None:
    model = _cuda_autoencoder(8, {8: 64, 16: 64, 32: 64})
    with torch.inference_mode():
        _cuda_parity(model, 64, 64, (1, 5, 9, 21, 9, 5, 1), monkeypatch)
    stats = model.encode_graph_stats()
    assert stats["captures"] == 4 and stats["rejected"] == 0
    assert stats["replays"] == 7 and stats["eager"] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")
def test_cuda_graph_matches_eager_serving_canvas(monkeypatch) -> None:
    """The full encoder at the DROID canvas and the observed-prefix lengths."""
    model = _cuda_autoencoder(160, vae.K.DEFAULT_NORM_BLOCK_C)
    with torch.inference_mode():
        _cuda_parity(model, 256, 832, (1, 25, 49, 25, 1), monkeypatch)
    # Outside inference mode (fine-tuning, the precompute) the encode is eager.
    with torch.no_grad():
        frames = torch.zeros(1, 3, 25, 256, 832, dtype=torch.uint8, device="cuda")
        model.encode(frames)
    stats = model.encode_graph_stats()
    assert stats["rejected"] == 0 and stats["captures"] == 3
    assert stats["replays"] == 5 and stats["eager"] == 0
