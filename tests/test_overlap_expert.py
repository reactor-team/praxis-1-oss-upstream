"""The action expert on a side CUDA stream beside the decoder.

``overlap_expert`` queues the decoder on the current stream and the action
expert on a side stream. The kernels and their inputs stay the same, so a
call keeps the bits of the same call with the overlap off, guided or not.
The CUDA test runs the real streams. The CPU test swaps in no-op streams,
so the split passes run on the CPU too.
"""

from __future__ import annotations

import contextlib

import pytest
import torch
from tiny_model import TINY_HW, tiny_policy
from torch.nn.attention.flex_attention import flex_attention

import praxis.layers
import praxis.model
from praxis.model import PraxisPolicy, SamplerSettings

C, LAT, MD = 192, 6, torch.bfloat16


def _inputs(depth: int) -> dict:
    h, w = TINY_HW
    g = torch.Generator().manual_seed(depth)
    return {
        "text": torch.randn(1, 5, 32, generator=g).to(MD),
        "text_uncond": torch.randn(1, 1, 32, generator=g).to(MD),
        "anchor": torch.randn(1, C, 1, h, w, generator=g).to(MD),
        "cleans": [
            torch.randn(1, C, LAT, h, w, generator=g).to(MD) for _ in range(depth)
        ],
        "proprio_all": torch.randn(1, depth + 1, 10, generator=g),
    }


CASES = [
    (1.0, 1.0, False),  # unguided: one branch, decoder beside its expert
    (1.0, 1.0, True),  # the last step runs no decoder
    (1.0, 4.0, False),  # video guidance (the served DROID settings)
    (3.0, 2.0, False),  # both guidance scales
]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA streams test")
@pytest.mark.parametrize("depth", [0, 2])
@pytest.mark.parametrize(("cfg", "cfg_v", "skip_last_video"), CASES)
def test_overlap_expert_keeps_the_bits(
    monkeypatch: pytest.MonkeyPatch,
    depth: int,
    cfg: float,
    cfg_v: float,
    skip_last_video: bool,
) -> None:
    # The compiled flex kernel finds no valid config for the tiny depth-0
    # window on CUDA. The eager kernel keeps the comparison on the streams.
    monkeypatch.setattr(praxis.layers, "_compiled_flex_attention", flex_attention)
    policy = tiny_policy().cuda()
    settings = SamplerSettings(num_steps=3, cfg_scale=cfg, cfg_video_scale=cfg_v)
    inputs = _inputs(depth)
    whole = policy.backbone.forward_chunk_causal
    whole_calls: list[bool] = []

    def counted(*args, **kwargs):
        whole_calls.append(kwargs.get("decode", True))
        return whole(*args, **kwargs)

    monkeypatch.setattr(policy.backbone, "forward_chunk_causal", counted)
    out = {}
    for overlap in (False, True):
        whole_calls.clear()
        out[overlap] = policy.sample_window_joint(
            seed=7,
            settings=settings,
            skip_last_video=skip_last_video,
            overlap_expert=overlap,
            **inputs,
        )
        torch.cuda.synchronize()
        if overlap:
            # Every pass that decodes is split around the side stream; only
            # the encoder-only passes of a skipped last video stay whole.
            assert not any(whole_calls)
        else:
            assert any(whole_calls)
    (v0, a0), (v1, a1) = out[False], out[True]
    assert (v0 is None) == (v1 is None) == skip_last_video
    if v0 is not None:
        assert torch.equal(v0, v1)
    assert torch.equal(a0, a1)


class _CpuAsCuda(str):
    """The CPU device, but ``type`` says CUDA: the sampler takes the stream
    path. Torch reads the string, so tensors stay on the CPU."""

    type = "cuda"


class _NoOpStream:
    def __init__(self, name: str, log: list[tuple]) -> None:
        self.name, self.log = name, log

    def wait_stream(self, other: _NoOpStream) -> None:
        self.log.append(("wait", self.name, other.name))


@pytest.mark.parametrize("depth", [0, 2])
@pytest.mark.parametrize(("cfg", "cfg_v", "skip_last_video"), CASES)
def test_split_passes_keep_the_bits_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
    depth: int,
    cfg: float,
    cfg_v: float,
    skip_last_video: bool,
) -> None:
    """The overlap path on the CPU with no-op streams: the split passes (the
    encoder, the decoder, the expert in the side stream's context) keep the
    bits of the whole passes, and every step that decodes takes them."""
    # The compiled CPU flex kernel fails to build for some masks.
    monkeypatch.setattr(praxis.layers, "_compiled_flex_attention", flex_attention)
    policy = tiny_policy()
    settings = SamplerSettings(num_steps=3, cfg_scale=cfg, cfg_video_scale=cfg_v)
    inputs = _inputs(depth)
    whole = policy.backbone.forward_chunk_causal
    whole_calls: list[bool] = []

    def counted(*args, **kwargs):
        whole_calls.append(kwargs.get("decode", True))
        return whole(*args, **kwargs)

    monkeypatch.setattr(policy.backbone, "forward_chunk_causal", counted)

    def sample(overlap: bool) -> tuple[torch.Tensor | None, torch.Tensor]:
        return policy.sample_window_joint(
            seed=7,
            settings=settings,
            skip_last_video=skip_last_video,
            overlap_expert=overlap,
            **inputs,
        )

    # The serial passes on the plain CPU device.
    v0, a0 = sample(False)
    assert any(whole_calls)

    log: list[tuple] = []
    current = _NoOpStream("main", log)
    entered: list[str] = []

    def stream(device: object = None) -> _NoOpStream:
        return _NoOpStream("side", log)

    @contextlib.contextmanager
    def in_stream(s: _NoOpStream):
        entered.append(s.name)
        yield

    def record_stream(t: torch.Tensor, s: _NoOpStream) -> None:
        log.append(("record", s.name))

    cpu_as_cuda = _CpuAsCuda("cpu")
    monkeypatch.setattr(PraxisPolicy, "device", property(lambda self: cpu_as_cuda))
    # Pinned memory needs CUDA. The copy to the CPU is the plain ``.to``.
    monkeypatch.setattr(praxis.model, "to_device_async", lambda t, d: t.to(d))
    monkeypatch.setattr(torch.cuda, "Stream", stream)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: current)
    monkeypatch.setattr(torch.cuda, "stream", in_stream)
    monkeypatch.setattr(torch.Tensor, "record_stream", record_stream)
    whole_calls.clear()
    v1, a1 = sample(True)

    # Every pass that decodes is split around the side stream; only the
    # encoder-only passes of a skipped last video stay whole.
    assert not any(whole_calls)
    guided = cfg != 1.0 or cfg_v != 1.0
    split_steps = settings.num_steps - (skip_last_video and not guided)
    assert entered == ["side"] * split_steps
    assert log.count(("wait", "side", "main")) == split_steps
    assert log.count(("wait", "main", "side")) == split_steps
    assert ("record", "side") in log and ("record", "main") in log
    assert (v0 is None) == (v1 is None) == skip_last_video
    if v0 is not None:
        assert v1 is not None and v1.device.type == "cpu"
        assert torch.equal(v0, v1)
    assert torch.equal(a0, a1)
