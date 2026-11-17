"""The public calls of a session run under ``torch.inference_mode()`` (CPU only).

A pipeline with the tiny random policy, a stub autoencoder and a stub text
encoder runs the real session: the rig session, the window sampler, the
plan and the encode memo. Each call must run in inference mode and give the
bits of the same call under ``torch.no_grad()``, the context of the calls
before inference mode. The tensors that a call keeps for the next one (the
text cache, the session context, the latents of the memo) and the values it
returns must work in the code around the calls, which runs outside
inference mode.
"""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from tiny_model import TINY_HW, tiny_policy
from torch import Tensor
from torch.nn.attention.flex_attention import flex_attention

import praxis.layers
from praxis import profiling
from praxis import single_arm as sa
from praxis.checkpoint import model_variant
from praxis.model import SamplerSettings
from praxis.pipeline import PraxisPipeline, Session, StepResult
from praxis.robot_io import NormStats

SETTINGS = SamplerSettings(num_steps=2, cfg_video_scale=4.0)
PROMPT = "Put the cube on the plate."
CALLS = 2


def _encode(clip: Tensor) -> Tensor:
    """A stub video encoder: ``(1, 3, T, H, W)`` in [0, 255] to tiny latents."""
    t = clip.shape[2]
    idx = [0] + [4 * k for k in range(1, (t - 1) // 4 + 1)]
    x = F.adaptive_avg_pool3d(clip[:, :, idx] / 255.0, (len(idx), *TINY_HW))
    return x.repeat(1, 64, 1, 1, 1).to(torch.bfloat16)


def _decode(latent: Tensor) -> Tensor:
    """A stub video decoder: uint8 frames, 4 per latent after the first."""
    x = latent[0, :3].float().permute(1, 0, 2, 3)
    x = torch.cat([x[:1], x[1:].repeat_interleave(4, dim=0)])
    return (x * 100 + 128).clamp(0, 255).to(torch.uint8)


def _encode_text(prompts: list[str]) -> list[Tensor]:
    g = torch.Generator().manual_seed(len(prompts[0]))
    return [torch.randn(1, 3, 32, generator=g).to(torch.bfloat16)]


def _pipe() -> PraxisPipeline:
    """A pipeline with the real session and the tiny policy."""
    policy = tiny_policy()
    pipe = object.__new__(PraxisPipeline)
    pipe.device = torch.device("cpu")
    pipe.policy = policy
    pipe.vae = SimpleNamespace(  # ty: ignore[invalid-assignment]
        encode=_encode, decode=_decode
    )
    pipe.text = SimpleNamespace(encode=_encode_text)  # ty: ignore[invalid-assignment]
    pipe.sampler = SETTINGS
    pipe.embodiment = "single_arm"
    pipe.variant = model_variant(policy.config)
    pipe.stats = NormStats(
        action_q01=np.full(10, -0.5, np.float32),
        action_q99=np.full(10, 0.5, np.float32),
        proprio_mean=np.zeros(10, np.float32),
        proprio_std=np.ones(10, np.float32),
    )
    pipe._text_cache = {}
    return pipe


def _obs(k: int) -> dict:
    return profiling.synthetic_obs(k, 1 if k == 0 else 25, 234, 416)


@dataclass
class Run:
    session: Session
    results: list[StepResult]
    videos: list[Tensor]
    #: The clips that each call encoded.
    clips: list[list[Tensor]]


def _run(no_grad: bool, plans: Run | None = None) -> Run:
    """``CALLS`` calls of one session with a growing context, and their videos.

    ``no_grad`` runs the undecorated calls under ``torch.no_grad()``. With
    ``plans`` (a run of the same calls), the session keeps the latents of each
    call after the first and plans it before the call, outside inference mode,
    as the precompute of a real-time session does (:mod:`praxis.serve.precompute`).
    """
    step, decode = Session.step, Session.decode_video
    if no_grad:
        step = step.__wrapped__  # ty: ignore[unresolved-attribute]
        decode = decode.__wrapped__  # ty: ignore[unresolved-attribute]
    session = _pipe().session(
        PROMPT, seed=3, max_context_chunks=1, precompute=plans is not None
    )
    clips: list[list[Tensor]] = []
    encode = session.state.encode

    def recorded(clip: Tensor) -> Tensor:
        clips[-1].append(clip)
        return encode(clip)

    session.state.encode = recorded
    run = Run(session, [], [], clips)
    for k in range(CALLS):
        clips.append([])
        if plans is not None and k:
            memo = session.memo
            assert memo is not None
            with torch.no_grad():
                for clip in plans.clips[k]:
                    assert not memo.store(clip.clone()).is_inference()
            inputs = plans.results[k].inputs
            anchor = inputs["anchor"].clone()
            session.plan(anchor, [c.clone() for c in inputs["cleans"]], PROMPT)
            assert session.has_plan
        with torch.no_grad() if no_grad else torch.enable_grad():
            res = step(session, _obs(k))
            run.videos.append(decode(session, res))
        run.results.append(res)
        if plans is not None:
            assert session.plan_result == ("hit" if k else None)
            assert session.memo_hits == (len(plans.clips[k]) if k else 0)
    return run


@pytest.fixture(scope="module")
def runs() -> dict[str, Run]:
    with pytest.MonkeyPatch.context() as mp:
        # The compiled CPU flex kernel of torch 2.13 fails to build for some masks.
        mp.setattr(praxis.layers, "_compiled_flex_attention", flex_attention)
        out = {"no_grad": _run(no_grad=True)}
        out["inference"] = _run(no_grad=False)
        out["planned"] = _run(no_grad=False, plans=out["no_grad"])
    return out


def _commands(res: StepResult) -> str:
    return json.dumps(res.commands, sort_keys=True, default=lambda a: a.tolist())


def _assert_same(a: StepResult, b: StepResult) -> None:
    assert torch.equal(a.pred_action_norm, b.pred_action_norm)
    assert a.pred_video_latent is not None and b.pred_video_latent is not None
    assert torch.equal(a.pred_video_latent, b.pred_video_latent)
    assert np.array_equal(a.action_rows_norm, b.action_rows_norm)
    assert np.array_equal(a.action_rel, b.action_rel)
    assert _commands(a) == _commands(b)
    assert (a.depth, a.seed, a.prompt) == (b.depth, b.seed, b.prompt)


def test_calls_run_in_inference_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """``plan``, ``step`` and ``decode_video`` run in inference mode, also when
    the caller runs with grad enabled."""
    monkeypatch.setattr(praxis.layers, "_compiled_flex_attention", flex_attention)
    pipe = _pipe()
    seen: dict[str, list[bool]] = {"sample": [], "plan": [], "decode": []}
    policy = pipe.policy

    def spy(name: str, fn):
        def call(*args, **kwargs):
            seen[name].append(torch.is_inference_mode_enabled())
            return fn(*args, **kwargs)

        return call

    sample, plan = policy.sample_window_joint, policy.plan_window_joint
    monkeypatch.setattr(policy, "sample_window_joint", spy("sample", sample))
    monkeypatch.setattr(policy, "plan_window_joint", spy("plan", plan))
    pipe.vae.decode = spy("decode", pipe.vae.decode)
    session = pipe.session(PROMPT, max_context_chunks=1, precompute=True)
    assert torch.is_grad_enabled() and not torch.is_inference_mode_enabled()
    res = session.step(_obs(0))
    frames = session.decode_video(res)
    session.plan(res.inputs["anchor"], [], PROMPT)
    assert seen == {"sample": [True], "plan": [True], "decode": [True]}
    for t in (res.pred_action_norm, res.pred_video_latent, frames):
        assert t is not None and t.is_inference()


def test_calls_give_the_bits_of_no_grad(runs: dict[str, Run]) -> None:
    """The calls and their decoded videos give the same bits under inference
    mode as under ``torch.no_grad()``. The second call reads the text cache
    and the context that the first call kept."""
    want, got = runs["no_grad"], runs["inference"]
    assert [r.depth for r in got.results] == list(range(CALLS))
    for a, b in zip(got.results, want.results, strict=True):
        assert a.pred_action_norm.is_inference()
        assert not b.pred_action_norm.is_inference()
        _assert_same(a, b)
    for a, b in zip(got.videos, want.videos, strict=True):
        assert a.is_inference() and not b.is_inference()
        assert torch.equal(a, b)
    cache = got.session.pipe._text_cache
    assert cache and all(t.is_inference() for t in cache.values())


def test_plans_and_kept_latents_made_outside_the_calls(runs: dict[str, Run]) -> None:
    """The latents kept in the memo and the plan inputs are ordinary tensors
    made outside inference mode. The call that takes them gives the bits of
    the call without them under ``torch.no_grad()``."""
    want, got = runs["no_grad"], runs["planned"]
    assert all(want.clips[1:]), "the later calls encode clips"
    for a, b in zip(got.results, want.results, strict=True):
        _assert_same(a, b)
    for a, b in zip(got.videos, want.videos, strict=True):
        assert torch.equal(a, b)


def test_results_work_outside_inference_mode(runs: dict[str, Run]) -> None:
    """The callers use the returned values outside inference mode
    (``scripts/run_inference.py``, ``scripts/benchmark_inference.py``, the
    server): they read, index, move and save them, and decode the video
    latent again."""
    run = runs["inference"]
    res, frames = run.results[-1], run.videos[-1]
    latent = res.pred_video_latent
    assert latent is not None and res.depth == 1
    with torch.no_grad():
        rec = {
            "pred_video_latent": latent.cpu(),
            "pred_action_norm": res.pred_action_norm.cpu(),
            "latent_anchor": res.inputs["anchor"].cpu(),
            "latent_cleans": [c.cpu() for c in res.inputs["cleans"]],
            "action_rows_norm": torch.from_numpy(res.action_rows_norm),
        }
        buf = io.BytesIO()
        torch.save(rec, buf)
        buf.seek(0)
        back = torch.load(buf, weights_only=False)
        assert torch.equal(back["pred_video_latent"], latent)
        assert torch.equal(back["latent_cleans"][0], res.inputs["cleans"][0])
        arr = frames.permute(0, 2, 3, 1).contiguous().cpu().numpy()
        assert arr.shape == (len(res.commands), *TINY_HW, 3)
        assert torch.equal(run.session.pipe.vae.decode(latent), _decode(latent))
        actions = res.pred_action_norm.detach().cpu().contiguous()
        assert torch.equal(actions[0, -1], res.pred_action_norm[0, -1])
        # An op on the outputs makes an ordinary tensor, which can be updated.
        x = res.pred_action_norm.float() * 2
        assert not x.is_inference()
        x.add_(1)
    assert isinstance(res.action_rows_norm, np.ndarray)
    assert res.action_rows_norm.shape[0] == len(res.commands)


def test_the_pinned_canvas_is_writable_in_and_outside_inference_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The first call makes the pinned canvas (:data:`praxis.single_arm.TILE_U8_PINNED`)
    in inference mode. It is an ordinary tensor, so the precompute can rewrite
    it outside the mode and the next call inside it. The CPU build has no
    pinned memory: the test makes plain memory."""
    empty = torch.empty

    def unpinned(*args, pin_memory: bool = False, **kwargs):
        return empty(*args, **kwargs)

    monkeypatch.setattr(sa, "_PINNED_CANVAS", {})
    monkeypatch.setattr(torch, "empty", unpinned)
    shape = (2, 3, 4, 8)
    with torch.inference_mode():
        buf = sa._pinned_canvas(shape, torch.channels_last)
        assert not buf.is_inference()
        buf.copy_(torch.ones(shape, dtype=torch.uint8))
    with torch.no_grad():
        assert sa._pinned_canvas(shape, torch.channels_last) is buf
        buf[..., :4].copy_(torch.zeros(2, 3, 4, 4, dtype=torch.uint8))
    with torch.inference_mode():
        assert sa._pinned_canvas(shape, torch.channels_last) is buf
        assert int(buf.sum()) == 2 * 3 * 4 * 4
