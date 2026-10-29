# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Accuracy of ``diffusion_use_graph`` on Wan2.1 T2V.

``diffusion_kv_cache_skip_steps`` keeps the early denoise steps in full
precision because quantizing their KV costs accuracy. A replayed graph would
freeze the capture step's choice, so graph mode must not change the frames of
any configuration, and skipping steps must still beat quantizing every step.

Override the checkpoint with ``VLLM_OMNI_DENOISE_GRAPH_MODEL``::

    VLLM_OMNI_DENOISE_GRAPH_MODEL=/models/Wan2.1-T2V-1.3B-Diffusers \
        pytest -s tests/e2e/accuracy/test_wan21_denoise_graph_accuracy.py
"""

from __future__ import annotations

import functools
import os

import numpy as np
import pytest
from PIL import Image

from tests.e2e.accuracy.helpers import assert_similarity_metrics, compute_image_ssim_psnr
from tests.helpers.mark import hardware_test
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = os.environ.get("VLLM_OMNI_DENOISE_GRAPH_MODEL", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
PROMPT = "A serene lakeside sunrise with mist over the water."
SKIP_STEPS = "0-1"

# Graph replay runs the same kernels as eager, so it must sit within eager's own
# run-to-run spread (measured by a second eager run) or above these floors.
GRAPH_SSIM, GRAPH_PSNR = 0.99, 40.0
NOISE_SSIM_MARGIN, NOISE_PSNR_MARGIN_DB = 0.005, 3.0
# Two eager runs must at least agree this well, or the device is too noisy to judge graphs.
EAGER_SSIM, EAGER_PSNR = 0.95, 30.0
# fp8 KV with the early steps kept in full precision, against the bf16 reference.
FP8_SKIP_SSIM, FP8_SKIP_PSNR = 0.80, 20.0

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


@functools.cache
def _frames(run: int = 0, **omni_kwargs) -> tuple[Image.Image, ...]:
    """``run`` only separates cache entries, so a second eager run is really regenerated."""
    from vllm_omni.entrypoints.omni import Omni

    engine = Omni(model=MODEL, num_gpus=1, **omni_kwargs)
    try:
        outputs = engine.generate(
            {"prompt": PROMPT},
            OmniDiffusionSamplingParams(
                output_type="np",
                seed=42,
                num_inference_steps=20,
                guidance_scale=5.0,
                height=512,
                width=512,
                num_frames=17,
            ),
        )
    finally:
        engine.close()
    video = outputs[0].images[0]
    video = video.numpy() if hasattr(video, "numpy") else np.asarray(video)
    video = video.reshape(-1, *video.shape[-3:])
    if video.dtype != np.uint8:
        video = np.rint(np.clip(video, 0.0, 1.0) * 255.0).astype(np.uint8)
    return tuple(Image.fromarray(frame) for frame in video)


def _similarity(prediction: tuple[Image.Image, ...], reference: tuple[Image.Image, ...]) -> tuple[float, float]:
    assert len(prediction) == len(reference)
    scores = [compute_image_ssim_psnr(prediction=p, reference=r) for p, r in zip(prediction, reference)]
    ssim, psnr = zip(*scores)
    return float(np.mean(ssim)), float(np.mean([min(v, 100.0) for v in psnr]))


def _assert_close(name: str, prediction, reference, ssim_floor: float, psnr_floor: float) -> tuple[float, float]:
    ssim, psnr = _similarity(prediction, reference)
    assert_similarity_metrics(
        model_name=name, ssim_score=ssim, psnr_score=psnr, ssim_threshold=ssim_floor, psnr_threshold=psnr_floor
    )
    return ssim, psnr


def _assert_graph_within_eager_noise(kv_kwargs: dict) -> None:
    reference = _frames(**kv_kwargs)
    noise_ssim, noise_psnr = _assert_close(
        "eager vs eager", _frames(run=1, **kv_kwargs), reference, EAGER_SSIM, EAGER_PSNR
    )
    _assert_close(
        "graph vs eager",
        _frames(diffusion_use_graph=True, **kv_kwargs),
        reference,
        min(GRAPH_SSIM, noise_ssim - NOISE_SSIM_MARGIN),
        min(GRAPH_PSNR, noise_psnr - NOISE_PSNR_MARGIN_DB),
    )


@hardware_test(res={"cuda": "H100", "xpu": "B70"}, num_cards=1)
@pytest.mark.parametrize(
    "kv_kwargs",
    [{}, {"diffusion_kv_cache_skip_steps": SKIP_STEPS}],
    ids=["bf16", "bf16-skip-steps"],
)
def test_graph_matches_eager_bf16(kv_kwargs) -> None:
    _assert_graph_within_eager_noise(kv_kwargs)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
@pytest.mark.parametrize(
    "kv_kwargs",
    [
        {"diffusion_kv_cache_dtype": "fp8"},
        {"diffusion_kv_cache_dtype": "fp8", "diffusion_kv_cache_skip_steps": SKIP_STEPS},
    ],
    ids=["fp8", "fp8-skip-steps"],
)
def test_graph_matches_eager_fp8(kv_kwargs) -> None:
    _assert_graph_within_eager_noise(kv_kwargs)


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_skip_steps_keep_fp8_accuracy_with_graph() -> None:
    reference = _frames()
    skip_ssim, skip_psnr = _assert_close(
        "fp8 + skip steps (graph) vs bf16",
        _frames(diffusion_use_graph=True, diffusion_kv_cache_dtype="fp8", diffusion_kv_cache_skip_steps=SKIP_STEPS),
        reference,
        FP8_SKIP_SSIM,
        FP8_SKIP_PSNR,
    )
    all_ssim, all_psnr = _similarity(_frames(diffusion_use_graph=True, diffusion_kv_cache_dtype="fp8"), reference)
    print(f"fp8 on every step vs bf16: SSIM={all_ssim:.6f} PSNR={all_psnr:.4f} dB")

    assert skip_ssim >= all_ssim and skip_psnr >= all_psnr, (
        f"skip steps {SKIP_STEPS} did not improve fp8 accuracy: "
        f"SSIM {skip_ssim:.6f} vs {all_ssim:.6f}, PSNR {skip_psnr:.4f} vs {all_psnr:.4f}"
    )
