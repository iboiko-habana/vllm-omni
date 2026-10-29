# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""End-to-end checks for ``diffusion_use_graph`` on Wan2.1 T2V.

Each case starts a fresh engine and reads the engine log, from ``caplog`` when
it runs in-process and ``capfd`` when a worker process writes it, to see whether
denoise graphs were captured or refused.

Override the checkpoint with ``VLLM_OMNI_DENOISE_GRAPH_MODEL``::

    VLLM_OMNI_DENOISE_GRAPH_MODEL=/models/Wan2.1-T2V-1.3B-Diffusers \
        pytest -s tests/e2e/offline_inference/test_wan21_denoise_graph.py
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pytest

from tests.helpers.mark import hardware_test
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

MODEL = os.environ.get("VLLM_OMNI_DENOISE_GRAPH_MODEL", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers")
PROMPT = "A serene lakeside sunrise with mist over the water."
CAPTURED = "captured denoise graph"
REFUSED = "diffusion_use_graph ignored because"
# Eager is not bitwise reproducible on every device (XPU B70 moves ~79% of pixels
# by up to 0.045 between two eager runs), so compare with PSNR rather than per pixel.
MIN_PSNR_DB = 40.0

pytestmark = [pytest.mark.full_model, pytest.mark.diffusion]


def _generate(capfd, caplog, **omni_kwargs) -> tuple[np.ndarray, str]:
    from vllm_omni.entrypoints.omni import Omni

    caplog.set_level(logging.INFO)
    caplog.clear()
    capfd.readouterr()
    engine = Omni(model=MODEL, num_gpus=1, **omni_kwargs)
    try:
        outputs = engine.generate(
            {"prompt": PROMPT},
            OmniDiffusionSamplingParams(
                output_type="np",
                seed=42,
                num_inference_steps=4,
                guidance_scale=5.0,
                height=256,
                width=256,
                num_frames=9,
            ),
        )
    finally:
        engine.close()
    out, err = capfd.readouterr()
    video = outputs[0].images[0]
    return (video.numpy() if hasattr(video, "numpy") else np.asarray(video)), out + err + caplog.text


@hardware_test(res={"cuda": "H100", "xpu": "B70"}, num_cards=1)
def test_graph_replay_matches_eager(capfd, caplog) -> None:
    eager, eager_log = _generate(capfd, caplog)
    graph, graph_log = _generate(capfd, caplog, diffusion_use_graph=True)

    assert CAPTURED not in eager_log
    assert CAPTURED in graph_log
    assert REFUSED not in graph_log
    assert eager.shape == graph.shape
    mse = float(np.mean((graph.astype(np.float64) - eager.astype(np.float64)) ** 2))
    psnr = 10 * np.log10(1.0 / max(mse, 1e-12))
    print(f"graph vs eager: PSNR={psnr:.2f} dB max_abs={np.abs(graph - eager).max():.5f}")
    assert psnr >= MIN_PSNR_DB, f"graph output drifted from eager: PSNR {psnr:.2f} dB < {MIN_PSNR_DB} dB"


@hardware_test(res={"cuda": "H100", "xpu": "B70"}, num_cards=1)
def test_skip_steps_without_kv_quant_keeps_graph(capfd, caplog) -> None:
    """Skip steps only gate KV quantization, so without it graphs stay safe."""
    _, log = _generate(capfd, caplog, diffusion_use_graph=True, diffusion_kv_cache_skip_steps="0-1")

    assert REFUSED not in log
    assert CAPTURED in log


@hardware_test(res={"cuda": "H100"}, num_cards=1)
def test_skip_steps_with_kv_quant_runs_eager(capfd, caplog) -> None:
    """Quantizing KV on some steps only would be frozen into a replayed graph."""
    _, log = _generate(
        capfd,
        caplog,
        diffusion_use_graph=True,
        diffusion_kv_cache_dtype="fp8",
        diffusion_kv_cache_skip_steps="0-1",
    )

    assert f"{REFUSED} diffusion_kv_cache_skip_steps" in log
    assert CAPTURED not in log
