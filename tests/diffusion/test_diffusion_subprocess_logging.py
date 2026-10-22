# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

import pytest
import vllm.logger

from vllm_omni.logger import configure_subprocess_logging

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion, pytest.mark.cpu]


def test_subprocess_logging_follows_inherited_env(monkeypatch, tmp_path):
    """Spawned diffusion stage/worker processes must apply the parent's VLLM_* logging env."""
    config_file = tmp_path / "logging.json"
    config_file.write_text("{}")
    monkeypatch.setenv("VLLM_LOGGING_LEVEL", "DEBUG")
    monkeypatch.setenv("VLLM_LOGGING_CONFIG_PATH", str(config_file))
    applied = []
    monkeypatch.setattr(vllm.logger, "configure_logging_if_needed", applied.append)

    configure_subprocess_logging()

    assert len(applied) == 1
    assert applied[0].log_level == "DEBUG"
    assert applied[0].pylogging_config_file == str(config_file)
