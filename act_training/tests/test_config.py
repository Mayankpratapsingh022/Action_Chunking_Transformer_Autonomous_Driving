from __future__ import annotations

import json
from pathlib import Path

import pytest

from urban_act.config import TrainConfig


def test_default_config_is_valid() -> None:
    TrainConfig().validate()


def test_default_h100_profile_preserves_sample_budget() -> None:
    config = TrainConfig()
    base_config = json.loads((Path(__file__).parents[1] / "configs" / "base.json").read_text())

    assert config.batch_size == base_config["batch_size"] == 64
    assert config.policy_version == base_config["policy_version"] == "v2"
    assert config.run_name == base_config["run_name"] == "act-driving-v2"
    assert config.kl_weight == base_config["kl_weight"] == 0.1
    assert config.max_steps == base_config["max_steps"] == 10_000
    assert config.warmup_steps == base_config["warmup_steps"] == 500
    assert config.eval_interval == base_config["eval_interval"] == 500
    assert config.checkpoint_interval == base_config["checkpoint_interval"] == 500
    assert config.eval_batches == base_config["eval_batches"] == 100
    assert config.test_batches == base_config["test_batches"] == 250
    assert config.batch_size * config.max_steps == 640_000
    assert config.batch_size * config.eval_batches == 6_400
    assert config.batch_size * config.test_batches == 16_000


def test_v1_reproduction_profile_keeps_the_original_objective() -> None:
    config_path = Path(__file__).parents[1] / "configs" / "v1.json"
    config = TrainConfig.from_json(config_path)

    assert config.policy_version == "v1"
    assert config.kl_weight == 10.0
    assert config.require_quality_gates is False
    config.validate()


def test_run_name_cannot_escape_artifact_root() -> None:
    with pytest.raises(ValueError, match="run_name"):
        TrainConfig(run_name="../outside").validate()


def test_transformer_width_must_match_head_count() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TrainConfig(d_model=250, nhead=8).validate()
