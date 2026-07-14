from pathlib import Path

import pytest

from left_turn_vla.config import TrainConfig


def test_left_turn_config_loads() -> None:
    config = TrainConfig.from_json(Path(__file__).parents[1] / "configs" / "left_turn.json")
    assert config.base_model == "lerobot/smolvla_base"
    assert config.chunk_size == 20
    assert config.action_steps == 3
    assert not config.freeze_vision_encoder
    assert config.rename_map == {"observation.images.front": "observation.images.camera1"}


def test_invalid_action_horizon_is_rejected() -> None:
    with pytest.raises(ValueError, match="action_steps"):
        TrainConfig(chunk_size=3, action_steps=4).validate()
