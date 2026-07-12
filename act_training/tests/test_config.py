from __future__ import annotations

import pytest

from urban_act.config import TrainConfig


def test_default_config_is_valid() -> None:
    TrainConfig().validate()


def test_run_name_cannot_escape_artifact_root() -> None:
    with pytest.raises(ValueError, match="run_name"):
        TrainConfig(run_name="../outside").validate()


def test_transformer_width_must_match_head_count() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TrainConfig(d_model=250, nhead=8).validate()

