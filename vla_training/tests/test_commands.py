from pathlib import Path

from left_turn_vla.commands import build_train_command, find_resume_config
from left_turn_vla.config import TrainConfig


def test_new_training_command_uses_pretrained_smolvla(tmp_path: Path) -> None:
    config = TrainConfig()
    command = build_train_command(config, output_dir=tmp_path, executable="lerobot-train")
    assert "--policy.path=lerobot/smolvla_base" in command
    assert "--policy.chunk_size=20" in command
    assert "--policy.n_action_steps=3" in command
    assert "--policy.freeze_vision_encoder=false" in command
    assert "--dataset.eval_split=0.1" in command
    assert '--rename_map={"observation.images.front":"observation.images.camera1"}' in command


def test_resume_command_uses_checkpoint_config(tmp_path: Path) -> None:
    resume = tmp_path / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    resume.parent.mkdir(parents=True)
    resume.write_text("{}")
    assert find_resume_config(tmp_path) == resume.resolve()
    command = build_train_command(TrainConfig(), output_dir=tmp_path, resume_config=resume)
    assert "--resume=true" in command
    assert not any(item.startswith("--policy.path=") for item in command)
