# Left-Turn SmolVLA Training

This folder converts `vla-urban-4` target-action left-turn episodes into LeRobot v3, fine-tunes [`lerobot/smolvla_base`](https://huggingface.co/lerobot/smolvla_base), evaluates held-out actions, and serves the resulting policy to the browser simulator.

The code pins LeRobot `0.6.0`. That matters because dataset writing, processor files, resume behavior, and training arguments have changed across LeRobot releases.

## Why SmolVLA

SmolVLA is a 450M-parameter open VLA with a pretrained vision-language backbone and a flow-matching action expert. It accepts camera images, state, and language, then emits continuous action chunks. The fresh dataset has 204 curated left-turn episodes and 39,417 frames after filtering, so it is large enough to justify a fine-tuning run, though closed-loop performance still has to be measured.

The default configuration:

| Setting | Value |
| --- | ---: |
| Base checkpoint | `lerobot/smolvla_base` |
| Clean episodes | 204 |
| Camera | Front RGB, `128 x 128` |
| State dimension | 4 |
| Action dimension | 2 (`target_speed_mps`, `target_steering`) |
| Chunk size | 20 actions |
| Actions executed before replanning | 3 |
| Optimizer steps | 20,000 |
| Batch size | 32 |
| Held-out episodes | 21, seed-disjoint |
| Vision encoder | Trainable |

The state is `[speed_mps, steering, previous_target_speed_mps, previous_target_steering]`. Steering uses the simulator convention: negative is left and positive is right. The converter keeps the descriptive dataset key `observation.images.front`. During fine-tuning, LeRobot maps it to the pretrained SmolVLA camera slot `observation.images.camera1`; the saved preprocessor retains that mapping for evaluation and inference.
Video decoding is pinned to PyAV for training and held-out evaluation so the run does not depend on the Pod image's TorchCodec and FFmpeg ABI combination.

## Environment

Use Python 3.11 or 3.12. FFmpeg is required for LeRobot video datasets.

```bash
python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .
```

Create `.env` without committing the token:

```bash
cp .env.example .env
printf '%s\n' 'HF_TOKEN=hf_your_write_token' > .env
```

Optional variables are `HF_DATASET_REPO`, `HF_MODEL_REPO`, `WANDB_API_KEY`, `WANDB_PROJECT`, and `VLA_WORKSPACE_ROOT`.

## Convert the human recordings

The converter reads the repository's sibling `../left-turn-target/` folder by default and considers only `human-*.json` files. It reviewed 322 source episodes, accepted 204, rejected 118, and left every raw file untouched. It orders the accepted episodes so LeRobot's final 10% holdout contains 21 episodes whose simulator seeds never appear in training.

Review the selection:

```bash
python convert_dataset.py --dry-run
```

Write and publish the clean dataset:

```bash
python convert_dataset.py \
  --overwrite \
  --push-to-hub \
  --repo-id Mayank022/urban-vla-left-turn-cruise-human
```

The converter rejects old `vla-urban-3` raw-pedal recordings. Its nominal gates also reject collisions, off-route runs, exact duplicates, incomplete routes, poorly aligned finishes, and episodes with excessive stopped frames. Do not add `--include-recovery` for the first nominal run. Recovery episodes can be added later only as separately reviewed, continuous human corrections with no teleport or reset inside the episode.

The converted release is public at [`Mayank022/urban-vla-left-turn-cruise-human`](https://huggingface.co/datasets/Mayank022/urban-vla-left-turn-cruise-human).

## Inspect the training command

This prints the full LeRobot command and does not initialize CUDA or download weights:

```bash
python train.py --dry-run
```

Useful overrides:

```bash
python train.py --dry-run \
  --run-name smolvla-left-turn-pilot \
  --max-steps 2000 \
  --batch-size 16 \
  --no-push-to-hub
```

## RunPod

Start with a CUDA image that already has a compatible PyTorch build. On an RTX PRO 6000 Pod:

```bash
cd /workspace/Action_Chunking_Transformer_Autonomous_Driving/vla_training
./scripts/setup_runpod.sh
printf '%s\n' 'HF_TOKEN=hf_your_write_token' > .env
./scripts/start_runpod_tmux.sh
```

The tmux script is self-contained, so copying a wrapped multi-line command into the shell is unnecessary.

Monitor the pane:

```bash
tmux attach -t smolvla-left-turn
```

Detach with `Ctrl+B`, then `D`. The process keeps running.

Monitor the persistent log:

```bash
tail -F /workspace/vla-driving/logs/smolvla-left-turn-v2-launcher.log
```

Check whether the process still exists:

```bash
pgrep -af 'runpod_main.py|lerobot-train'
```

Run a 2,000-step pilot by passing arguments through the tmux script:

```bash
VLA_RUN_NAME=smolvla-left-turn-pilot \
VLA_MAX_STEPS=2000 \
VLA_BATCH_SIZE=16 \
./scripts/start_runpod_tmux.sh --no-push-to-hub
```

## Resume

`train.py` defaults to `--resume auto`. Running the same run name again finds:

```text
/workspace/vla-driving/runs/<run-name>/checkpoints/last/pretrained_model/train_config.json
```

It resumes model, optimizer, scheduler, step, and dataset order through LeRobot's native checkpoint loader.

Use `--resume require` when an absent checkpoint should be treated as an error. Use `--resume never` only for a new, empty run directory.

## Outputs

For the default run:

```text
/workspace/vla-driving/
|-- logs/smolvla-left-turn-v2.log
`-- runs/smolvla-left-turn-v2/
    |-- checkpoints/
    `-- run_artifacts/
        |-- run_manifest.json
        |-- training_metrics.jsonl
        |-- training_curves.png
        `-- evaluation/
            |-- metrics.json
            |-- prediction_scatter.png
            `-- action_trace.png
```

LeRobot pushes the final model, preprocessor, postprocessor, normalization statistics, model card, and training configuration to the model repository. The wrapper then uploads the evaluation files and log below `runs/<run-name>/`.

## Offline evaluation

Evaluation runs automatically after successful training. It uses the same final 10% of episodes that LeRobot holds out during training.

Run it separately when needed:

```bash
python evaluate.py \
  --model-path /workspace/vla-driving/runs/smolvla-left-turn-v2/checkpoints/last/pretrained_model \
  --dataset-repo Mayank022/urban-vla-left-turn-cruise-human \
  --output-dir /workspace/vla-driving/runs/smolvla-left-turn-v2/run_artifacts/evaluation \
  --device cuda
```

The included quality gates reject obvious action collapse, but they are not release approval. Always run the checkpoint in the simulator on unseen seeds.

## Local inference

Download the model:

```bash
python scripts/download_model.py
```

Start the server from the repository root:

```bash
npm run inference -- --model-path vla_training/artifacts/smolvla-left-turn-v2 --action-steps 1
```

Start the simulator separately with `npm run dev`, select the left-turn intent, and press `I`. Inference reads the learned normalizer shapes rather than the base checkpoint's generic feature metadata. It supports both the legacy `[throttle, brake, steering]` checkpoint and the current `[target_speed_mps, target_steering]` checkpoint, and the browser waits for that contract before sending its first frame. The server also translates the legacy checkpoint's positive-left steering to the current negative-left simulator convention in both observations and actions. `--action-steps 1` is the closed-loop default; it makes SmolVLA replan on every request instead of executing three stale actions from its queue.

## Tests

The tests do not download SmolVLA weights:

```bash
PYTHONPATH=src python -m pytest
python -m compileall src convert_dataset.py train.py evaluate.py inference_server.py runpod_main.py
```

They cover raw episode filtering, schema isolation, duplicate-safe dataset selection, unit conversion, previous-target alignment, training and resume command construction, metric parsing, and both browser action protocols.
