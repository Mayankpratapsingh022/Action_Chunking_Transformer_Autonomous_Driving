---
library_name: pytorch
datasets:
- Mayank022/urban-vla-expert-v1
tags:
- robotics
- autonomous-driving
- imitation-learning
- action-chunking-transformer
- vision-language-action
- modal
---

# Urban VLA Language ACT

This repository contains the training code for a language-conditioned Action Chunking Transformer on [`Mayank022/urban-vla-expert-v1`](https://huggingface.co/datasets/Mayank022/urban-vla-expert-v1).

No training job has been launched yet, and there is no checkpoint in this repository at the time of publication. The code is set up for a Modal A10G run. When training finishes, it saves the best and latest checkpoints to a Modal Volume, downloads the run folder to this machine, and uploads the model weights, metrics, plots, and tokenizer to this Hugging Face repository.

<!-- TRAINING_RESULTS_START -->
Training has not been run yet. This section is replaced with the measured validation and test results after a successful run.
<!-- TRAINING_RESULTS_END -->

## Why this is not stock ACT

Plain ACT consumes images and robot state. It does not consume a sentence. That is a problem here because the first camera frames for "turn left," "continue straight," and "turn right" can be nearly identical.

This model adds a frozen MiniLM text encoder and gives its language token to the ACT transformer. The rest follows the ACT idea closely:

- a pretrained ResNet-18 produces spatial image tokens
- the four-dimensional ego state is a separate token
- MiniLM encodes the instruction text
- a CVAE posterior models variation in valid action chunks during training
- learned action queries predict 20 future controls at 10 Hz
- the output is `[throttle, brake, steering]`

The action transformer and CVAE are trained on this dataset. ResNet-18 and MiniLM start from public pretrained weights because 8.9 hours of simulator data is too small for sensible vision and language pretraining from random initialization.

## What the run produces

Every run writes to `runs/<run-name>/` on the `urban-vla-act-artifacts` Modal Volume:

```text
runs/<run-name>/
|-- best.pt
|-- last.pt
|-- model.safetensors
|-- config.json
|-- training_config.json
|-- tokenizer/
|-- history.json
|-- metrics.json
|-- logs/train.jsonl
`-- plots/
    |-- training_curves.png
    |-- action_mae.png
    |-- prediction_scatter.png
    `-- sample_chunks.png
```

`best.pt` and `last.pt` contain optimizer state for resuming. `model.safetensors` is the inference weight file. The Hugging Face upload also keeps a copy of each run under `runs/<run-name>/` and publishes the best weights at the repository root.

## One-time setup

Run these commands from this folder:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-local.txt

hf auth login
modal setup
modal secret create huggingface HF_TOKEN=hf_your_write_token
```

The Hugging Face token must be able to write to `Mayank022/urban-vla-language-act`. Do not put the token in this repository. The public dataset can be downloaded without a token; the secret is needed for the model upload.

## Start the training job later

This is the command for the first real run. It is documented here but has not been executed:

```bash
modal run modal_app.py --run-name urban-act-v1
```

The local entrypoint waits for the GPU function to finish, then downloads the run to `artifacts/urban-act-v1/`. It also pushes the trained model to Hugging Face from inside the Modal function.

Useful overrides:

```bash
modal run modal_app.py \
  --run-name urban-act-v1 \
  --max-steps 20000 \
  --batch-size 32
```

Watch the logs from another terminal:

```bash
modal app logs urban-vla-language-act-training
```

Each log line reports the step, percentage, loss, learning rate, elapsed time, ETA, and validation metrics when available.

## Resume after an interruption

Use the same run name:

```bash
modal run modal_app.py --run-name urban-act-v1
```

With `resume` set to `auto`, the trainer loads `last.pt` from the Modal Volume and continues from its recorded step. Checkpoints are written every 1,000 steps by default.

You can also download a run manually:

```bash
modal volume get urban-vla-act-artifacts \
  runs/urban-act-v1 \
  artifacts/urban-act-v1
```

## Evaluation

The trainer selects the best checkpoint by validation action MAE and touches the test split only after training. It reports:

- masked L1 loss over the full action chunk
- throttle, brake, and steering MAE and RMSE
- steering-direction accuracy when the target turn is large enough to be meaningful
- braking classification accuracy
- simultaneous throttle-and-brake rate
- MAE broken down by language intent

These are open-loop metrics. They tell us whether the model matches recorded controls, but they do not tell us whether the car completes a route. A checkpoint should only be called useful after closed-loop simulator rollouts measure success rate, collisions, off-road time, red-light violations, pedestrian violations, and recovery success.

## Data policy

Only `raw/accepted/` is used for behavior cloning. This includes nominal expert driving and controlled expert recoveries. The deliberately unsafe files in `raw/failures/` are excluded. Treating an unsafe action as another target would teach the policy to reproduce it.

The supplied manifest split is used as-is. The loader does not reshuffle episodes into new splits. Horizontal image flips are also disabled because a flip would require changing both the steering target and the meaning of left/right instructions.

## Local checks

The repository can be checked without starting Modal or downloading model weights:

```bash
python -m compileall modal_app.py src tests scripts
python -m pytest
```

The tests use small dummy encoders. They do not train a policy and do not make a GPU request.

## Safety note

This code and dataset are for simulator research. Do not connect a checkpoint from this project to a real vehicle.
