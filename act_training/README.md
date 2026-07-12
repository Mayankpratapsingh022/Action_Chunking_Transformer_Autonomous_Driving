---
library_name: pytorch
datasets:
- Mayank022/urban-vla-expert-v1
tags:
- deep-learning
- autonomous-driving
- imitation-learning
- action-chunking-transformer
- vision-language-action
- modal
---

# Action Chunking Transformer for Autonomous Driving

A language-conditioned deep-learning policy that predicts short sequences of driving controls from a front-camera image, vehicle state, and a written instruction.

The policy is built around an **Action Chunking Transformer (ACT)**. Instead of predicting one steering command at a time, it predicts the next 20 controls together. At the dataset rate of 10 Hz, each chunk covers two seconds of driving. The controller executes a few actions, observes the road again, and replans.

> **Project status:** the dataset and training code are ready, but no GPU training run has been launched. There is no trained checkpoint or claimed driving result yet.

<!-- TRAINING_RESULTS_START -->
Training has not been run yet. This section is replaced with measured validation and test results after a successful run.
<!-- TRAINING_RESULTS_END -->

## The learning problem

Autonomous driving is a sequential decision problem. A useful policy has to understand what is visible, remember the current vehicle condition, follow the requested route, and produce controls that make sense together over time.

The model receives:

| Input | Shape | Meaning |
| --- | --- | --- |
| Front camera | `3 x 128 x 128` | Current road scene, resized from the 256 x 256 source video |
| Vehicle state | `4` | Speed, steering, previous throttle, previous brake |
| Language instruction | Variable text | Requested driving behavior |

It predicts:

| Output | Shape | Range |
| --- | --- | --- |
| Action chunk | `20 x 3` | 20 future `[throttle, brake, steering]` controls |
| Throttle | Scalar per step | `[0, 1]` |
| Brake | Scalar per step | `[0, 1]` |
| Steering | Scalar per step | `[-1, 1]` |

Language matters here. The first camera frame can look almost identical for "turn left," "continue straight," and "turn right." A vision-only ACT policy cannot know which route the driver requested.

## Model architecture

```text
Front camera
    |
    v
ResNet-18 feature map -------> 4 x 4 spatial vision tokens

Vehicle state --------------> state MLP -------------> state token

Language instruction -------> frozen MiniLM ---------> language token

Target action chunk --------> CVAE posterior --------> latent token
                              (training only)

vision + state + language + latent tokens
                    |
                    v
          Transformer context encoder
                    |
       20 learned action queries
                    |
                    v
          Transformer action decoder
                    |
                    v
       throttle, brake, steering chunk
```

### Vision

A pretrained ResNet-18 keeps the final spatial feature map instead of collapsing it to a single classification vector. At `128 x 128`, the map becomes 16 visual tokens. The backbone is fine-tuned with a smaller learning rate than the policy layers.

### Language

MiniLM encodes the instruction. Its token embeddings are mean-pooled and projected into the ACT transformer width. MiniLM is frozen by default. The dataset is large enough to learn the driving policy, but it is not large enough to teach a language encoder from random initialization.

### Action chunking

Twenty learned queries decode the full control horizon in parallel. This gives the model an explicit two-second control plan rather than a collection of unrelated single-step guesses. During inference, `RecedingHorizonController` executes three actions by default and then requests a fresh chunk from the latest observation.

### CVAE latent

During training, a posterior transformer reads the target action chunk and predicts a latent distribution. This gives ACT a way to model variation between valid demonstrations. At inference time, the latent is set to zero, so the policy output is deterministic for a given observation and instruction.

## Training objective

The policy uses masked action reconstruction plus KL regularization:

```text
loss = masked_L1(predicted_actions, expert_actions) + beta * KL(q(z|a) || N(0, I))
```

The default KL weight is `10.0`. Tail positions near the end of an episode are padded, then removed from both the loss and evaluation metrics with an action mask.

The default optimization setup is:

| Setting | Value |
| --- | ---: |
| Optimizer | AdamW |
| GPU | Modal A10G |
| Batch size | 32 |
| Training steps | 20,000 |
| Policy learning rate | `1e-4` |
| ResNet learning rate | `1e-5` |
| Warmup | 1,000 steps |
| Schedule | Cosine decay |
| Precision | BF16 |
| Gradient clipping | `1.0` |
| Checkpoint interval | 1,000 steps |

All defaults live in [`configs/base.json`](configs/base.json).

## Driving dataset

Training uses [`Mayank022/urban-vla-expert-v1`](https://huggingface.co/datasets/Mayank022/urban-vla-expert-v1), a synchronized simulator dataset with camera video, language, vehicle state, and continuous controls.

| Split | Expert episodes | Frames | Hours |
| --- | ---: | ---: | ---: |
| Train | 756 | 224,133 | 6.23 |
| Validation | 162 | 47,614 | 1.32 |
| Test | 162 | 47,933 | 1.33 |

The driving instructions cover left, right, and straight intersection routes, traffic lights, pedestrians, slow-vehicle passing, cut-in yielding, curved-road following, and blocked-lane detours.

The loader follows the supplied split assignments. It does not generate a fresh random split. This keeps held-out world seeds and instruction paraphrases out of training.

The published MP4 files stay at their original `256 x 256` resolution. PyAV resizes each decoded frame directly to `128 x 128` for training. This happens in memory and does not create another dataset folder.

### Recovery and failure data

Controlled recovery demonstrations are part of the expert set. They teach the policy how to return from lateral offsets, heading errors, overspeed, and stalled starts.

The 90 deliberately unsafe episodes under `raw/failures/` are not behavior-cloning targets. They are reserved for analysis. Asking the model to imitate those actions would be the wrong objective.

## Repository structure

```text
act-autonomous-driving/
|-- configs/
|   `-- base.json                 # Reproducible training defaults
|-- src/urban_act/
|   |-- config.py                 # Validated training configuration
|   |-- data.py                   # Video streaming and action chunks
|   |-- model.py                  # Language-conditioned ACT model
|   |-- losses.py                 # Masked L1 and KL loss
|   |-- metrics.py                # Open-loop driving metrics
|   |-- plots.py                  # Training and prediction figures
|   |-- checkpoints.py            # Resume and inference weights
|   |-- train.py                  # Training and evaluation loop
|   |-- inference.py              # Policy and receding-horizon control
|   `-- hub.py                    # Hugging Face model publishing
|-- tests/                        # CPU unit tests with dummy encoders
|-- modal_app.py                  # A10G training entrypoint
|-- requirements-local.txt        # Local Modal and Hub tools
|-- requirements.txt              # Full training environment
`-- README.md
```

The video loader decodes each episode sequentially and constructs future action chunks in memory. It does not extract hundreds of thousands of PNG files before training.

## Modal setup

Run the setup from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-local.txt

hf auth login
modal setup
modal secret create huggingface HF_TOKEN=hf_your_write_token
```

The Hugging Face token needs write access to [`Mayank022/urban-vla-language-act`](https://huggingface.co/Mayank022/urban-vla-language-act). Keep it in the Modal secret. Do not add it to a config file or commit it to Git.

## Start training

The following command is ready for the first run, but it has not been executed:

```bash
modal run modal_app.py --run-name act-driving-v1
```

Override the two settings most likely to change during an initial experiment:

```bash
modal run modal_app.py \
  --run-name act-driving-v1 \
  --max-steps 20000 \
  --batch-size 32
```

Watch progress in a second terminal:

```bash
modal app logs urban-vla-language-act-training
```

The logs report the current step, completion percentage, loss, learning rate, elapsed time, ETA, and validation metrics.

## Checkpoints and resume

Training writes to the persistent `urban-vla-act-artifacts` Modal Volume. To continue an interrupted run, use the same run name:

```bash
modal run modal_app.py --run-name act-driving-v1
```

With `resume: "auto"`, the trainer loads `last.pt` and restores the model, optimizer, scheduler, gradient scaler, step counter, and random-number states.

When training finishes, the local entrypoint downloads the run automatically. A manual download is also available:

```bash
modal volume get urban-vla-act-artifacts \
  runs/act-driving-v1 \
  artifacts/act-driving-v1
```

## Artifacts

```text
artifacts/act-driving-v1/
|-- best.pt                       # Best resumable validation checkpoint
|-- last.pt                       # Latest resumable checkpoint
|-- model.safetensors             # Inference weights
|-- config.json                   # Model architecture and normalization
|-- training_config.json          # Full run configuration
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

The completed run is also pushed to Hugging Face. Each run remains under `runs/<run-name>/`, while the best inference weights are copied to the repository root.

## Evaluation

The trainer selects the best checkpoint using validation mean action MAE. It reads the test split once, after model selection.

Reported metrics include:

- masked action L1 over the complete chunk
- throttle, brake, and steering MAE and RMSE
- steering-direction accuracy for meaningful turns
- braking classification accuracy
- simultaneous throttle-and-brake rate
- action MAE for each language intent

These are **open-loop imitation metrics**. A low error means the policy resembles the recorded expert on held-out frames. It does not prove that the vehicle can finish a route.

The next evaluation stage must run the trained policy inside the simulator and measure route success, collisions, off-road time, traffic-light violations, pedestrian violations, control smoothness, and recovery success. Until that closed-loop evaluation exists, the repository should not claim autonomous-driving performance.

## Inference API

After a checkpoint has been downloaded:

```python
import numpy as np

from urban_act.inference import ACTPolicy, RecedingHorizonController

policy = ACTPolicy("artifacts/act-driving-v1")
controller = RecedingHorizonController(policy, replan_interval=3)

camera_rgb = np.zeros((128, 128, 3), dtype=np.uint8)
vehicle_state = np.array([8.2, 0.03, 0.4, 0.0], dtype=np.float32)
instruction = "Turn left at the next intersection."

throttle, brake, steering = controller.act(
    camera_rgb,
    vehicle_state,
    instruction,
)
```

The zero image is only a shape example. Real inference must use the simulator's current RGB frame and synchronized vehicle state.

## Local verification

These checks stay on the CPU. They do not request a Modal GPU or start training.

```bash
python -m compileall modal_app.py src tests scripts
python -m pytest
```

The tests cover configuration validation, dataset split handling, failure-data exclusion, action bounds, deterministic inference, masked loss, and gradient flow through the ACT policy.

## Scope and safety

This is a simulator research project. The images, traffic behavior, vehicle dynamics, and expert controls are synthetic. A model trained here must not be connected to a real vehicle.
