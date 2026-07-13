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
- runpod
- modal
---

# Action Chunking Transformer for Autonomous Driving

A language-conditioned deep-learning policy that predicts short sequences of driving controls from a front-camera image, vehicle state, and a written instruction.

The policy is built around an **Action Chunking Transformer (ACT)**. Instead of predicting one steering command at a time, it predicts the next 20 controls together. At the dataset rate of 10 Hz, each chunk covers two seconds of driving. The controller executes a few actions, observes the road again, and replans.

> **Project status:** `act-driving-v1` completed 10,000 steps, but closed-loop testing exposed longitudinal median collapse: it predicts almost no throttle or brake. The v2 training code now uses balanced activity and magnitude heads, critical-window weighting, and collapse gates. `act-driving-v2` has not been trained yet.

<!-- TRAINING_RESULTS_START -->
Latest completed run: `act-driving-v1`.

| Metric | Value |
| --- | ---: |
| Best validation mean action MAE | 0.06392 |
| Test mean action MAE | 0.06696 |
| Test throttle MAE | 0.09242 |
| Test brake MAE | 0.04452 |
| Test steering MAE | 0.06395 |
| Test brake accuracy | 91.65% |
| Test steering-direction accuracy | 65.64% |
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
       activity + magnitude heads for throttle/brake
                + steering head
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

The first run used plain masked L1. Because most longitudinal targets are zero, that objective rewarded a near-zero median policy even though aggregate MAE looked reasonable. V2 separates longitudinal activity from magnitude:

```text
activity = balanced_BCE(throttle_active, brake_active)
magnitude = SmoothL1(throttle, brake | target is active)
steering = weighted_SmoothL1(steering, extra weight on turns)
overlap = mean(P(throttle active) * P(brake active))

loss = activity + magnitude + 2 * steering + 0.25 * overlap
     + beta * KL(q(z|a) || N(0, I))
```

Throttle and brake positive weights are calculated from the training split and capped at `12`. Samples covering a stationary launch, braking, turning, throttle use, or recovery maneuver receive additional weight. The strongest applicable weight is used, so overlapping cases do not multiply into unstable loss values.

KL starts near zero and warms to `0.1` over 2,000 steps. Tail positions near episode boundaries are padded and excluded from every loss and metric. The original setup remains in [`configs/v1.json`](configs/v1.json) for reproduction only.

The default optimization setup is:

| Setting | Value |
| --- | ---: |
| Optimizer | Fused AdamW on CUDA |
| GPU | RunPod or Modal H100 |
| Batch size | 64 |
| Training steps | 10,000 |
| Policy learning rate | `1e-4` |
| ResNet learning rate | `1e-5` |
| Warmup | 500 steps |
| Schedule | Cosine decay |
| Precision | BF16 |
| Float32 matmul | TF32 enabled |
| Gradient clipping | `1.0` |
| KL weight | `0.1`, warmed over 2,000 steps |
| Validation interval | 500 steps |
| Checkpoint interval | 500 steps |

All defaults live in [`configs/base.json`](configs/base.json).

The H100 profile processes `64 x 10,000 = 640,000` training samples. This is the same sample budget as the earlier batch-32, 20,000-step setup. Warmup, validation, checkpointing, and final-test batch counts are scaled with the larger batch, so their sample coverage remains comparable. The learning rates stay conservative for the first real run; batch-size scaling alone is not a reason to double them before a loss curve exists.

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
|   |-- base.json                 # V2 balanced-objective defaults
|   `-- v1.json                   # Original objective for reproduction only
|-- src/urban_act/
|   |-- config.py                 # Validated training configuration
|   |-- data.py                   # Video streaming and action chunks
|   |-- model.py                  # Language-conditioned ACT model
|   |-- losses.py                 # Balanced activity, magnitude, steering, and KL losses
|   |-- metrics.py                # Activity, active-action, collapse, and open-loop metrics
|   |-- plots.py                  # Training and prediction figures
|   |-- checkpoints.py            # Resume and inference weights
|   |-- train.py                  # Training and evaluation loop
|   |-- inference.py              # Policy and receding-horizon control
|   `-- hub.py                    # Hugging Face model publishing
|-- tests/                        # CPU unit tests with dummy encoders
|-- scripts/runpod_bootstrap.sh  # Remote dependency and training bootstrap
|-- scripts/check_v2_objective.py # CPU collapse-regression check
|-- scripts/download_hf_run.py   # Copy a completed Hub run back locally
|-- inference_server.py          # Local HTTP health and WebSocket inference server
|-- runpod_main.py               # One-command trainer for a manually created Pod
|-- runpod_train.py              # Standalone persistent-volume trainer
|-- runpod_launcher.py           # RunPod REST lifecycle client
|-- modal_app.py                  # H100 training entrypoint
|-- requirements-local.txt        # Local Modal and Hub tools
|-- requirements-inference.txt    # Local model-serving environment
|-- requirements-runpod.txt       # Packages not supplied by the GPU image
|-- requirements.txt              # Full training environment
`-- README.md
```

The video loader decodes each episode sequentially and constructs future action chunks in memory. It does not extract hundreds of thousands of PNG files before training.

## Manual RunPod training

This is the simplest RunPod path. Create the GPU Pod in the RunPod dashboard, open its terminal, clone the repository, and execute one Python file. `runpod_main.py` handles Python dependencies, CUDA validation, persistent paths, training, resume, evaluation, logs, and the final Hugging Face upload. It does not create or terminate cloud resources.

### 1. Create the Pod

Use an on-demand H100 SXM or RTX PRO 6000 Pod with:

- image: `pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime`
- container disk: at least 30 GB
- Pod volume mounted at `/workspace`: 50 GB
- `HF_TOKEN`: a Hugging Face write token supplied as a RunPod secret/environment variable

Pod-local `/workspace` data survives stopping and restarting the same Pod. It is deleted when that Pod is terminated, so wait for the Hugging Face upload or download the artifacts before termination.

### 2. Clone the repository

In the RunPod web terminal:

```bash
apt-get update && apt-get install -y git tmux
cd /workspace
git clone https://github.com/Mayankpratapsingh022/Action_Chunking_Transformer_Autonomous_Driving.git
cd Action_Chunking_Transformer_Autonomous_Driving/act_training
```

The new manual entrypoint must be committed and pushed before cloning.

### 3. Verify and run a 2,000-step pilot

The dry run prints paths and arguments without installing packages or starting training:

```bash
python3 runpod_main.py --dry-run
```

Run the CPU objective check before spending GPU time:

```bash
PYTHONPATH=src python3 scripts/check_v2_objective.py
```

Start a persistent terminal session and launch a pilot without publishing it:

```bash
tmux new -s act-training

python3 runpod_main.py \
  --run-name act-driving-v2 \
  --max-steps 2000 \
  --batch-size 64 \
  --no-push-to-hub
```

Detach with `Ctrl+B`, then `D`. Reattach later with:

```bash
tmux attach -t act-training
```

The first invocation installs the non-Torch requirements and the CUDA 12.8 TorchVision wheel. It then verifies the GPU before downloading the dataset. The entrypoint disables RunPod's deprecated `HF_HUB_ENABLE_HF_TRANSFER` default and uses standard HTTP instead of Xet by default, avoiding optional-transfer and Xet token failures for this small dataset. Dataset downloads retry transient Hub failures and honor the resolver reset delay after a `429` response. Set `HF_HUB_DISABLE_XET=0` before launch to opt back into Xet. If the selected template does not already contain Torch 2.8, add `--install-pytorch`. On later invocations, `--skip-setup` avoids the pip checks.

After the pilot, inspect the gates:

```bash
jq '{best_step, validation: .validation.quality_gates, test: .test.quality_gates}' \
  /workspace/act-driving/artifacts/runs/act-driving-v2/metrics.json
```

If throttle, brake, startup, steering, and zero-baseline gates pass or are clearly improving, continue the same checkpoint to 10,000 steps and enable publishing:

```bash
tmux new -s act-training

python3 runpod_main.py \
  --skip-setup \
  --run-name act-driving-v2 \
  --max-steps 10000 \
  --batch-size 64
```

### 4. Monitor and resume

From another RunPod web terminal:

```bash
tail -F /workspace/act-driving/logs/act-driving-v2.log
watch -n 2 nvidia-smi
cat /workspace/act-driving/artifacts/status/act-driving-v2.json
```

Rerunning with the same run name resumes from `/workspace/act-driving/artifacts/runs/act-driving-v2/last.pt`. The trainer always saves local artifacts. It only promotes the model, tokenizer, metrics, history, and plots to Hugging Face when both validation and test quality gates pass.

## Automated RunPod API setup

RunPod Pods are the recommended path when the account already has RunPod credits. The launcher uses the official REST API and Python's standard library; it does not require the RunPod Python SDK. No GPU is created unless `launch --yes` is used.

### 1. Publish the code

The remote Pod clones the Git repository and branch configured in `.env`. Commit and push these files before launching. The default source is this repository's `main` branch.

### 2. Configure local credentials

Create a RunPod API key in the RunPod console, then prepare the ignored environment file:

```bash
test -f .env || cp .env.example .env
```

Set these values in `.env`:

```dotenv
HF_TOKEN=hf_your_write_token
RUNPOD_API_KEY=your_runpod_api_key
RUNPOD_NETWORK_VOLUME_ID=
RUNPOD_HF_SECRET_NAME=huggingface_token
RUNPOD_GPU_TYPE_IDS=NVIDIA H100 80GB HBM3
RUNPOD_GIT_REF=main
```

`RUNPOD_API_KEY` stays on the Mac and authenticates lifecycle API calls. It is never included in the Pod specification or saved state.

In the RunPod console's **Secrets** section, create a secret named `huggingface_token` whose value is the Hugging Face write token. The Pod receives the reference `{{ RUNPOD_SECRET_huggingface_token }}`, not the plaintext value from the local `.env`.

### 3. Create persistent storage

Choose a Secure Cloud data center with H100 availability. Network volumes are tied to one data center, so that selection determines which GPUs can be attached. Create a 50 GB volume through the launcher:

```bash
python3 runpod_launcher.py create-volume \
  --name act-driving-training \
  --size 50 \
  --data-center-id YOUR_DATA_CENTER_ID \
  --yes
```

Copy the returned ID into `RUNPOD_NETWORK_VOLUME_ID` in `.env`. Existing volumes can be inspected with:

```bash
python3 runpod_launcher.py list-volumes
```

Checkpoints, Hugging Face caches, persistent logs, and final artifacts are written below `/workspace/act-driving/` on this volume.

### 4. Inspect and launch

The dry run prints the exact Pod request and never contacts the billable create endpoint:

```bash
python3 runpod_launcher.py launch --dry-run
```

After checking the GPU, image, branch, batch size, and volume ID, launch the on-demand Pod:

```bash
python3 runpod_launcher.py launch --yes
```

The default is one `NVIDIA H100 80GB HBM3`, batch size 64, and 10,000 steps. A 12-hour watchdog stops a hung training command; override it with `--timeout-hours` only when necessary. The launcher refuses an accidental second active Pod unless `--allow-duplicate` is explicitly supplied. Spot capacity is available with `--spot`, but it is not recommended for the first training run.

### 5. Monitor and clean up

The launch response is stored in ignored `.runpod/last_pod.json`, without credentials. These commands use that saved Pod ID:

```bash
python3 runpod_launcher.py status
python3 runpod_launcher.py watch
python3 runpod_launcher.py logs
```

`watch` reports lifecycle status and hourly cost. `logs` prints the RunPod dashboard link; container logs carry the trainer's step, percent, component losses, learning rate, elapsed time, ETA, and validation events. The same output is retained at `/workspace/act-driving/logs/act-driving-v2.log`.

The training command exits after final evaluation and Hugging Face publishing, releasing the GPU. Verify that the Pod reaches `EXITED`, then delete the Pod record:

```bash
python3 runpod_launcher.py terminate --yes
```

RunPod network-volume Pods cannot be stopped and restarted in place. Termination does not delete the attached network volume.

### 6. Resume or download

To resume, launch again with the same run name, Git branch, and network volume. `resume: "auto"` loads `/workspace/act-driving/artifacts/runs/act-driving-v2/last.pt` with the optimizer, scheduler, scaler, step, and random-number states.

After a successful run, copy the published artifact set back to this machine:

```bash
python3 scripts/download_hf_run.py --run-name act-driving-v2
```

This creates `artifacts/act-driving-v2/`. Point `ACT_ARTIFACT_DIR` at this directory when starting the inference API.

## Modal setup

Run the setup from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-local.txt

hf auth login
modal setup

cp .env.example .env
# Edit .env and set HF_TOKEN. The Modal token fields are optional after modal setup.
```

The Hugging Face token needs write access to [`Mayank022/urban-vla-language-act`](https://huggingface.co/Mayank022/urban-vla-language-act). The local `.env` is ignored by both Git and the Hugging Face publishing script. At launch, only `HF_TOKEN` is transferred to the GPU container as an ephemeral Modal secret.

## Start Modal training

The following command is ready for the first run, but it has not been executed:

```bash
./scripts/start_h100_tmux.sh
```

The launcher reads `.env`, starts a detached `act-h100` tmux session, and writes output to `logs/act-driving-v2.log`. Only `HF_TOKEN` is passed to the GPU container; local Modal credentials are used only by the CLI.

Override the two settings most likely to change during an initial experiment:

```bash
./scripts/run_modal.sh \
  --run-name act-driving-v2 \
  --max-steps 10000 \
  --batch-size 64
```

Watch progress in a second terminal:

```bash
modal app logs urban-vla-language-act-training
```

The logs report the current step, completion percentage, loss, learning rate, elapsed time, ETA, and validation metrics.

## Modal checkpoints and resume

Training writes to the persistent `urban-vla-act-artifacts` Modal Volume. To continue an interrupted run, use the same run name:

```bash
./scripts/start_h100_tmux.sh
```

With `resume: "auto"`, the trainer loads `last.pt` and restores the model, optimizer, scheduler, gradient scaler, step counter, and random-number states.

When training finishes, the local entrypoint downloads the run automatically. A manual download is also available:

```bash
modal volume get urban-vla-act-artifacts \
  runs/act-driving-v2 \
  artifacts/act-driving-v2
```

## Artifacts

```text
artifacts/act-driving-v2/
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

The trainer selects the best checkpoint with a collapse-resistant validation score. It combines active-action MAE with throttle F1, brake F1, steering-direction accuracy, and stationary-launch throttle recall. It reads the test split once, after model selection.

Reported metrics include:

- throttle, brake, and steering MAE and RMSE
- active-target MAE for throttle, brake, and steering
- zero-action baseline MAE and relative improvement
- throttle and brake precision, recall, F1, accuracy, and confusion counts
- stationary-launch throttle recall and mean predicted throttle
- steering-direction accuracy for meaningful turns
- simultaneous throttle-and-brake rate
- action MAE for each language intent
- an explicit `collapse_detected` flag and per-gate pass/fail results

The default publication gates require at least 80% stationary-launch throttle recall, 55% throttle recall, 55% brake recall, 65% steering-direction accuracy, a 5% MAE improvement over the all-zero baseline for both throttle and brake, and no more than 1% simultaneous throttle/brake actions. A run that misses a gate is retained on the Pod for diagnosis but is not promoted to the Hugging Face model root.

These are **open-loop imitation metrics**. A low error means the policy resembles the recorded expert on held-out frames. It does not prove that the vehicle can finish a route.

The next evaluation stage must run the trained policy inside the simulator and measure route success, collisions, off-road time, traffic-light violations, pedestrian violations, control smoothness, and recovery success. Until that closed-loop evaluation exists, the repository should not claim autonomous-driving performance.

The first simulator smoke run connected successfully and produced MPS predictions in roughly 30 ms, but `act-driving-v1` remained stationary. Direct checks on held-out dataset frames confirmed that the throttle and brake heads predict values near zero even when the expert target is active. The prediction plots show this longitudinal median collapse as well. Steering retains partial signal, but this checkpoint is not a usable closed-loop driving policy. The runtime does not add heuristic throttle because that would mask the learned-policy result.

## Simulator inference

Download the completed artifact set:

```bash
python3 scripts/download_hf_run.py --run-name act-driving-v1
```

Create an inference environment when needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-inference.txt
```

From the repository root, start the WebSocket model server:

```bash
npm run inference
```

Start Vite in a second terminal with `npm run dev`, open the printed URL, and press `I`. The client sends the current camera image, instruction, and `[speed, steering, previous throttle, previous brake]` state. The server returns continuous `[throttle, brake, steering]` controls and keeps a three-step receding horizon before predicting a fresh 20-action chunk.

Readiness and loaded model details are available at `http://localhost:8000/health`.

From the repository root, `npm run smoke:inference` checks the browser-to-model protocol. `npm run eval:closed-loop` also requires nonzero vehicle movement; it intentionally remains failing for `act-driving-v1` until a corrected checkpoint replaces it.

## Python inference API

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

These checks stay on the CPU. They do not request a RunPod or Modal GPU and do not start training.

```bash
python -m compileall modal_app.py runpod_launcher.py runpod_main.py runpod_train.py src tests scripts
python -m pytest
PYTHONPATH=src python scripts/check_v2_objective.py
```

The tests cover configuration validation, dataset split handling, failure-data exclusion, critical-window weights, v1 checkpoint compatibility, bounded actions, deterministic inference, balanced loss, collapse detection, publication gates, and gradient flow through the ACT policy.

## Scope and safety

This is a simulator research project. The images, traffic behavior, vehicle dynamics, and expert controls are synthetic. A model trained here must not be connected to a real vehicle.
