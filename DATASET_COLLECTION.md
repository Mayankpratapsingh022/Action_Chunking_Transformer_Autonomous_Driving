# Dataset collection

The collector generates simulator demonstrations only. It does not train the VLA policy.

## Human left-turn collection

The current SmolVLA experiment uses human `vla-urban-4` episodes from the repository's sibling `../left-turn-target/` folder. Start the simulator with `npm run dev`, click **Collect**, and choose that folder once. The collector locks a clear, low-traffic protected-left-turn scene, a green protected signal, and 128 x 128 observations. Collection runs at 2x wall-clock speed, but samples still describe the same normal-time dynamics and are recorded at 10 Hz in simulated time. Model inference and ordinary driving stay at 1x. Every accepted route is saved automatically as one independent `human-*.json` file.

The controls are latched to produce cleaner action chunks:

- Tap `Up` or `W` once to cruise at `12 m/s` while collection is armed.
- Hold `Left` or `A` while turning; release it to recenter smoothly.
- Hold `Right` or `D` only for correction; release it to recenter.
- Tap `Down` or `S` once for a smooth non-reversing stop.
- Hold `Space` for an emergency stop.

The collection turn target is automatically limited to `8 m/s`. Ordinary manual mode keeps the faster `24 m/s` cruise and `18 m/s` turn targets. Each frame stores the learned action `[target_speed_mps, target_steering]`, where negative steering means left and positive steering means right, plus the applied throttle/brake/steer diagnostics and route-relative progress, lateral error, and heading error. The converter rejects older raw-pedal schemas, so keep new recordings as a separate collection.

A collision or severe route departure invalidates the whole current attempt. Its samples and video are deleted, and the next attempt starts from the original spawn; there is no mid-route respawn or auto-alignment. The collector also requires at least 80 samples and correct position and heading in the destination lane before saving. Use **Reset** or `N` to discard a run early without disabling collection.

Review without writing the LeRobot dataset:

```bash
cd vla_training
python convert_dataset.py --dry-run
```

The current published release contains 204 curated episodes and 39,417 frames from 152 unique seeds. It is available at [`Mayank022/urban-vla-left-turn-cruise-human`](https://huggingface.co/datasets/Mayank022/urban-vla-left-turn-cruise-human). The converter keeps 21 seed-disjoint episodes as the final holdout and records all 118 rejections in `conversion_report.json`.

## Automated expert collection

## Requirements

- Node.js 22 or newer
- Google Chrome or Playwright Chromium
- FFmpeg with either `h264_videotoolbox` or `libx264`
- 20 GB of free disk space recommended

## Before the full run

```bash
npm run dataset:qualify -- --episodes-per-task 5 --workers 2
npm run dataset:benchmark -- --episodes 2
```

Qualification must reach at least 98%. The benchmark recommends one or two workers for
the current machine.

## Background collection

```bash
# Recommended while continuing to use the laptop.
npm run dataset:start -- --workers 1

# Live progress, percentage, rate, ETA, current task, and storage.
npm run dataset:status -- --watch

# Human-readable logs.
npm run dataset:logs -- --follow

# Finish the active episode and stop.
npm run dataset:stop
```

Starting again resumes automatically:

```bash
npm run dataset:start -- --workers 1
```

Completed episode IDs are read from SQLite and never regenerated. An episode interrupted
by a crash or forced stop restarts from its deterministic seed. Use
`npm run dataset:stop -- --force` only when cooperative stop is unresponsive.

## Output

```text
datasets/urban-vla-expert-v1/
├── manifests/
├── raw/accepted/
├── raw/failures/
├── raw/partial/
├── raw/rejected/
├── state/collection.sqlite
├── logs/
├── reports/
└── lerobot/
```

`raw/accepted` contains nominal and expert recovery demonstrations. `raw/failures`
contains analysis-only unsafe trajectories and is excluded from LeRobot conversion.

## Validation and conversion

```bash
npm run dataset:validate

# Optional, after installing LeRobot, NumPy, and OpenCV.
npm run dataset:convert
```

The raw accepted episodes remain the source of truth if the LeRobot API changes.
