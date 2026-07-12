# Dataset collection

The collector generates simulator demonstrations only. It does not train ACT or a VLA.

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
