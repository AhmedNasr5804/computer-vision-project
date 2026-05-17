# Edge-Deployable Driver Assistance Vision Project

[![Status](https://img.shields.io/badge/status-active-success)](./README.md)
[![Platform](https://img.shields.io/badge/platform-Android%20%7C%20Raspberry%20Pi-blue)](./README.md)
[![Framework](https://img.shields.io/badge/framework-TensorFlow%20%2F%20TFLite-orange)](./README.md)
[![Paper](https://img.shields.io/badge/report-IEEE%20style-lightgrey)](paper/paper.tex)

This repository contains a full computer vision term project implementing, comparing, and deploying two edge-AI pipelines:

- Eye-state classification (open vs. closed), targeted for Samsung Galaxy S24 Ultra.
- Lane-direction classification (left, right, straight, stop), targeted for Raspberry Pi 4.

The project is organized as reproducible experiment stages from data checks to final deployment artifacts and paper generation.

## Table of Contents

1. [Project Objectives](#project-objectives)
2. [High-Level Results](#high-level-results)
3. [Repository Layout](#repository-layout)
4. [Experiment Workflow](#experiment-workflow)
5. [Quick Start (Run Commands)](#quick-start-run-commands)
6. [Artifacts and Traceability](#artifacts-and-traceability)
7. [Reproducibility Notes](#reproducibility-notes)
8. [Typical Usage](#typical-usage)
9. [License and Academic Context](#license-and-academic-context)

## Project Objectives

- Build classical computer vision baselines for both tasks.
- Build and evaluate custom lightweight CNN baselines.
- Evaluate transfer learning variants (MobileNetV2/V3).
- Run controlled improved-experiment studies.
- Select winners using accuracy/F1/efficiency trade-offs.
- Export edge-ready deployment formats (TFLite, OpenVINO where applicable).
- Produce publication-ready figures and paper assets.

## High-Level Results

Based on the recorded summaries and artifacts:

- Eye winner: `lw_wide` family (edge-efficient and accurate).
- Lane winner: `lw_baseline` family.
- Eye PTQ int8 model: ~106 KB.
- Lane PTQ int8 model: ~98 KB.

See `paper/summary.md` and `artifacts/` JSON files for exact numeric metrics.

## Repository Layout

- `artifacts/`: all generated models, metrics JSON files, figures, deployment outputs, and manifests.
- `datasets/`: source datasets (CEW and TuSimple) used by the pipelines.
- `fine_tuning/`: phone/Pi domain adaptation datasets and captures.
- `notebooks/`: staged notebook pipelines for setup, eye track, lane track, and paper figures.
- `paper/`: IEEE paper source, tables, generated figures, and compiled outputs.
- `tools/`: helper/mobile benchmarking utilities.

Each directory includes a `DESCRIPTION.md` with role, contents, and usage notes.

## Experiment Workflow

### 1. Environment setup

Start with:

- `notebooks/00_setup_environment.ipynb`

This prepares dependencies, seeds, and reproducibility conventions.

### 2. Eye pipeline notebooks

In order:

1. `notebooks/eye/01_eye_data_exploration.ipynb`
2. `notebooks/eye/02_eye_classical_baseline.ipynb`
3. `notebooks/eye/03_eye_cnn_baseline.ipynb`
4. `notebooks/eye/04_eye_transfer_learning.ipynb`
5. `notebooks/eye/05_eye_improved_experiments.ipynb`
6. `notebooks/eye/06_eye_model_selection.ipynb`
7. `notebooks/eye/07_eye_finetune.ipynb`
8. `notebooks/eye/08_eye_mobile_deployment.ipynb`

### 3. Lane pipeline notebooks

In order:

1. `notebooks/lane/01_lane_data_exploration.ipynb`
2. `notebooks/lane/02_lane_classical_baseline.ipynb`
3. `notebooks/lane/03_lane_cnn_baseline.ipynb`
4. `notebooks/lane/04_lane_transfer_learning.ipynb`
5. `notebooks/lane/05_lane_improved_experiments.ipynb`
6. `notebooks/lane/06_lane_model_selection.ipynb`
7. `notebooks/lane/07_lane_finetune.ipynb`
8. `notebooks/lane/08_lane_rpi_deployment.ipynb`

### 4. Paper assets

- Generate/refresh figure assets with `notebooks/09_paper_figures.ipynb`.
- Build paper from `paper/paper.tex`.

## Quick Start (Run Commands)

### Option A: Run notebooks interactively in Jupyter/VS Code

1. Open `notebooks/00_setup_environment.ipynb` and run all cells.
2. Run the eye notebooks in numeric order (`01` to `08`).
3. Run the lane notebooks in numeric order (`01` to `08`).
4. Run `notebooks/09_paper_figures.ipynb`.

### Option B: Run all notebooks from PowerShell (headless execution)

From repository root:

```powershell
# 0) Setup
jupyter nbconvert --to notebook --execute --inplace notebooks/00_setup_environment.ipynb

# 1) Eye pipeline
jupyter nbconvert --to notebook --execute --inplace notebooks/eye/01_eye_data_exploration.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/eye/02_eye_classical_baseline.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/eye/03_eye_cnn_baseline.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/eye/04_eye_transfer_learning.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/eye/05_eye_improved_experiments.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/eye/06_eye_model_selection.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/eye/07_eye_finetune.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/eye/08_eye_mobile_deployment.ipynb

# 2) Lane pipeline
jupyter nbconvert --to notebook --execute --inplace notebooks/lane/01_lane_data_exploration.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/lane/02_lane_classical_baseline.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/lane/03_lane_cnn_baseline.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/lane/04_lane_transfer_learning.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/lane/05_lane_improved_experiments.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/lane/06_lane_model_selection.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/lane/07_lane_finetune.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/lane/08_lane_rpi_deployment.ipynb

# 3) Paper figures
jupyter nbconvert --to notebook --execute --inplace notebooks/09_paper_figures.ipynb
```

### Build the paper

```powershell
Set-Location paper
pdflatex paper.tex
pdflatex paper.tex
Set-Location ..
```

## Artifacts and Traceability

The `artifacts/` folder is the source of truth for:

- split definitions,
- quantitative metrics,
- model binaries,
- deployment packages,
- visual diagnostics.

For a complete per-file description, see:

- `artifacts/DESCRIPTION.md`

## Reproducibility Notes

- A global seed is used across notebooks (`seed = 42` in manifests/summary).
- Splits are persisted to JSON (`eye_split.json`, `lane_split.json`) to avoid leakage.
- Metrics and deployment measurements are stored as JSON/PNG under `artifacts/`.
- Paper figures are generated from experiment outputs, not manually drawn.

## Typical Usage

1. Verify data availability under `datasets/` and `fine_tuning/`.
2. Run notebooks sequentially for each track.
3. Inspect generated results in `artifacts/`.
4. Regenerate paper figures and compile `paper/paper.tex`.

## License and Academic Context

This repository appears to be a university term-project codebase with research/reporting outputs. Confirm dataset licenses and third-party usage terms before redistribution of data assets.
