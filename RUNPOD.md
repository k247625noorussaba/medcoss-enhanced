# RunPod Setup Notes for MedCoSS

## Purpose

This document records the RunPod environment and operational rules for MedCoSS training/reproduction.

## RunPod Configuration

Chosen configuration:

- GPU: 4× A100 SXM, 80 GB each
- Total VRAM: 320 GB
- CPU: 96 vCPU
- RAM: 468 GB
- Container image: `pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel`
- Template: `runpod-torch-v240`

## Storage Configuration

- Volume disk: `/workspace`
- `/workspace` size: 500 GB
- Container disk: 200 GB
- Total disk: 700 GB

Important:

- `/workspace` persists when the pod is stopped and restarted.
- `/workspace` is deleted if the pod is terminated.
- Container disk is temporary and should not store important data.
- Network volume was not used because the selected 4× A100 SXM configuration did not support it.

## SSH Setup

Local SSH key generated using:

```bash
ssh-keygen -t ed25519 -C "runpod"
```

Add key in:

**RunPod → Settings → SSH Public Keys**

Connect:

```bash
ssh <pod-id>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

## Start Command Fix

If container exits immediately:

```bash
sleep infinity
```

## Folder Structure

```text
/workspace/
  data/
    processed/
    processed_small/

  MedCoSS/
  output_dir/
  checkpoints/
  logs/
```

## Dataset Structure

### Pretraining (`/workspace/data/processed/`)

```text
us_report/
  master.csv

us_xray/
  *.png

us_pathology/
  *.tif
```

### Downstream (`/workspace/data/processed/`)

```text
ds_report/
  train.txt
  dev.txt
  test.txt

ds_xray/
  train/
    covid/
    normal/
    pneumonia/
  validation/
    covid/
    normal/
    pneumonia/

ds_pathology_cls/
  ADI/
  BACK/
  DEB/
  LYM/
  MUC/
  MUS/
  NORM/
  STR/
  TUM/

ds_pathology_seg/
  train/images
  train/labels
  test/images
  test/labels
```

## Critical Runtime Notes

### Do NOT upload old `pretrain_data_list.json`

```bash
find /workspace/data/processed -name "pretrain_data_list.json"
find /workspace/data/processed -name "pretrain_data_list.json" -delete
```

Reason:

- Contains local paths (Windows)
- Prevents dataloader from rescanning
- Causes silent bugs

## Dependencies

```bash
pip install nltk
python -c "import nltk; nltk.download('punkt')"
```

Optional:

```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

## First Run Validation

```bash
ls /workspace/data/processed/us_xray | wc -l
ls /workspace/data/processed/us_pathology | wc -l
ls -lh /workspace/data/processed/us_report/master.csv
find /workspace/data/processed -maxdepth 2 -type d
```

## Monitoring

```bash
watch -n 2 nvidia-smi
tail -f /workspace/output_dir/<stage>/log.txt
find /workspace/output_dir -name "checkpoint-*.pth"
```

## Expected Behavior

- Logs printed periodically
- Checkpoints saved
- Final checkpoint: `checkpoint-299.pth`
- Training ends with: `Training time ...`

## Workflow Strategy

**Phase 1 — Small dataset**

- Upload `processed_small`
- Run full pipeline
- Fix errors

**Phase 2 — Full dataset**

- Upload `processed`
- Run full training

## Pod Usage Rules

When idle:

- **STOP** the pod

Never **TERMINATE** the pod until:

- Checkpoints saved
- Logs saved
- Code committed

## Git Workflow

```bash
cd /workspace/MedCoSS
git status
git log --oneline -5
git add .
git commit -m "clean run setup"
```
