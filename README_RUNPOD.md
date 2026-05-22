# RunPod Setup Notes for MedCoSS

## Purpose

This document records the RunPod environment, setup steps, folder structure, dependency installation, and operational rules for MedCoSS training/reproduction.

---

## RunPod Configuration

Chosen configuration:

- GPU: 4× A100 SXM, 80 GB each
- Total VRAM: 320 GB
- CPU: 96 vCPU
- RAM: 468 GB
- Container image: `pytorch/pytorch:1.11.0-cuda11.3-cudnn8-devel`
- Template: `runpod-torch-v240`

---

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

---

## SSH Setup

Local SSH key generated using:

```bash
ssh-keygen -t ed25519 -C "runpod"
```

Add the public key in:

**RunPod → Settings → SSH Public Keys**

### Normal RunPod SSH

This is for terminal access only.

```bash
ssh <pod-id>@ssh.runpod.io -i ~/.ssh/id_ed25519
```

RunPod may show a command like:

```bash
ssh crw2ownj4wbdf5-64412097@ssh.runpod.io -i ~/.ssh/id_ed25519
```

Important:

- This route uses RunPod's SSH proxy.
- It does **not** support SCP/SFTP.
- Use it only for terminal access.

### SSH over Exposed TCP

This supports SSH, SCP, and SFTP.

Example:

```bash
ssh root@154.54.102.51 -p 11528 -i ~/.ssh/id_ed25519
```

On Windows CMD, use:

```bat
ssh root@154.54.102.51 -p 11528 -i %USERPROFILE%\.ssh\id_ed25519
```

Use this method for uploading files with `scp`.

Example SCP upload from Windows CMD:

```bat
scp -P 11528 -i %USERPROFILE%\.ssh\id_ed25519 C:\D\4th\Thesis\data_small\data_small.tar.gz root@154.54.102.51:/tmp/data/
```

General format:

```bat
scp -P <PORT> -i %USERPROFILE%\.ssh\id_ed25519 <LOCAL_FILE_PATH> root@<POD_IP>:/tmp/data/
```

---

## Start Command Fix

If the container exits immediately, set the RunPod start command to:

```bash
sleep infinity
```

---

## SSH Server Setup Inside Pod

Some containers do not have an SSH server installed/running by default. If SSH over exposed TCP gives `Connection refused`, install and start `sshd`.

Run inside the RunPod web terminal:

```bash
apt update
apt install -y openssh-server

mkdir -p /run/sshd
ssh-keygen -A
/usr/sbin/sshd
```

If SSH asks for a password or gives `Permission denied`, add the local public key to the pod.

On Windows CMD, show the public key:

```bat
type %USERPROFILE%\.ssh\id_ed25519.pub
```

Copy the full line, then inside the pod:

```bash
mkdir -p /root/.ssh
chmod 700 /root/.ssh

echo "PASTE_FULL_PUBLIC_KEY_HERE" >> /root/.ssh/authorized_keys

chmod 600 /root/.ssh/authorized_keys
```

Restart SSH server:

```bash
pkill sshd
/usr/sbin/sshd
```

Then retry from Windows:

```bat
ssh root@154.54.102.51 -p 11528 -i %USERPROFILE%\.ssh\id_ed25519
```

---

## File Upload

### Upload Small Dataset Using SCP

Run this from Windows CMD, not inside the pod:

```bat
scp -P 11528 -i %USERPROFILE%\.ssh\id_ed25519 C:\D\4th\Thesis\data_small\data_small.tar.gz root@154.54.102.51:/tmp/data/
```

After upload, verify inside the pod:

```bash
cd /tmp/data
ls -lh data_small.tar.gz
```

Extract into `processed_small`:

```bash
cd /tmp/data

mkdir -p processed_small
tar -xzf data_small.tar.gz -C processed_small
```

Verify:

```bash
ls -lh processed_small
find processed_small -maxdepth 2 -type d | head
```

If the archive contains a top-level folder such as `data_small/`, the extracted structure may become:

```text
/tmp/data_small/
```

To inspect archive structure before extraction:

```bash
tar -tzf data_small.tar.gz | head
```

---

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

---

## Dataset Structure

### Pretraining Dataset

Expected full dataset path:

```text
/tmp/data/
```

Expected small dataset path:

```text
/tmp/data_small/
```

Pretraining structure:

```text
us_report/
  master.csv

us_xray/
  *.png

us_pathology/
  *.tif
```

### Downstream Dataset

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

---

## Critical Runtime Notes

### Do Not Upload Old `pretrain_data_list.json`

Check for old files:

```bash
find /tmp/data -name "pretrain_data_list.json"
find /tmp/data_small -name "pretrain_data_list.json"
```

Delete them if found:

```bash
find /tmp/data -name "pretrain_data_list.json" -delete
find /tmp/data_small -name "pretrain_data_list.json" -delete
```

Reason:

- They may contain local Windows paths.
- They prevent the dataloader from rescanning the RunPod dataset.
- They can cause silent path bugs.

---

## System Dependencies

Install system-level dependencies with `apt`, not `pip`.

```bash
apt update
apt install -y libgl1 libglib2.0-0 git curl openssh-server python3-pip rsync tree tmux
```

These are needed for:

- Git repository operations
- OpenCV runtime libraries
- SSH/SCP access
- Python package installation

---

## Python Version Issue

The selected container may come with:

```bash
Python 3.6.9
```

Python 3.6 is too old for several required packages, including:

- `tensorboard==2.14.0`
- `pandas==2.0.3`
- `matplotlib==3.7.5`
- `monai==1.3.0`

Therefore, install Python 3.8 and use a virtual environment.

---

## Install Python 3.8

Install helper tools:

```bash
apt update
apt install -y software-properties-common curl
```

Add the Deadsnakes PPA:

```bash
add-apt-repository ppa:deadsnakes/ppa -y
apt update
```

Install Python 3.8:

```bash
apt install -y python3.8 python3.8-dev python3.8-venv python3.8-distutils
```

Verify:

```bash
python3.8 --version
```

Expected output:

```text
Python 3.8.x
```

---

## Create Virtual Environment

Go to the MedCoSS repository:

```bash
cd /workspace/MedCoSS
```

Create virtual environment:

```bash
python3.8 -m venv venv
```

Activate it:

```bash
source venv/bin/activate
```

After activation, the terminal should show something like:

```text
(venv) root@...
```

Important:

- After activating the venv, use `python`, not system `python3`.
- `python` should point to the venv Python 3.8.

Check:

```bash
python --version
```

---

## Install Pip in Virtual Environment

Install pip inside the venv:

```bash
curl -sS https://bootstrap.pypa.io/pip/3.8/get-pip.py | python
```

Upgrade pip tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

---

## Python Requirements

Keep only Python packages in `requirements_runpod.txt`.

Do **not** put `apt`, `add-apt-repository`, or shell setup commands inside `requirements_runpod.txt`.

Recommended `requirements_runpod.txt`:

```txt
tensorboard==2.14.0
tensorboard-data-server==0.7.2
timm==0.4.12
pandas==2.0.3
nltk==3.9.1
transformers==4.30.2
opencv-python==4.8.1.78

pydicom==2.4.4
matplotlib==3.7.5
nibabel==5.2.1
batchgenerators==0.25.1
einops==0.8.1
SimpleITK==2.2.1
tensorboardX==2.6.2.2
monai==1.3.0
medpy==0.5.2
```

Note:

- `opencv-python==4.13.0.92` should not be used because it may not be available or compatible.
- Use `opencv-python==4.8.1.78` instead.

Install requirements:

```bash
cd /workspace/MedCoSS
source venv/bin/activate

python -m pip install -r requirements_runpod.txt
```

---

## NLTK Setup

Download required NLTK data:

```bash
python -c "import nltk; nltk.download('punkt')"
```

Optional:

```bash
python -c "import nltk; nltk.download('punkt_tab')"
```

---

## Installation Test

Run:

```bash
python -c "import torch, timm, pandas, cv2, transformers, monai; print('OK')"
```

If this prints:

```text
OK
```

then the main dependencies are installed correctly.

---

## First Run Validation

For full dataset:

```bash
ls /tmp/data/us_xray | wc -l
ls /tmp/data/us_pathology | wc -l
ls -lh /tmp/data/us_report/master.csv
find /tmp/data -maxdepth 2 -type d
```

For small dataset:

```bash
find /tmp/data_small -maxdepth 2 -type d
find /tmp/data_small -type f | wc -l
```

---

## Monitoring

Monitor GPU usage:

```bash
watch -n 2 nvidia-smi
```

Monitor logs:

```bash
tail -f /workspace/output_dir/<stage>/log.txt
```

Find checkpoints:

```bash
find /workspace/output_dir -name "checkpoint-*.pth"
```

---

## Expected Behavior

During training:

- Logs should print periodically.
- Checkpoints should be saved.
- Final checkpoint may look like:

```text
checkpoint-299.pth
```

Training should end with something like:

```text
Training time ...
```

---

## Copy Debug Outputs from `/tmp` to Persistent Storage

For debug runs, outputs may be written to local container disk first:

```bash
OUTPUT_ROOT=/tmp/output_dir
LOG_ROOT=/tmp/logs
```

This helps avoid slow or unstable checkpoint writes directly to the mounted `/workspace` volume.

After the SSL/debug run completes successfully, copy the outputs back to persistent storage:

```bash
rsync -ah --progress /tmp/output_dir/ /workspace/output_dir/
rsync -ah --progress /tmp/logs/ /workspace/logs/
rsync -ah --progress /tmp/snapshots/ /workspace/MedCoSS/snapshots/
```

Verify copied checkpoints:

```bash
find /workspace/output_dir -name "checkpoint-*.pth" -ls
```

Optional: verify that a checkpoint can be loaded:

```bash
python - <<'PY'
import torch

ckpt = "/workspace/output_dir/1D_text_RUNPOD_2GPU_DEBUG_1/checkpoint-0.pth"
print("Testing:", ckpt)
obj = torch.load(ckpt, map_location="cpu")
print("OK:", obj.keys() if isinstance(obj, dict) else type(obj))
PY
```

Once outputs are safely copied to `/workspace`, the `/tmp` copies may be removed if needed:

```bash
rm -rf /tmp/output_dir /tmp/logs
```

---

## Workflow Strategy

### Phase 1 — Small Dataset

Use this first:

```text
/tmp/data_small/
```

Goal:

- Run the full pipeline.
- Catch dataloader/path/dependency errors.
- Confirm pretraining, downstream fine-tuning, and evaluation work end-to-end.

### Phase 2 — Full Dataset

Use this after the small dataset pipeline works:

```text
/tmp/data/
```

Goal:

- Run full training.
- Save checkpoints and logs.
- Run downstream evaluation.

---

## Pod Usage Rules

When idle:

- **STOP** the pod.

Do not **TERMINATE** the pod until:

- Checkpoints are saved.
- Logs are saved.
- Code is committed.
- Important outputs are copied from `/workspace`.

Important:

- Stopping the pod preserves `/workspace`.
- Terminating the pod deletes `/workspace`.

---

## Git Workflow

```bash
cd /workspace/MedCoSS

git status
git log --oneline -5

git add .
git commit -m "clean run setup"
```

---

## Common Issues

### `python: command not found`

Use:

```bash
python3 --version
```

or, after activating the venv:

```bash
source /workspace/MedCoSS/venv/bin/activate
python --version
```

### `No matching distribution found for tensorboard==2.14.0`

Cause:

- System Python is too old, usually Python 3.6.

Fix:

- Install Python 3.8.
- Create and activate a venv.
- Reinstall requirements inside the venv.

### `ssh: connect to host ... Connection refused`

Cause:

- SSH server is not running inside the pod.
- `openssh-server` may not be installed.

Fix:

```bash
apt update
apt install -y openssh-server
mkdir -p /run/sshd
ssh-keygen -A
/usr/sbin/sshd
```

### SSH asks for password

Cause:

- Public key is not added to `/root/.ssh/authorized_keys`.

Fix:

```bash
mkdir -p /root/.ssh
chmod 700 /root/.ssh

echo "PASTE_FULL_PUBLIC_KEY_HERE" >> /root/.ssh/authorized_keys

chmod 600 /root/.ssh/authorized_keys
pkill sshd
/usr/sbin/sshd
```

### `ss: command not found`

This is only a missing network utility. It does not mean SSH failed.

Install net tools if needed:

```bash
apt install -y net-tools
netstat -tlnp | grep :22
```

Or simply test SSH from Windows.

---

## Useful Commands

Check current directory:

```bash
pwd
```

Check disk usage:

```bash
df -h
```

Check folder size:

```bash
du -sh /tmp/data/*
```

Check files:

```bash
ls -lh
```

Find files:

```bash
find /workspace -name "requirements_runpod.txt"
```

Check Python path:

```bash
which python
python --version
```

Check pip path:

```bash
which pip
python -m pip --version
```