#!/usr/bin/env bash
set -euo pipefail

echo "== Updating apt =="
apt update

echo "== Installing system packages =="
apt install -y libgl1 libglib2.0-0 git curl

echo "== Installing Python requirements =="
pip install -r requirements_runpod.txt

echo "== Downloading NLTK punkt =="
python - <<'PY'
import nltk
nltk.download("punkt")
PY

echo "== Verifying key imports =="
python - <<'PY'
mods = [
    "torch", "tensorboard", "timm", "pandas", "nltk",
    "transformers", "cv2", "pydicom", "matplotlib",
    "nibabel", "batchgenerators", "einops", "SimpleITK",
    "tensorboardX", "monai", "medpy"
]
for m in mods:
    __import__(m)
    print("OK:", m)
PY

echo "RunPod setup complete."
