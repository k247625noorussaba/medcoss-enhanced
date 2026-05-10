#!/usr/bin/env bash
# =============================================================================
# LOCAL DEBUG ONLY — not for final training / not for RunPod production
# Safe to delete or revert; does not replace run_ds.sh
#
# Functional smoke test (WSL2 / Git Bash): four downstream tasks, seed 0 only.
# Results are not meaningful.
#
# Expects debug SSL checkpoint from run_ssl_debug_local.sh:
#   ${OUTPUT_ROOT}/MedCoSS_Report_Xray_Path_DEBUG_2D_Path_1/checkpoint-0.pth
#
# TODO: If any downstream CLI flag is missing in your branch, fix invocation here
#       (do not edit Python in this phase unless unavoidable).
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=/mnt/c/D/4th/Thesis/data_small
DS_REPORT="${DATA_ROOT}/ds_report"
DS_XRAY="${DATA_ROOT}/ds_xray"
DS_PATH_CLS="${DATA_ROOT}/ds_pathology_cls"
DS_PATH_SEG="${DATA_ROOT}/ds_pathology_seg"

# Linux $HOME: avoids WSL copytree/permission failures on /mnt/c/... repo tree.
OUTPUT_ROOT=$HOME/medcoss_outputs_debug/output_dir
SNAPSHOT_ROOT=$HOME/medcoss_outputs_debug/snapshots
PRETRAINED_CKPT="${OUTPUT_ROOT}/MedCoSS_Report_Xray_Path_DEBUG_2D_Path_1/checkpoint-0.pth"

gpu_id=0
reload_from_pretrained=True
pretrained_path="${PRETRAINED_CKPT}"
exp_name='MedCoSS_Report_Xray_Path_DEBUG'

EPOCHS=1
BATCH=2
WORKERS=0
seed=0

# -----------------------------------------------------------------------------
# 1) PubMed20k (report classification)
# -----------------------------------------------------------------------------
task_id='1D_PubMed_DEBUG'
model_name='model'
data_path="${DS_REPORT}"
lr=0.0002
meid="_${exp_name}/seed_${seed}/lr_${lr}/"
path_id="${task_id}${meid}"
snapshot_dir="${SNAPSHOT_ROOT}/downstream/dim_1/${path_id}"
mkdir -p "${snapshot_dir}"

CUDA_VISIBLE_DEVICES=${gpu_id} python -u Downstream/Dim_1/PudMed20k/main.py \
  --arch='unified_vit' \
  --data_path="${data_path}" \
  --snapshot_dir="${snapshot_dir}" \
  --input_size=112 \
  --batch_size="${BATCH}" \
  --num_gpus=1 \
  --num_epochs="${EPOCHS}" \
  --start_epoch=0 \
  --learning_rate="${lr}" \
  --num_classes=5 \
  --num_workers="${WORKERS}" \
  --reload_from_pretrained="${reload_from_pretrained}" \
  --pretrained_path="${pretrained_path}" \
  --val_only=0 \
  --random_seed="${seed}" \
  --model_name="${model_name}"

# -----------------------------------------------------------------------------
# 2) Chest_XR (X-ray classification)
# -----------------------------------------------------------------------------
task_id='2D_ChestXR_DEBUG'
data_path="${DS_XRAY}"
lr=0.00005
meid="_${exp_name}/seed_${seed}/lr_${lr}/"
path_id="${task_id}${meid}"
snapshot_dir="${SNAPSHOT_ROOT}/downstream/dim_2/${path_id}"
mkdir -p "${snapshot_dir}"

CUDA_VISIBLE_DEVICES=${gpu_id} python -u Downstream/Dim_2/Chest_XR/main.py \
  --arch='unified_vit' \
  --data_path="${data_path}" \
  --snapshot_dir="${snapshot_dir}" \
  --input_size='224,224' \
  --batch_size="${BATCH}" \
  --num_gpus=1 \
  --num_epochs="${EPOCHS}" \
  --start_epoch=0 \
  --learning_rate="${lr}" \
  --num_classes=3 \
  --num_workers="${WORKERS}" \
  --reload_from_pretrained="${reload_from_pretrained}" \
  --pretrained_path="${pretrained_path}" \
  --val_only=0 \
  --random_seed="${seed}"

# -----------------------------------------------------------------------------
# 3) NCT_CRC_HE (pathology classification)
# -----------------------------------------------------------------------------
task_id='2D_NCT_CRC_HE_DEBUG'
data_path="${DS_PATH_CLS}"
lr=0.0001
meid="_${exp_name}/seed_${seed}/lr_${lr}/"
path_id="${task_id}${meid}"
snapshot_dir="${SNAPSHOT_ROOT}/downstream/dim_2/${path_id}"
mkdir -p "${snapshot_dir}"

CUDA_VISIBLE_DEVICES=${gpu_id} python -u Downstream/Dim_2/NCT_CRC_HE/main.py \
  --arch='unified_vit' \
  --data_path="${data_path}" \
  --snapshot_dir="${snapshot_dir}" \
  --input_size='224,224' \
  --batch_size="${BATCH}" \
  --num_gpus=1 \
  --num_epochs="${EPOCHS}" \
  --start_epoch=0 \
  --learning_rate="${lr}" \
  --num_classes=9 \
  --num_workers="${WORKERS}" \
  --reload_from_pretrained="${reload_from_pretrained}" \
  --pretrained_path="${pretrained_path}" \
  --val_only=0 \
  --random_seed="${seed}"

# -----------------------------------------------------------------------------
# 4) GlaS (pathology segmentation) — train + evaluate one seed
# -----------------------------------------------------------------------------
nnudata="${DS_PATH_SEG}"
task_id='Glas_DEBUG'
lr=0.0001
meid="_${exp_name}/seed_${seed}/lr_${lr}/"
path_id="${task_id}${meid}"
snapshot_dir="${SNAPSHOT_ROOT}/downstream/dim_2/${path_id}"
mkdir -p "${snapshot_dir}"

CUDA_VISIBLE_DEVICES=${gpu_id} python -u Downstream/Dim_2/Glas/train.py \
  --arch='unified_vit' \
  --data_dir="${nnudata}" \
  --snapshot_dir="${snapshot_dir}" \
  --nnUNet_preprocessed="${nnudata}" \
  --input_size='512,512' \
  --batch_size="${BATCH}" \
  --num_gpus=1 \
  --num_epochs="${EPOCHS}" \
  --start_epoch=0 \
  --learning_rate="${lr}" \
  --num_classes=1 \
  --num_workers="${WORKERS}" \
  --weight_std=False \
  --random_seed="${seed}" \
  --reload_from_pretrained="${reload_from_pretrained}" \
  --pretrained_path="${pretrained_path}" \
  --itrs_each_epoch=2

output_dir="${SNAPSHOT_ROOT}/downstream/dim_2/${path_id}prediction/"
mkdir -p "${output_dir}"
CUDA_VISIBLE_DEVICES=${gpu_id} python -u Downstream/Dim_2/Glas/evaluate.py \
  --arch='unified_vit' \
  --data_dir="${nnudata}" \
  --nnUNet_preprocessed="${nnudata}" \
  --reload_from_checkpoint=True \
  --checkpoint_path="${snapshot_dir}/checkpoint.pth" \
  --save_path="${output_dir}" \
  --input_size='512,512' \
  --batch_size=1 \
  --num_classes=1 \
  --num_gpus=1 \
  --FP16=False \
  --random_seed="${seed}" \
  --weight_std=False \
  --isHD=True

echo "Debug downstream done."
