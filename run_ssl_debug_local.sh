#!/usr/bin/env bash
# =============================================================================
# LOCAL DEBUG ONLY — not for final training / not for RunPod production
# Safe to delete or revert; does not replace run_ssl.sh
#
# Functional smoke test (WSL2 / Git Bash): 3-modality SSL chain with tiny settings.
# Results are not meaningful.
#
# Prerequisites:
#   - CUDA + PyTorch env from repo README
#   - Small data under DATA_ROOT (same folder names as production)
#   - Uni-Perceiver init weights (set UNI_PERCEIVER_CKPT or place file at default path)
#   - Run from repo root OR any cwd: script cds to its directory (repo root)
#
# NOTE: main_buffer_kmean.py hardcodes DataLoader batch_size=128 (no CLI flag).
# NOTE: Local debug buffer/k-means uses reduced NUM_CENTER / BUFFER_RATIO below so tiny
#       datasets do not request one center per sample (CUDA index issues). Production
#       run_ssl.sh (RunPod) keeps the original buffer hyperparameters unchanged.
# =============================================================================

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${REPO_ROOT}"

export CUDA_VISIBLE_DEVICES=0

DATA_ROOT=/mnt/c/D/4th/Thesis/data_small
US_REPORT="${DATA_ROOT}/us_report"
US_XRAY="${DATA_ROOT}/us_xray"
US_PATHOLOGY="${DATA_ROOT}/us_pathology"

OUTPUT_ROOT=$HOME/medcoss_outputs_debug/output_dir
LOG_ROOT=$HOME/medcoss_outputs_debug/logs

# Override: export UNI_PERCEIVER_CKPT=/path/to/uni-perceiver-base-L12-H768-224size-torch-pretrained.pth
UNI_PERCEIVER_CKPT="${UNI_PERCEIVER_CKPT:-$HOME/medcoss_outputs_debug/uni-perceiver-base-L12-H768-224size-torch-pretrained.pth}"

EPOCHS=1
BATCH=2
WORKERS=0
WARMUP=1

# Local debug only: gentler k-means buffer than 1.0/1.0 (too many centers for tiny N).
# run_ssl.sh is unchanged and still uses the full training buffer settings.
NUM_CENTER=0.1
BUFFER_RATIO=0.1

OUT_1D="${OUTPUT_ROOT}/1D_text_DEBUG_1"
LOG_1D="${LOG_ROOT}/1D_text_DEBUG_1"
OUT_XRAY="${OUTPUT_ROOT}/MedCoSS_Report_Xray_Path_DEBUG_2D_Xray_1"
LOG_XRAY="${LOG_ROOT}/MedCoSS_Report_Xray_Path_DEBUG_2D_Xray_1"
OUT_PATH="${OUTPUT_ROOT}/MedCoSS_Report_Xray_Path_DEBUG_2D_Path_1"
LOG_PATH="${LOG_ROOT}/MedCoSS_Report_Xray_Path_DEBUG_2D_Path_1"

mkdir -p "${OUT_1D}" "${LOG_1D}" "${OUT_XRAY}" "${LOG_XRAY}" "${OUT_PATH}" "${LOG_PATH}"

# -----------------------------------------------------------------------------
# Stage 1 — Report-only pretraining (1 epoch → checkpoint-0.pth)
# -----------------------------------------------------------------------------
python3 main_pretrain_single_modal.py \
  --model "unified_vit" \
  --batch_size "${BATCH}" \
  --num_workers "${WORKERS}" \
  --norm_pix_loss \
  --mask_ratio 0.75 \
  --epochs "${EPOCHS}" \
  --warmup_epochs "${WARMUP}" \
  --blr 1.5e-4 --weight_decay 0.05 \
  --data_path "${US_REPORT}" \
  --task_modality "1D_text" \
  --load_current_pretrained_weight "${UNI_PERCEIVER_CKPT}" \
  --output_dir="${OUT_1D}" \
  --log_dir="${LOG_1D}"

# -----------------------------------------------------------------------------
# Stage 2 — Report buffer (k-means; reads checkpoint-0.pth from stage 1)
# -----------------------------------------------------------------------------
python3 main_buffer_kmean.py \
  --model "unified_vit" \
  --num_workers "${WORKERS}" \
  --norm_pix_loss \
  --data_path "${US_REPORT}" \
  --task_modality "1D_text" \
  --load_current_pretrained_weight "${OUT_1D}/checkpoint-0.pth" \
  --num_center "${NUM_CENTER}" \
  --buffer_ratio "${BUFFER_RATIO}" \
  --exp_name "kmean"

# -----------------------------------------------------------------------------
# Stage 3 — X-ray continual MedCoSS (1 epoch)
# -----------------------------------------------------------------------------
python3 main_pretrain_medcoss.py \
  --model "unified_vit" \
  --batch_size "${BATCH}" \
  --num_workers "${WORKERS}" \
  --norm_pix_loss \
  --mask_ratio 0.75 \
  --epochs "${EPOCHS}" \
  --warmup_epochs "${WARMUP}" \
  --blr 1.5e-4 --weight_decay 0.05 \
  --task_modality "2D_xray" \
  --load_current_pretrained_weight "${OUT_1D}/checkpoint-0.pth" \
  --data_path_1D_text "${US_REPORT}" \
  --data_path_2D_xray "${US_XRAY}" \
  --output_dir="${OUT_XRAY}" \
  --log_dir="${LOG_XRAY}" \
  --num_center "${NUM_CENTER}" \
  --buffer_ratio "${BUFFER_RATIO}" \
  --exp_name "kmean" \
  --mix_up 1

# -----------------------------------------------------------------------------
# Stage 4 — X-ray buffer (k-means)
# -----------------------------------------------------------------------------
python3 main_buffer_kmean.py \
  --model "unified_vit" \
  --num_workers "${WORKERS}" \
  --norm_pix_loss \
  --data_path "${US_XRAY}" \
  --task_modality "2D_xray" \
  --load_current_pretrained_weight "${OUT_XRAY}/checkpoint-0.pth" \
  --num_center "${NUM_CENTER}" \
  --buffer_ratio "${BUFFER_RATIO}" \
  --exp_name "kmean"

# -----------------------------------------------------------------------------
# Stage 5 — Pathology continual MedCoSS (1 epoch → final debug checkpoint)
# -----------------------------------------------------------------------------
python3 main_pretrain_medcoss.py \
  --model "unified_vit" \
  --batch_size "${BATCH}" \
  --num_workers "${WORKERS}" \
  --norm_pix_loss \
  --mask_ratio 0.75 \
  --epochs "${EPOCHS}" \
  --warmup_epochs "${WARMUP}" \
  --blr 1.5e-4 --weight_decay 0.05 \
  --task_modality "2D_path" \
  --load_current_pretrained_weight "${OUT_XRAY}/checkpoint-0.pth" \
  --data_path_1D_text "${US_REPORT}" \
  --data_path_2D_xray "${US_XRAY}" \
  --data_path_2D_path "${US_PATHOLOGY}" \
  --output_dir="${OUT_PATH}" \
  --log_dir="${LOG_PATH}" \
  --num_center "${NUM_CENTER}" \
  --buffer_ratio "${BUFFER_RATIO}" \
  --exp_name "kmean" \
  --mix_up 1

echo "Debug SSL done. Final checkpoint (expected): ${OUT_PATH}/checkpoint-0.pth"