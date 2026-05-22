#!/usr/bin/env bash
# =============================================================================
# MedCoSS SSL pretraining — STAGE-WISE RunPod script
#
# Modality order:
#   Stage 1: Report-only SSL pretraining
#   b1:      Build report rehearsal buffer using KMeans
#   Stage 2: X-ray continual SSL using report buffer
#   b2:      Build X-ray rehearsal buffer using KMeans
#   Stage 3: Pathology continual SSL using report + X-ray buffers
#
# Usage:
#   bash run_ssl_stage.sh 1      # Report SSL
#   bash run_ssl_stage.sh b1     # Report buffer
#   bash run_ssl_stage.sh 2      # X-ray continual SSL
#   bash run_ssl_stage.sh b2     # X-ray buffer
#   bash run_ssl_stage.sh 3      # Pathology continual SSL
#
# Recommended:
#   Run long stages inside tmux so SSH disconnects do not kill the job.
# =============================================================================

set -euo pipefail

# =============================================================================
# 1) Run configuration
# =============================================================================

# Full SSL run length.
# For quick debug, set EPOCHS=1.
# For full run, keep EPOCHS=300.
EPOCHS=300
LAST_EPOCH=$((EPOCHS - 1))
RUN_TAG="${EPOCHS}epoch"

# Warmup rule:
# - Full/long runs use 40 warmup epochs.
# - Short debug runs use 1 warmup epoch.
if (( EPOCHS > 80 )); then
  WARMUP_EPOCHS=40
else
  WARMUP_EPOCHS=1
fi

# =============================================================================
# 2) Dataset and output roots
# =============================================================================

# Data is read from /tmp for faster local container access.
DATA_ROOT=/tmp/data
US_REPORT="${DATA_ROOT}/us_report"
US_XRAY="${DATA_ROOT}/us_xray"
US_PATHOLOGY="${DATA_ROOT}/us_pathology"

# SSL outputs/logs are written to /tmp first.
# After each successful stage, copy/rsync important results to /workspace.
OUTPUT_ROOT=/tmp/output_dir
LOG_ROOT=/tmp/logs

# Stage output folder names.
STAGE1_DIR="1D_text_${RUN_TAG}"
STAGE2_DIR="MedCoSS_Report_Xray_Path_buff_0.05_cen_0.01_2D_Xray_${RUN_TAG}"
STAGE3_DIR="MedCoSS_Report_Xray_Path_buff_0.05_cen_0.01_2D_Path_${RUN_TAG}"

# Uni-Perceiver initialization checkpoint for Stage 1.
# Can be overridden from environment:
#   UNI_PERCEIVER_CKPT=/path/to/file bash run_ssl_stage.sh 1
UNI_PERCEIVER_CKPT="${UNI_PERCEIVER_CKPT:-/tmp/checkpoints/uni-perceiver-base-L12-H768-224size-torch-pretrained.pth}"

# =============================================================================
# 3) Distributed training setup
# =============================================================================

# Use two A100 GPUs.
export CUDA_VISIBLE_DEVICES=0,1

# Original MedCoSS uses torch.distributed.launch.
DIST_LAUNCH="python -m torch.distributed.launch --nproc_per_node=2"

# Selected stage from command line.
STAGE="${1:-}"

# Print config for logging/debugging.
echo "============================================================"
echo "MedCoSS SSL stage runner"
echo "============================================================"
echo "EPOCHS=${EPOCHS}"
echo "LAST_EPOCH=${LAST_EPOCH}"
echo "RUN_TAG=${RUN_TAG}"
echo "WARMUP_EPOCHS=${WARMUP_EPOCHS}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "OUTPUT_ROOT=${OUTPUT_ROOT}"
echo "LOG_ROOT=${LOG_ROOT}"
echo "STAGE1_DIR=${STAGE1_DIR}"
echo "STAGE2_DIR=${STAGE2_DIR}"
echo "STAGE3_DIR=${STAGE3_DIR}"
echo "STAGE=${STAGE}"
echo "============================================================"

# =============================================================================
# 4) Helper function: verify checkpoint can be loaded
# =============================================================================

verify_checkpoint() {
  local ckpt_path="$1"

  test -f "$ckpt_path" || {
    echo "Missing checkpoint: $ckpt_path"
    exit 1
  }

  python - <<PY
import torch
f = "${ckpt_path}"
torch.load(f, map_location="cpu")
print("OK:", f)
PY
}

# =============================================================================
# 5) Stage selector
# =============================================================================

case "$STAGE" in

  # ---------------------------------------------------------------------------
  # Stage 1 — Report-only SSL pretraining
  # ---------------------------------------------------------------------------
  1)
    echo "Running Stage 1 — Report SSL"

    output_dir="${OUTPUT_ROOT}/${STAGE1_DIR}"
    log_dir="${LOG_ROOT}/${STAGE1_DIR}"
    mkdir -p "${output_dir}" "${log_dir}"

    ${DIST_LAUNCH} --master_port='29502' main_pretrain_single_modal.py \
      --model "unified_vit" \
      --batch_size 128 \
      --num_workers 10 \
      --norm_pix_loss \
      --mask_ratio 0.75 \
      --epochs "${EPOCHS}" \
      --warmup_epochs "${WARMUP_EPOCHS}" \
      --blr 1.5e-4 --weight_decay 0.05 \
      --data_path "${US_REPORT}" \
      --task_modality "1D_text" \
      --load_current_pretrained_weight "${UNI_PERCEIVER_CKPT}" \
      --output_dir="${output_dir}" \
      --log_dir="${log_dir}"

    STAGE1_CKPT="${OUTPUT_ROOT}/${STAGE1_DIR}/checkpoint-${LAST_EPOCH}.pth"
    verify_checkpoint "$STAGE1_CKPT"
    ;;

  # ---------------------------------------------------------------------------
  # Buffer 1 — Build report rehearsal buffer
  # ---------------------------------------------------------------------------
  b1)
    echo "Running Buffer 1 — Report buffer"

    STAGE1_CKPT="${OUTPUT_ROOT}/${STAGE1_DIR}/checkpoint-${LAST_EPOCH}.pth"
    verify_checkpoint "$STAGE1_CKPT"

    CUDA_VISIBLE_DEVICES=0 python main_buffer_kmean.py \
      --model "unified_vit" \
      --num_workers 10 \
      --norm_pix_loss \
      --data_path "${US_REPORT}" \
      --task_modality "1D_text" \
      --load_current_pretrained_weight "$STAGE1_CKPT" \
      --num_center 0.01 \
      --buffer_ratio 0.05 \
      --exp_name "kmean"

    REPORT_BUFFER="${OUTPUT_ROOT}/${STAGE1_DIR}/1D_text_0.01_0.05_kmean.csv"
    test -f "$REPORT_BUFFER" || {
      echo "Missing report buffer: $REPORT_BUFFER"
      exit 1
    }
    echo "OK: $REPORT_BUFFER"
    ;;

  # ---------------------------------------------------------------------------
  # Stage 2 — X-ray continual SSL using report buffer
  # ---------------------------------------------------------------------------
  2)
    echo "Running Stage 2 — X-ray continual SSL"

    output_dir="${OUTPUT_ROOT}/${STAGE2_DIR}"
    log_dir="${LOG_ROOT}/${STAGE2_DIR}"
    mkdir -p "${output_dir}" "${log_dir}"

    STAGE1_CKPT="${OUTPUT_ROOT}/${STAGE1_DIR}/checkpoint-${LAST_EPOCH}.pth"
    REPORT_BUFFER="${OUTPUT_ROOT}/${STAGE1_DIR}/1D_text_0.01_0.05_kmean.csv"

    verify_checkpoint "$STAGE1_CKPT"
    test -f "$REPORT_BUFFER" || {
      echo "Missing report buffer: $REPORT_BUFFER"
      exit 1
    }

    ${DIST_LAUNCH} --master_port='29361' main_pretrain_medcoss.py \
      --model "unified_vit" \
      --batch_size 128 \
      --num_workers 10 \
      --norm_pix_loss \
      --mask_ratio 0.75 \
      --epochs "${EPOCHS}" \
      --warmup_epochs "${WARMUP_EPOCHS}" \
      --blr 1.5e-4 --weight_decay 0.05 \
      --task_modality "2D_xray" \
      --load_current_pretrained_weight "$STAGE1_CKPT" \
      --data_path_1D_text "${US_REPORT}" \
      --data_path_2D_xray "${US_XRAY}" \
      --output_dir="${output_dir}" \
      --log_dir="${log_dir}" \
      --num_center 0.01 \
      --buffer_ratio 0.05 \
      --exp_name "kmean" \
      --mix_up 1

    STAGE2_CKPT="${OUTPUT_ROOT}/${STAGE2_DIR}/checkpoint-${LAST_EPOCH}.pth"
    verify_checkpoint "$STAGE2_CKPT"
    ;;

  # ---------------------------------------------------------------------------
  # Buffer 2 — Build X-ray rehearsal buffer
  # ---------------------------------------------------------------------------
  b2)
    echo "Running Buffer 2 — X-ray buffer"

    STAGE2_CKPT="${OUTPUT_ROOT}/${STAGE2_DIR}/checkpoint-${LAST_EPOCH}.pth"
    verify_checkpoint "$STAGE2_CKPT"

    CUDA_VISIBLE_DEVICES=0 python main_buffer_kmean.py \
      --model "unified_vit" \
      --num_workers 10 \
      --norm_pix_loss \
      --data_path "${US_XRAY}" \
      --task_modality "2D_xray" \
      --load_current_pretrained_weight "$STAGE2_CKPT" \
      --num_center 0.01 \
      --buffer_ratio 0.05 \
      --exp_name "kmean"

    XRAY_BUFFER="${OUTPUT_ROOT}/${STAGE2_DIR}/2D_xray_0.01_0.05_kmean.json"
    test -f "$XRAY_BUFFER" || {
      echo "Missing X-ray buffer: $XRAY_BUFFER"
      exit 1
    }
    echo "OK: $XRAY_BUFFER"
    ;;

  # ---------------------------------------------------------------------------
  # Stage 3 — Pathology continual SSL using report + X-ray buffers
  # ---------------------------------------------------------------------------
  3)
    echo "Running Stage 3 — Pathology continual SSL"

    output_dir="${OUTPUT_ROOT}/${STAGE3_DIR}"
    log_dir="${LOG_ROOT}/${STAGE3_DIR}"
    mkdir -p "${output_dir}" "${log_dir}"

    STAGE2_CKPT="${OUTPUT_ROOT}/${STAGE2_DIR}/checkpoint-${LAST_EPOCH}.pth"
    REPORT_BUFFER_IN_STAGE2="${OUTPUT_ROOT}/${STAGE2_DIR}/1D_text_0.01_0.05_kmean.csv"
    XRAY_BUFFER="${OUTPUT_ROOT}/${STAGE2_DIR}/2D_xray_0.01_0.05_kmean.json"

    verify_checkpoint "$STAGE2_CKPT"

    test -f "$REPORT_BUFFER_IN_STAGE2" || {
      echo "Missing copied report buffer in Stage 2 output: $REPORT_BUFFER_IN_STAGE2"
      exit 1
    }

    test -f "$XRAY_BUFFER" || {
      echo "Missing X-ray buffer: $XRAY_BUFFER"
      exit 1
    }

    ${DIST_LAUNCH} --master_port='29362' main_pretrain_medcoss.py \
      --model "unified_vit" \
      --batch_size 128 \
      --num_workers 10 \
      --norm_pix_loss \
      --mask_ratio 0.75 \
      --epochs "${EPOCHS}" \
      --warmup_epochs "${WARMUP_EPOCHS}" \
      --blr 1.5e-4 --weight_decay 0.05 \
      --task_modality "2D_path" \
      --load_current_pretrained_weight "$STAGE2_CKPT" \
      --data_path_1D_text "${US_REPORT}" \
      --data_path_2D_xray "${US_XRAY}" \
      --data_path_2D_path "${US_PATHOLOGY}" \
      --output_dir="${output_dir}" \
      --log_dir="${log_dir}" \
      --num_center 0.01 \
      --buffer_ratio 0.05 \
      --exp_name "kmean" \
      --mix_up 1

    STAGE3_CKPT="${OUTPUT_ROOT}/${STAGE3_DIR}/checkpoint-${LAST_EPOCH}.pth"
    verify_checkpoint "$STAGE3_CKPT"
    ;;

  # ---------------------------------------------------------------------------
  # Invalid/missing argument
  # ---------------------------------------------------------------------------
  *)
    echo "Usage:"
    echo "  bash run_ssl_stage.sh 1      # Stage 1: report SSL"
    echo "  bash run_ssl_stage.sh b1     # Buffer 1: report buffer"
    echo "  bash run_ssl_stage.sh 2      # Stage 2: X-ray continual SSL"
    echo "  bash run_ssl_stage.sh b2     # Buffer 2: X-ray buffer"
    echo "  bash run_ssl_stage.sh 3      # Stage 3: pathology continual SSL"
    exit 1
    ;;
esac
