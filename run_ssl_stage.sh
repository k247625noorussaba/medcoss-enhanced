#!/usr/bin/env bash
# =============================================================================
# MedCoSS SSL pretraining — 3 modalities (RunPod): Report → X-ray → Pathology
# =============================================================================

set -euo pipefail

# --- Run length ---
EPOCHS=1
LAST_EPOCH=$((EPOCHS - 1))
RUN_TAG="${EPOCHS}epoch"
if (( EPOCHS > 80 )); then WARMUP_EPOCHS=40; else WARMUP_EPOCHS=1; fi

# --- Dataset & I/O roots (RunPod) ---
DATA_ROOT=/tmp/data
US_REPORT="${DATA_ROOT}/us_report"
US_XRAY="${DATA_ROOT}/us_xray"
US_PATHOLOGY="${DATA_ROOT}/us_pathology"

OUTPUT_ROOT=/tmp/output_dir
LOG_ROOT=/tmp/logs

STAGE1_DIR="1D_text_${RUN_TAG}"
STAGE2_DIR="MedCoSS_Report_Xray_Path_buff_0.05_cen_0.01_2D_Xray_${RUN_TAG}"
STAGE3_DIR="MedCoSS_Report_Xray_Path_buff_0.05_cen_0.01_2D_Path_${RUN_TAG}"

# Uni-Perceiver init for stage 1 (download per README; override if stored elsewhere)
UNI_PERCEIVER_CKPT="${UNI_PERCEIVER_CKPT:-/tmp/checkpoints/uni-perceiver-base-L12-H768-224size-torch-pretrained.pth}"

# --- Distributed ---
export CUDA_VISIBLE_DEVICES=0,1
DIST_LAUNCH="python -m torch.distributed.launch --nproc_per_node=2"

STAGE="${1:-}"

echo "EPOCHS=${EPOCHS}"
echo "LAST_EPOCH=${LAST_EPOCH}"
echo "RUN_TAG=${RUN_TAG}"
echo "WARMUP_EPOCHS=${WARMUP_EPOCHS}"
echo "DATA_ROOT=${DATA_ROOT}"
echo "STAGE=${STAGE}"

case "$STAGE" in

  1)
    echo "Running Stage 1 — Report SSL"
    # =============================================================================
    # Stage 1 — Report-only pretraining (single-modal MAE)
    # =============================================================================
    output_dir="${OUTPUT_ROOT}/${STAGE1_DIR}"
    log_dir="${LOG_ROOT}/${STAGE1_DIR}"
    mkdir -p "${output_dir}" "${log_dir}"

    ${DIST_LAUNCH} --master_port='29502' main_pretrain_single_modal.py \
    --model "unified_vit" \
    --batch_size 128 \
    --num_workers 2 \
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

python - <<PY
import torch
f="${OUTPUT_ROOT}/${STAGE1_DIR}/checkpoint-${LAST_EPOCH}.pth"
torch.load(f, map_location="cpu")
print("OK:", f)
PY
    ;;

  b1)
    echo "Running Buffer 1 — Report buffer"
    # =============================================================================
    # Buffer — Report (k-means subset for continual buffer)
    # =============================================================================
    STAGE1_CKPT="${OUTPUT_ROOT}/${STAGE1_DIR}/checkpoint-${LAST_EPOCH}.pth"
    test -f "$STAGE1_CKPT" || { echo "Missing checkpoint: $STAGE1_CKPT"; exit 1; }

    CUDA_VISIBLE_DEVICES=0 python main_buffer_kmean.py \
    --model "unified_vit" \
    --num_workers 2 \
    --norm_pix_loss \
    --data_path "${US_REPORT}" \
    --task_modality "1D_text" \
    --load_current_pretrained_weight "$STAGE1_CKPT" \
    --num_center 0.001 \
    --buffer_ratio 0.05 \
    --exp_name "kmean"
    ;;

  2)
    echo "Running Stage 2 — X-ray continual SSL"
    # =============================================================================
    # Stage 2 — X-ray continual MedCoSS (report buffer + x-ray task)
    # =============================================================================
    output_dir="${OUTPUT_ROOT}/${STAGE2_DIR}"
    log_dir="${LOG_ROOT}/${STAGE2_DIR}"
    mkdir -p "${output_dir}" "${log_dir}"
    
    STAGE1_CKPT="${OUTPUT_ROOT}/${STAGE1_DIR}/checkpoint-${LAST_EPOCH}.pth"
    test -f "$STAGE1_CKPT" || { echo "Missing checkpoint: $STAGE1_CKPT"; exit 1; }

    ${DIST_LAUNCH} --master_port='29361' main_pretrain_medcoss.py \
    --model "unified_vit" \
    --batch_size 128 \
    --num_workers 2 \
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

python - <<PY
import torch
f="${OUTPUT_ROOT}/${STAGE2_DIR}/checkpoint-${LAST_EPOCH}.pth"
torch.load(f, map_location="cpu")
print("OK:", f)
PY
    ;;

  b2)
    echo "Running Buffer 2 — X-ray buffer"
    # =============================================================================
    # Buffer — X-ray (k-means subset for continual buffer)
    # =============================================================================
    STAGE2_CKPT="${OUTPUT_ROOT}/${STAGE2_DIR}/checkpoint-${LAST_EPOCH}.pth"
    test -f "$STAGE2_CKPT" || { echo "Missing checkpoint: $STAGE2_CKPT"; exit 1; }

    CUDA_VISIBLE_DEVICES=0 python main_buffer_kmean.py \
    --model "unified_vit" \
    --num_workers 2 \
    --norm_pix_loss \
    --data_path "${US_XRAY}" \
    --task_modality "2D_xray" \
    --load_current_pretrained_weight "$STAGE2_CKPT" \
    --num_center 0.01 \
    --buffer_ratio 0.05 \
    --exp_name "kmean"
    ;;

  3)
    echo "Running Stage 3 — Pathology continual SSL"
    # =============================================================================
    # Stage 3 — Pathology continual MedCoSS (report + x-ray buffers + pathology task)
    # =============================================================================
    output_dir="${OUTPUT_ROOT}/${STAGE3_DIR}"
    log_dir="${LOG_ROOT}/${STAGE3_DIR}"
    mkdir -p "${output_dir}" "${log_dir}"

    STAGE2_CKPT="${OUTPUT_ROOT}/${STAGE2_DIR}/checkpoint-${LAST_EPOCH}.pth"
    test -f "$STAGE2_CKPT" || { echo "Missing checkpoint: $STAGE2_CKPT"; exit 1; }

    ${DIST_LAUNCH} --master_port='29362' main_pretrain_medcoss.py \
    --model "unified_vit" \
    --batch_size 128 \
    --num_workers 2 \
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

python - <<PY
import torch
f="${OUTPUT_ROOT}/${STAGE3_DIR}/checkpoint-${LAST_EPOCH}.pth"
torch.load(f, map_location="cpu")
print("OK:", f)
PY
    ;;

  *)
    echo "Usage:"
    echo "  bash run_ssl_stage.sh 1"
    echo "  bash run_ssl_stage.sh b1"
    echo "  bash run_ssl_stage.sh 2"
    echo "  bash run_ssl_stage.sh b2"
    echo "  bash run_ssl_stage.sh 3"
    exit 1
    ;;
esac
