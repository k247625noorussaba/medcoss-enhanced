#!/usr/bin/env bash
# =============================================================================
# MedCoSS SSL pretraining — 3 modalities (RunPod): Report → X-ray → Pathology
# =============================================================================

set -e

# --- Dataset & I/O roots (RunPod) ---
DATA_ROOT=/workspace/data/processed
US_REPORT="${DATA_ROOT}/us_report"
US_XRAY="${DATA_ROOT}/us_xray"
US_PATHOLOGY="${DATA_ROOT}/us_pathology"

OUTPUT_ROOT=/workspace/output_dir
LOG_ROOT=/workspace/logs

# Uni-Perceiver init for stage 1 (download per README; override if stored elsewhere)
UNI_PERCEIVER_CKPT="${UNI_PERCEIVER_CKPT:-/workspace/checkpoints/uni-perceiver-base-L12-H768-224size-torch-pretrained.pth}"

# --- Distributed ---
export CUDA_VISIBLE_DEVICES=0,1
DIST_LAUNCH="python -m torch.distributed.launch --nproc_per_node=2"

# =============================================================================
# Stage 1 — Report-only pretraining (single-modal MAE)
# =============================================================================
output_dir="${OUTPUT_ROOT}/1D_text_300"
log_dir="${LOG_ROOT}/1D_text_300"
mkdir -p "${output_dir}" "${log_dir}"

${DIST_LAUNCH} --master_port='29502' main_pretrain_single_modal.py \
--model "unified_vit" \
--batch_size 128 \
--num_workers 10 \
--norm_pix_loss \
--mask_ratio 0.75 \
--epochs 300 \
--warmup_epochs 40 \
--blr 1.5e-4 --weight_decay 0.05 \
--data_path "${US_REPORT}" \
--task_modality "1D_text" \
--load_current_pretrained_weight "${UNI_PERCEIVER_CKPT}" \
--output_dir="${output_dir}" \
--log_dir="${log_dir}"

# =============================================================================
# Buffer — Report (k-means subset for continual buffer)
# =============================================================================
CUDA_VISIBLE_DEVICES=0 python main_buffer_kmean.py \
--model "unified_vit" \
--num_workers 10 \
--norm_pix_loss \
--data_path "${US_REPORT}" \
--task_modality "1D_text" \
--load_current_pretrained_weight "${OUTPUT_ROOT}/1D_text_300/checkpoint-299.pth" \
--num_center 0.01 \
--buffer_ratio 0.05 \
--exp_name "kmean"

# =============================================================================
# Stage 2 — X-ray continual MedCoSS (report buffer + x-ray task)
# =============================================================================
output_dir="${OUTPUT_ROOT}/MedCoSS_Report_Xray_Path_buff_0.05_cen_0.01_2D_Xray_300"
log_dir="${LOG_ROOT}/MedCoSS_Report_Xray_Path_buff_0.05_cen_0.01_2D_Xray_300"
mkdir -p "${output_dir}" "${log_dir}"

${DIST_LAUNCH} --master_port='29361' main_pretrain_medcoss.py \
--model "unified_vit" \
--batch_size 128 \
--num_workers 10 \
--norm_pix_loss \
--mask_ratio 0.75 \
--epochs 300 \
--warmup_epochs 40 \
--blr 1.5e-4 --weight_decay 0.05 \
--task_modality "2D_xray" \
--load_current_pretrained_weight "${OUTPUT_ROOT}/1D_text_300/checkpoint-299.pth" \
--data_path_1D_text "${US_REPORT}" \
--data_path_2D_xray "${US_XRAY}" \
--output_dir="${output_dir}" \
--log_dir="${log_dir}" \
--num_center 0.01 \
--buffer_ratio 0.05 \
--exp_name "kmean" \
--mix_up 1

# =============================================================================
# Buffer — X-ray (k-means subset for continual buffer)
# =============================================================================
CUDA_VISIBLE_DEVICES=0 python main_buffer_kmean.py \
--model "unified_vit" \
--num_workers 10 \
--norm_pix_loss \
--data_path "${US_XRAY}" \
--task_modality "2D_xray" \
--load_current_pretrained_weight "${OUTPUT_ROOT}/MedCoSS_Report_Xray_Path_buff_0.05_cen_0.01_2D_Xray_300/checkpoint-299.pth" \
--num_center 0.01 \
--buffer_ratio 0.05 \
--exp_name "kmean"

# =============================================================================
# Stage 3 — Pathology continual MedCoSS (report + x-ray buffers + pathology task)
# =============================================================================
output_dir="${OUTPUT_ROOT}/MedCoSS_Report_Xray_Path_buff_0.05_cen_0.01_2D_Path_300"
log_dir="${LOG_ROOT}/MedCoSS_Report_Xray_Path_buff_0.05_cen_0.01_2D_Path_300"
mkdir -p "${output_dir}" "${log_dir}"

${DIST_LAUNCH} --master_port='29361' main_pretrain_medcoss.py \
--model "unified_vit" \
--batch_size 128 \
--num_workers 10 \
--norm_pix_loss \
--mask_ratio 0.75 \
--epochs 300 \
--warmup_epochs 40 \
--blr 1.5e-4 --weight_decay 0.05 \
--task_modality "2D_path" \
--load_current_pretrained_weight "${OUTPUT_ROOT}/MedCoSS_Report_Xray_Path_buff_0.05_cen_0.01_2D_Xray_300/checkpoint-299.pth" \
--data_path_1D_text "${US_REPORT}" \
--data_path_2D_xray "${US_XRAY}" \
--data_path_2D_path "${US_PATHOLOGY}" \
--output_dir="${output_dir}" \
--log_dir="${log_dir}" \
--num_center 0.01 \
--buffer_ratio 0.05 \
--exp_name "kmean" \
--mix_up 1
