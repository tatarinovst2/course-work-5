#!/bin/bash

LR=1e-6
PER_DEVICE_TRAIN_BATCH_SIZE=4

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --data_path)
      DATA_PATH="$2"
      shift
      shift
      ;;
    --mm_tunable_parts)
      MM_TUNABLE_PARTS="$2"
      shift
      shift
      ;;
    --model_name_or_path)
      MODEL_NAME_OR_PATH="$2"
      shift
      shift
      ;;
    --learning_rate)
      LR="$2"
      shift
      shift
      ;;
    --per_device_train_batch_size)
      PER_DEVICE_TRAIN_BATCH_SIZE="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 --model_name_or_path <model_name_or_path> --data_path <data_path> --mm_tunable_parts <mm_tunable_parts> [--learning_rate <learning_rate>] [--per_device_train_batch_size <batch_size>]"
      exit 1
      ;;
  esac
done

if [ -z "$DATA_PATH" ] || [ -z "$MM_TUNABLE_PARTS" ] || [ -z "$MODEL_NAME_OR_PATH" ]; then
  echo "Usage: $0 --model_name_or_path <model_name_or_path> --data_path <data_path> --mm_tunable_parts <mm_tunable_parts> [--learning_rate <learning_rate>] [--per_device_train_batch_size <batch_size>]"
  exit 1
fi

IMAGE_FOLDER=$(dirname "$DATA_PATH")

python llava_qwen/train.py \
  --model_name_or_path "$MODEL_NAME_OR_PATH" \
  --version "qwen_1_5" \
  --data_path "$DATA_PATH" \
  --image_folder "$IMAGE_FOLDER" \
  --mm_tunable_parts="$MM_TUNABLE_PARTS" \
  --mm_vision_tower_lr="$LR" \
  --vision_tower "google/siglip-so400m-patch14-384" \
  --mm_projector_type mlp2x_gelu \
  --mm_vision_select_layer -2 \
  --mm_use_im_start_end False \
  --mm_use_im_patch_token False \
  --group_by_modality_length True \
  --image_aspect_ratio anyres_max_9 \
  --image_grid_pinpoints "(1x1),...,(6x6)" \
  --mm_patch_merge_type spatial_unpad \
  --bf16 True \
  --run_name "RUN_NAME" \
  --output_dir "./checkpoints/llava-qwen-finetune" \
  --num_train_epochs 20 \
  --per_device_train_batch_size "$PER_DEVICE_TRAIN_BATCH_SIZE" \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 1 \
  --evaluation_strategy "no" \
  --save_strategy "steps" \
  --save_steps 250 \
  --save_total_limit 21 \
  --learning_rate "$LR" \
  --weight_decay 0.0 \
  --warmup_ratio 0.03 \
  --lr_scheduler_type "constant" \
  --logging_steps 1 \
  --tf32 False \
  --gradient_checkpointing False \
  --dataloader_num_workers 1 \
  --lazy_preprocess True \
  --report_to none \
  --torch_compile False \
  --dataloader_drop_last True \
  --frames_upbound 32 \
  --attn_implementation sdpa \
  --trainer_mode="zo" \
  --zo_eps="1e-3" \
  --zo_num_directions 1 \
  --model_max_length 1792
