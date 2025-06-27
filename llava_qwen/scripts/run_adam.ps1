param (
    [Parameter(Mandatory=$true)]
    [string]$model_name_or_path,

    [Parameter(Mandatory=$true)]
    [string]$data_path,

    [Parameter(Mandatory=$true)]
    [string]$mm_tunable_parts,

    [Parameter(Mandatory=$false)]
    [string]$lr = "1e-6",

    [Parameter(Mandatory=$false)]
    [string]$per_device_train_batch_size = "1"
)

$data_path_full = Resolve-Path $data_path
if (-not $data_path_full) {
    Write-Host "Data path does not exist: $data_path"
    exit 1
}

$image_folder = Split-Path $data_path_full.Path -Parent

python llava_qwen/train.py `
  --model_name_or_path $model_name_or_path `
  --version "qwen_1_5" `
  --data_path $data_path `
  --image_folder $image_folder `
  --mm_tunable_parts=$mm_tunable_parts `
  --mm_vision_tower_lr=$lr `
  --vision_tower "google/siglip-so400m-patch14-384" `
  --mm_projector_type mlp2x_gelu `
  --mm_vision_select_layer -2 `
  --mm_use_im_start_end False `
  --mm_use_im_patch_token False `
  --group_by_modality_length True `
  --image_aspect_ratio anyres_max_9 `
  --image_grid_pinpoints "(1x1),...,(6x6)" `
  --mm_patch_merge_type spatial_unpad `
  --bf16 True `
  --run_name "RUN_NAME" `
  --output_dir "./checkpoints/llava-qwen-finetune" `
  --num_train_epochs 3 `
  --per_device_train_batch_size $per_device_train_batch_size `
  --per_device_eval_batch_size 1 `
  --gradient_accumulation_steps 1 `
  --evaluation_strategy "no" `
  --save_strategy "steps" `
  --save_steps 1000 `
  --save_total_limit 21 `
  --learning_rate $lr `
  --weight_decay 0.0 `
  --warmup_ratio 0.03 `
  --lr_scheduler_type "constant" `
  --logging_steps 1 `
  --tf32 False `
  --gradient_checkpointing False `
  --dataloader_num_workers 1 `
  --lazy_preprocess True `
  --report_to none `
  --torch_compile False `
  --dataloader_drop_last True `
  --frames_upbound 32 `
  --attn_implementation sdpa `
  --trainer_mode="regular" `
  --model_max_length 1792 `
  --save_only_model True
