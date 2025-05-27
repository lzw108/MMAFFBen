#! /bin/bash

#export CUDA_VISIBLE_DEVICES='2,3'
export ABS_PATH=$(pwd)
export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=3600
export DISABLE_VERSION_CHECK=1

output_dir="$ABS_PATH/saved_models/qwen2.5_vl-2b/fullsft"
model_name_or_path=Qwen/Qwen2.5-VL-2B-Instruct 
mkdir -p ${output_dir}

cache_dir=hf_cache_dir
mkdir -p ${cache_dir}
cutoff_len=4096

datasets=SAMSEMO_train,MELD_EC_train,MELD_SP_train,CHSIMS_SP_train,CHSIMS_SI_train,semeval2018_english_train,semeval2018_arabic_train,semeval2018_spanish_train,EWECT_usual_train,EWECT_virus_train,onlineshopping_train,EMOTIC_EC_train,EMOTIC_SI_train,FER2013_train,CFAPS_EC_train,CFAPS_EI_train,MMS_train,XED_train

# full eval dataset, it will cost much long time
#evaldatasets=SAMSEMO_val,MELD_EC_val,MELD_SP_val,CHSIMS_SP_val,CHSIMS_SI_val,semeval2018_english_val,semeval2018_arabic_val,semeval2018_spanish_val,EWECT_usual_val,EWECT_virus_val,onlineshopping_val,EMOTIC_EC_val,EMOTIC_SI_val,FER2013_val,CFAPS_EC_val,CFAPS_EI_val,MMS_val,XED_val
evaldatasets=semeval2018_arabic_val # for test process

#FT

max_steps=450

DISTRIBUTED_ARGS="
    --nproc_per_node 2 \
    --nnodes 1 \
    --node_rank 0 \
    --master_port 20004
    "
torchrun $DISTRIBUTED_ARGS src/train.py \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --stage sft \
    --do_train \
    --model_name_or_path $model_name_or_path \
    --dataset $datasets \
    --preprocessing_batch_size 256 \
    --streaming true \
    --buffer_size 256 \
    --dispatch_batches false \
    --max_steps $max_steps \
    --image_max_pixels 262144 \
    --video_max_pixels 16384 \
    --mix_strategy interleave_over \
    --template qwen2_vl \
    --finetuning_type full \
    --output_dir $output_dir \
    --overwrite_cache 1\
    --overwrite_output_dir \
    --warmup_steps 100 \
    --weight_decay 0.1 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 128 \
    --ddp_timeout 1800000000 \
    --learning_rate 5e-6 \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len $cutoff_len \
    --save_steps 60 \
    --plot_loss \
    --num_train_epochs 2 \
    --mix_strategy interleave_over \
    --eval_dataset  $evaldatasets\
    --per_device_eval_batch_size 2 \
    --eval_strategy steps \
    --eval_steps 160 \
    --bf16 
