#! /bin/bash

#export CUDA_VISIBLE_DEVICES='1'
export ABS_PATH=$(pwd)
export NCCL_P2P_LEVEL=NVL
export NCCL_TIMEOUT=3600
export DISABLE_VERSION_CHECK=1
export WANDB__SERVICE_WAIT=300

output_dir="$ABS_PATH/predicts/texts" # texts/videos/images
model_name_or_path=Qwen/Qwen2.5-VL-3B-Instruct #
mkdir -p ${output_dir}

cache_dir=hf_cache_dir
mkdir -p ${cache_dir}
cutoff_len=4096
# text
evaldatasets=semeval2018_english_test,semeval2018_arabic_test,semeval2018_spanish_test,EWECT_usual_test,EWECT_virus_test,onlineshopping_test,MMS_test,XED_test
# images
# evaldatasets=EMOTIC_EC_test,EMOTIC_SI_test,FER2013_test,CFAPS_EC_test,CFAPS_EI_test
# videos
# evaldatasets=SAMSEMO_test,MELD_EC_test,MELD_SP_test,CHSIMS_SP_test,CHSIMS_SI_test

#FT
DISTRIBUTED_ARGS="
    --nproc_per_node 1 \
    --nnodes 1 \
    --node_rank 0 \
    --master_port 20005
    "

torchrun $DISTRIBUTED_ARGS src/train.py \
    --stage sft \
    --trust_remote_code true \
    --do_predict true \
    --image_max_pixels 262144 \
    --video_max_pixels 16384 \
    --model_name_or_path $model_name_or_path \
    --template qwen2_vl \
    --finetuning_type full \
    --eval_dataset  $evaldatasets\
    --cutoff_len $cutoff_len \
    --overwrite_cache true \
    --preprocessing_num_workers 16 \
    --output_dir $output_dir \
    --overwrite_output_dir true \
    --ddp_timeout 1800000000 \
    --per_device_eval_batch_size 8 \
    --predict_with_generate true \
