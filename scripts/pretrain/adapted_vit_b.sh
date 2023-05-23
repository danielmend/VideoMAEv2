#!/usr/bin/env bash
set -x

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1
source /fsx/daniel_mend/dataset_experiments/video_mae/venv/bin/activate

OUTPUT_DIR='/fsx/daniel_mend/dataset_experiments/video_mae/vit_b_hybrid_pt_800e_old'
DATA_PATH='s3://stability-west/video_cc/video_cc_clipped/{00900..01200}.tar'

JOB_NAME="videomae"
PARTITION=${PARTITION:-"g80"}
# 8 for 1 node, 16 for 2 node, etc.
GPUS=${GPUS:-8}
GPUS_PER_NODE=${GPUS_PER_NODE:-8}
CPUS_PER_TASK=${CPUS_PER_TASK:-12}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}
# batch_size can be adjusted according to the graphics card
srun -p $PARTITION \
        --job-name=${JOB_NAME} \
        --gres=gpu:${GPUS_PER_NODE} \
        --ntasks=${GPUS} \
        --nodes=1 \
        --ntasks-per-node=${GPUS_PER_NODE} \
        --cpus-per-task=${CPUS_PER_TASK} \
        --kill-on-bad-exit=1 \
        --account stablediffusion \
        --output=/fsx/daniel_mend/dataset_experiments/video_mae/VideoMAEv2_old/scripts/logs/videomae-%j.log \
        ${SRUN_ARGS} \
        python /fsx/daniel_mend/dataset_experiments/video_mae/VideoMAEv2/run_mae_pretraining.py \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.9 \
        --decoder_mask_type run_cell \
        --decoder_mask_ratio 0.5 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 32 \
        --num_sample 4 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_workers 10 \
        --lr 1e-3 \
        --opt adamw \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 0 \
        --save_ckpt_freq 1 \
        --epochs 200 \
        --use_video2dataset \
        --train_num_samples 1500000 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        ${PY_ARGS}
