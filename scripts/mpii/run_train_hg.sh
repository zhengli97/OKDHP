#!/usr/bin/env bash

pushd ../../

export CUDA_VISIBLE_DEVICES=0

python tools/train.py \
    --cfg experiments/mpii/hourglass/hg_template.yaml \
    GPUS '(0,)' \
    DATASET.COLOR_RGB False \
    DATASET.DATASET 'mpii' \
    DATASET.ROOT 'your_data_directory' \
    DATASET.NUM_JOINTS_HALF_BODY 8 \
    DATASET.PROB_HALF_BODY -1.0 \
    MODEL.NAME 'hourglass_okd_share_less'\
    MODEL.EXP 'stack4_ens_weight_1_kd_2' \
    MODEL.NUM_JOINTS  16 \
    MODEL.INIT_WEIGHTS False \
    MODEL.IMAGE_SIZE 256,256 \
    MODEL.HEATMAP_SIZE 64,64 \
    MODEL.SIGMA 2 \
    MODEL.EXTRA.NUM_FEATURES  256 \
    MODEL.EXTRA.NUM_STACKS 4 \
    MODEL.EXTRA.NUM_BLOCKS 1 \
    TRAIN.BATCH_SIZE_PER_GPU 16 \
    TRAIN.KD_WEIGHT 2.0 \
    TRAIN.ENS_WEIGHT 1.0 \
    TRAIN.BEGIN_EPOCH 0 \
    TRAIN.END_EPOCH 150 \
    TEST.BATCH_SIZE_PER_GPU 16 \
    DEBUG.DEBUG False

popd
