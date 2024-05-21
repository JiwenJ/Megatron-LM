#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=1
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NNODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

CHECKPOINT_PATH=/data/jjw/llama3-4B/llama3-merge-4B-mega

DATA_PATH=/data/jjw/llama3-4B/llama3_text_document

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT
"

LLAMA_ARGS=" \
--seq-length 1024 \
--max-position-embeddings 1024 \
--exit-on-missing-checkpoint \
--use-checkpoint-args \
--no-load-optim \
--no-load-rng \
--untie-embeddings-and-output-weights \
--use-rotary-position-embeddings \
--normalization RMSNorm \
--no-position-embedding \
--no-masked-softmax-fusion \
--fp16 \
--tokenizer-type Llama3Tokenizer \
--no-rope-fusion \
"
LOG_ARGS=" \
--wandb-project llama3-4B \
--wandb-exp-name llama3 \
--log-params-norm \
--log-num-zeros-in-grad \
--log-throughput \
--log-progress \
--tensorboard-dir /data/jjw/llama3-4B/expreriment \
--log-batch-size-to-tensorboard \
--log-validation-ppl-to-tensorboard \
--log-memory-to-tensorboard \
"
GPT_ARGS="
--train-iters 100000 \
--micro-batch-size 1 \
--tensor-model-parallel-size 1 \
--pipeline-model-parallel-size 1 \
--lr 0.0005 \
--transformer-impl local \
--data-path ${DATA_PATH} \
--tokenizer-model ${TOKENIZER_MODEL} \
--load ${CHECKPOINT_DIR} \
--norm-epsilon=1e-6 \
--disable-bias-linear \
--save-interval 2500 \
--log-interval 2500 \
--save ${SAVE_PATH} \
--recompute-activations \
--no-gradient-accumulation-fusion \
"
OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"
python ../get_training_info.py $DISTRIBUTED_ARGS \
    $LLAMA_ARGS \
    $GPT_ARGS \
    $LOG_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
    --save $CHECKPOINT_PATH \
    --load $CHECKPOINT_PATH