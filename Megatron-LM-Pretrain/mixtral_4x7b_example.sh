#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=8
MASTER_ADDR=localhost
MASTER_PORT=12345
NNODES=1
NODE_RANK=0

CHECKPOINT_PATH=YOUR_MEGATRON_CHECKPOINT_PATH_HERE
BLENDED_DATA=YOUR_BLENDED_DATA_PATH_HERE

TP=4
PP=2
EP=1

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port $MASTER_PORT \
"


GPT_ARGS="
    --num-layers 32 \
    --hidden-size 4096 \
    --ffn-hidden-size 14336 \
    --group-query-attention \
    --num-attention-heads 32 \
    --num-query-groups 8 \
    --num-experts 4 \
    --moe-router-topk 2 \
    --moe-type megatron \
    --moe-grouped-gemm \
    --moe-router-load-balancing-type aux_loss \
    --moe-aux-loss-coeff 0.001 \
    --swiglu \
    --seq-length 4096 \
    --max-position-embeddings 32768 \
    --untie-embeddings-and-output-weights \
    --use-rotary-position-embeddings \
    --disable-bias-linear \
    --normalization RMSNorm \
    --no-position-embedding \
    --no-masked-softmax-fusion \
    --hidden-dropout 0.02 \
    --attention-dropout 0.02 \
    --tensor-model-parallel-size ${TP} \
    --pipeline-model-parallel-size ${PP} \
    --expert-model-parallel-size ${EP} \
    --sequence-parallel \
    --train-iters 250000 \
        --lr-warmup-iters 512 \
        --weight-decay 0.01 \
        --clip-grad 1.0 \
    --lr 1e-5 \
        --min-lr 1e-6 \
        --lr-decay-style cosine \
    --adam-beta1 0.9 \
	--adam-beta2 0.95 \
    --bf16 \
    --use-flash-attn
"


DATA_ARGS="
    --dataloader-type cyclic \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model  \
    --vocab-size 55424 \
    --data-path ${BLENDED_DATA} \
    --split 1000,0,0 \
    --micro-batch-size 1 \
    --global-batch-size 512
"

OUTPUT_ARGS="
    --load ${CHECKPOINT_PATH} \
    --no-load-optim \
    --no-load-rng \
    --save ${CHECKPOINT_PATH} \
    --no-save-optim \
    --no-save-rng \
    --log-interval 1 \
    --save-interval 1024 \
    --eval-interval 100000 \
    --eval-iters 100000
"

torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --distributed-backend nccl \
