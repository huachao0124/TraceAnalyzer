#!/usr/bin/env bash
# Four-node adaptation of Uni-Agent's Qwen3-Coder 30B MoE recipe.
#
# Trainer and optimization settings match the official recipe where supported
# by the pinned stack. PP is halved with the node count so the non-expert and
# expert data-parallel degrees remain equal to the 64-GPU recipe.
set -euo pipefail

SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export PROJECT_NAME="${PROJECT_NAME:-P2A-SWE-Agent}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-Official-Qwen3-MoE-Colocate-32GPU}"

export BASELINE_TRAINER_MODE=colocate_async
export NUM_WARMUP_BATCHES=1

export NNODES=4
export NGPUS_PER_NODE=8
export GEN_TP=4
export TRAIN_TP=4
export TRAIN_PP=1
export TRAIN_CP=4
export TRAIN_EP=8
export TRAIN_ETP=1
export INFER_EP=1

export SYNC_TRAIN_PROMPT_BSZ=64
export N_RESP_PER_PROMPT=8
export TRAIN_PROMPT_MINI_BSZ=16

export MAX_PROMPT_LENGTH=8192
export MAX_RESPONSE_LENGTH=131072
export USE_FUSED_KERNELS=True

export ROLLOUT_IS=token
export ROLLOUT_IS_THRESHOLD=2.0
export ROLLOUT_IS_BATCH_NORMALIZE=False
export ROLLOUT_RS=null
export ROLLOUT_RS_THRESHOLD=0.999_1.001
export BYPASS_MODE=False

export NUM_AGENT_WORKERS=8
export GATEWAY_COUNT=8
export AGENT_CONCURRENCY=512
export TASK_CONFIG_PATH="${TASK_CONFIG_PATH:-${SRC_ROOT}/env/agent_config_nexus_official.yaml}"

export VAL_BEFORE_TRAIN=False
export TEST_FREQ=-1
export SAVE_FREQ=10
export TOTAL_EPOCHS=10
export LR_DECAY_STEPS=2000
export GPU_MEMORY_UTILIZATION=0.7
export DISABLE_LOG_STATS=True
export CHECKPOINT_SAVE_CONTENTS="['model','hf_model']"

export TEST_FILE="${TEST_FILE:-${SRC_ROOT}/../../datasets/p2a/swe_bench_verified.parquet}"

exec bash "${SRC_ROOT}/scripts/train_sync_baseline.sh" "$@"
