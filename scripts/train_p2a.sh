#!/usr/bin/env bash
# ARL-aligned Uni-Agent launcher for both baseline and P2A runs.
#
# Leave P2A_BONUS_MAP_DIR unset for the vanilla baseline. P2A_EVAL_BONUS_MAP_DIR
# is independent: it enables validation graph metrics for baseline and P2A runs.
#
# Usage:
#   TRAIN_FILE=... TEST_FILE=... MODEL_PATH=... bash scripts/train_p2a.sh
#   TRAIN_FILE=... TEST_FILE=... MODEL_PATH=... P2A_BONUS_MAP_DIR=... P2A_M_MAX=3.0 P2A_CREDIT_GRANULARITY=step bash scripts/train_p2a.sh
#   P2A_TRAIN_ROLLOUT_N=8 P2A_VAL_ROLLOUT_N=1 bash scripts/train_p2a.sh
set -xeuo pipefail
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
# Let Megatron/TE auto-select attention backend (do not override NVTE_FUSED_ATTN)

SCRIPT_SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P2A_STAGE_LOCAL_RUNTIME="${P2A_STAGE_LOCAL_RUNTIME:-1}"
source "${SCRIPT_SRC_ROOT}/scripts/lib.sh"
p2a_source_local_env "${SCRIPT_SRC_ROOT}"
P2A_VENV_DIR="$(p2a_runtime_venv_rel "${SCRIPT_SRC_ROOT}")"
export P2A_VENV_DIR
export UV_PROJECT_ENVIRONMENT="${SCRIPT_SRC_ROOT}/${P2A_VENV_DIR}"
export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
p2a_source_runtime_profile "${UV_PROJECT_ENVIRONMENT}"
p2a_stage_local_runtime "${SCRIPT_SRC_ROOT}"
SRC_ROOT="${P2A_RUNTIME_SRC_ROOT}"
UV_PROJECT_ENVIRONMENT="${SRC_ROOT}/${P2A_VENV_DIR}"
export UV_PROJECT_ENVIRONMENT
export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:${PATH}"
p2a_source_runtime_profile "${UV_PROJECT_ENVIRONMENT}"
UNI_AGENT_DIR="${SRC_ROOT}/uni-agent"
cd "${SRC_ROOT}"

project_name='P2A-SWE-Agent'
exp_name='P2A-GSPO-R2E-Fully-Async'

RAY_DATA_HOME=${RAY_DATA_HOME:-"/apdcephfs_sgfd/share_303735497/yixianliu/arimazhu/verl"}
MODEL_PATH=${MODEL_PATH:-"$(default_model_path)"}
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${RAY_DATA_HOME}/data/swe_agent/r2e_gym_subset_p2a.train.parquet"}
TEST_FILE=${TEST_FILE:-"${RAY_DATA_HOME}/data/swe_agent/swe_bench_verified_hard.parquet"}
MODEL_PATH="$(resolve_shared_path "${MODEL_PATH}")"
TRAIN_FILE="$(resolve_shared_path "${TRAIN_FILE}")"
TEST_FILE="$(resolve_shared_path "${TEST_FILE}")"
resolve_env_path_if_set() {
    local key="$1"
    local value="${!key:-}"
    if [[ -n "${value}" ]]; then
        printf -v "${key}" '%s' "$(resolve_shared_path "${value}")"
        export "${key}"
    fi
}
resolve_env_path_if_set P2A_BONUS_MAP_DIR
resolve_env_path_if_set P2A_EVAL_BONUS_MAP_DIR
resolve_env_path_if_set P2A_EVAL_FILTER_BONUS_MAP_DIR
resolve_env_path_if_set P2A_EVAL_DETAILS_DIR
RUNTIME_ENV=${RUNTIME_ENV:-"${RAY_DATA_HOME}/data/swe_agent/runtime_env_arl.yaml"}
# Agent-loop actors on every node load this path, so it must resolve on all of
# them; the (staged) source tree is present per-node at the same location,
# unlike RAY_DATA_HOME which is node-local to the head.
DEFAULT_AGENT_CONFIG_PATH="${SRC_ROOT}/env/agent_config_nexus.yaml"
AGENT_CONFIG_PATH=${AGENT_CONFIG_PATH:-"${DEFAULT_AGENT_CONFIG_PATH}"}
PYTHON_BIN=${PYTHON_BIN:-"${UV_PROJECT_ENVIRONMENT}/bin/python"}
RAY_BIN=${RAY_BIN:-"${UV_PROJECT_ENVIRONMENT}/bin/ray"}
echo "[P2A] UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT}"
if [[ "${P2A_SHARED_SRC_ROOT:-${SRC_ROOT}}" != "${SRC_ROOT}" ]]; then
    echo "[P2A] shared source: ${P2A_SHARED_SRC_ROOT}"
    echo "[P2A] runtime source: ${SRC_ROOT}"
fi
if [[ ! -x "${PYTHON_BIN}" || ! -x "${RAY_BIN}" ]]; then
    echo "[P2A] Missing ${PYTHON_BIN} or ${RAY_BIN}" >&2
    echo "[P2A] Build the shared src/.venv first, then rerun this script." >&2
    exit 2
fi
RAY_API_URL="${RAY_API_SERVER_ADDRESS:-http://localhost:8265}"
export RAY_API_SERVER_ADDRESS="${RAY_API_URL}"
RAY_API_HOST="${RAY_API_URL#*://}"
RAY_API_HOST="${RAY_API_HOST%%/*}"
RAY_API_HOST="${RAY_API_HOST%%:*}"
RAY_NO_PROXY="localhost,::1"
if [[ -n "${RAY_API_HOST}" && "${RAY_API_HOST}" != "${RAY_API_URL}" ]]; then
    RAY_NO_PROXY="${RAY_NO_PROXY},${RAY_API_HOST}"
fi
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${RAY_NO_PROXY}"
export no_proxy="${no_proxy:+${no_proxy},}${RAY_NO_PROXY}"

normalize_bool() {
    local key="$1"
    local value="$2"
    case "${value}" in
        true|True|1|yes|YES) printf 'True\n' ;;
        false|False|0|no|NO) printf 'False\n' ;;
        *)
            echo "[P2A] Invalid ${key}=${value}; use true or false." >&2
            return 2
            ;;
    esac
}

rollout_mode="async"
rollout_name="vllm"

# Uni-Agent's GSPO recipe keeps GRPO advantage estimation and switches the
# actor policy loss to GSPO below.
adv_estimator=grpo

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=4e-4
clip_ratio_high=4e-4

max_prompt_length=$((1024 * 4))
max_response_length=$((1024 * 64))
enable_overlong_buffer=False
overlong_buffer_len=$((1024 * 4))
overlong_penalty_factor=1.0

loss_agg_mode="token-mean"
# Match the Uni-Agent example by default; override with P2A_LOSS_MODE for ablations.
loss_mode=${P2A_LOSS_MODE:-gspo}

temperature=1.0
top_p=1.0
top_k=-1
val_temperature=1.0
val_top_p=0.95
val_top_k=-1

use_dynamic_bsz=True
offload=True
# Defaults assume the 2-train-node topology (16 GPUs: TP=4 x CP=4); override via
# env for other node counts, e.g. TRAIN_CP=2 on a single 8-GPU train node.
gen_tp=${GEN_TP:-4}
train_tp=${TRAIN_TP:-4}
train_pp=${TRAIN_PP:-1}
train_cp=${TRAIN_CP:-4}
train_ep=${TRAIN_EP:-8}
train_etp=${TRAIN_ETP:-1}
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) / train_cp))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) / train_cp))

optimizer_offload_fraction=1.0

USE_MBRIDGE=True
VANILLA_MBRIDGE="${VANILLA_MBRIDGE:-True}"
USE_DIST_CKPT=False

NNODES_ROLLOUT=${NNODES_ROLLOUT:-4}
NNODES_TRAIN=${NNODES_TRAIN:-4}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-8}
infer_ep=${INFER_EP:-1}
# These three require Uni-Agent's patched vLLM; stock vLLM 0.11.0 rejects the
# --performance-mode / --enable-return-routed-experts server flags, so they
# default off and are opt-in for runtimes that carry the patched build.
vllm_performance_mode="${VLLM_PERFORMANCE_MODE:-}"
router_replay_mode="${ROUTER_REPLAY_MODE:-R3}"
enable_routing_replay="${ENABLE_ROUTING_REPLAY:-True}"
rollout_engine_params=()
if [[ -n "${vllm_performance_mode}" ]]; then
    rollout_engine_params+=("+actor_rollout_ref.rollout.engine_kwargs.vllm.performance_mode=${vllm_performance_mode}")
fi
update_weights_bucket_megabytes="${P2A_UPDATE_WEIGHTS_BUCKET_MB:-2048}"
nccl_timeout="${P2A_NCCL_TIMEOUT:-9600}"

train_prompt_bsz=0
n_resp_per_prompt=${P2A_TRAIN_ROLLOUT_N:-${P2A_ROLLOUT_N:-8}}
val_resp_per_prompt=${P2A_VAL_ROLLOUT_N:-${P2A_VALIDATION_ROLLOUT_N:-1}}
train_prompt_mini_bsz=${TRAIN_PROMPT_MINI_BSZ:-16}
total_rollout_steps=200000
test_freq=100
val_before_train=${VAL_BEFORE_TRAIN:-True}
staleness_threshold=1.0
trigger_parameter_sync_step=4
require_batches=1
partial_rollout=True

if [[ -n "${P2A_BONUS_MAP_DIR:-}${P2A_EVAL_BONUS_MAP_DIR:-}" && -z "${UNI_AGENT_P2A_TRACE:-}" ]]; then
    export UNI_AGENT_P2A_TRACE=1
fi

mkdir -p "$(dirname "${RUNTIME_ENV}")"
if [[ ! -f "${RUNTIME_ENV}" ]]; then
    cp "${UNI_AGENT_DIR}/examples/swe_agent_235b/runtime_env.yaml" "${RUNTIME_ENV}"
fi
if [[ ! -f "${AGENT_CONFIG_PATH}" ]]; then
    echo "[P2A] AGENT_CONFIG_PATH does not exist: ${AGENT_CONFIG_PATH}" >&2
    exit 2
fi

# Ray runtime_env working_dir packages and uploads the checkout. The GPU nodes
# already share SRC_ROOT, so run there directly and keep runtime_env env-only.
"${PYTHON_BIN}" -m p2a.runtime_env "${RUNTIME_ENV}" --src-root "${SRC_ROOT}" --drop-working-dir

ensure_model_path

if [[ -n "${P2A_EVAL_CASE_TYPES:-}${P2A_EVAL_PATTERN_COMPUTABLE:-}" ]]; then
    eval_pattern_computable="${P2A_EVAL_PATTERN_COMPUTABLE:-}"
    eval_filter_bonus_map_dir="${P2A_EVAL_FILTER_BONUS_MAP_DIR:-${P2A_EVAL_BONUS_MAP_DIR:-}}"
    if [[ -z "${eval_filter_bonus_map_dir}" ]]; then
        echo "[P2A] P2A_EVAL_CASE_TYPES/P2A_EVAL_PATTERN_COMPUTABLE requires P2A_EVAL_BONUS_MAP_DIR or P2A_EVAL_FILTER_BONUS_MAP_DIR." >&2
        exit 2
    fi
    scope_slug="${P2A_EVAL_CASE_TYPES:-all}"
    scope_slug="${scope_slug//,/+}"
    scope_slug="${scope_slug//\//_}"
    if [[ "${eval_pattern_computable}" == "1" || "${eval_pattern_computable,,}" == "true" ]]; then
        scope_slug="${scope_slug}+pattern"
    fi
    test_stem="$(basename "${TEST_FILE}")"
    test_stem="${test_stem%.*}"
    filtered_test_file="${P2A_FILTERED_TEST_FILE:-$(dirname "${TEST_FILE}")/${test_stem}.${scope_slug}.parquet}"
    filter_cmd=(
        "${PYTHON_BIN}" -m p2a.filter_bonus_map_instances
        "${TEST_FILE}"
        --bonus-map-dir "${eval_filter_bonus_map_dir}"
        --out "${filtered_test_file}"
    )
    if [[ -n "${P2A_EVAL_CASE_TYPES:-}" ]]; then
        IFS=',' read -r -a eval_case_types <<< "${P2A_EVAL_CASE_TYPES}"
        for case_type in "${eval_case_types[@]}"; do
            [[ -n "${case_type}" ]] && filter_cmd+=(--case-type "${case_type}")
        done
    fi
    if [[ "${eval_pattern_computable}" == "1" || "${eval_pattern_computable,,}" == "true" ]]; then
        filter_cmd+=(--pattern-computable true)
    fi
    "${filter_cmd[@]}"
    TEST_FILE="${filtered_test_file}"
    export TEST_FILE
    export P2A_EVAL_SCOPE_FILE="${filtered_test_file}.scope.json"
    echo "[P2A] filtered validation data: ${TEST_FILE}"
    echo "[P2A] validation scope metadata: ${P2A_EVAL_SCOPE_FILE}"
fi

if [[ "${P2A_SKIP_MEGATRON_PREFLIGHT:-0}" != "1" ]]; then
    "${PYTHON_BIN}" - <<'PY'
import importlib

from verl.workers.engine import EngineRegistry, MegatronEngine

registered = sorted(EngineRegistry._engines.get("language_model", {}))
if MegatronEngine is None or "megatron" not in registered:
    try:
        importlib.import_module("verl.workers.engine.megatron")
    except Exception as exc:  # noqa: BLE001 - include the real import failure in the launcher error
        import_error = f"{type(exc).__name__}: {exc}"
    else:
        import_error = "direct import succeeded, but registry still lacks megatron"
    raise SystemExit(
        "[P2A] Megatron engine is not registered in this Python environment. "
        f"registered_language_model_backends={registered}. "
        f"megatron_import={import_error}. "
        "Run `uv sync --extra train --extra gpu` on the GPU/shared runtime, "
        "then rerun ray_setup/main so Ray workers use the refreshed .venv."
    )
print(f"[P2A] Megatron backend registered (language_model backends: {registered})")
PY
fi

if [[ "${P2A_SKIP_TRANSFORMER_ENGINE_PREFLIGHT:-0}" != "1" ]]; then
    "${PYTHON_BIN}" - <<'PY'
import importlib.metadata
import os
import sys

try:
    import torch
    import transformer_engine.pytorch  # noqa: F401
except Exception as exc:  # noqa: BLE001 - report binary/linker failures directly
    raise SystemExit(
        "[P2A] TransformerEngine is required by Uni-Agent's Megatron/mbridge path, "
        f"but transformer_engine.pytorch is not importable: {type(exc).__name__}: {exc}. "
        "Use the native CUDA 13.0 `.venv` stack: "
        "`CUDA_HOME=/usr/local/cuda-13.0 UV_PROJECT_ENVIRONMENT=$PWD/.venv "
        "uv sync --locked --extra train --extra gpu`."
    )
else:
    try:
        te_version = importlib.metadata.version("transformer-engine")
    except importlib.metadata.PackageNotFoundError:
        te_version = "unknown"
    print(
        "[P2A] TransformerEngine import check passed "
        f"(python={sys.executable}, torch={torch.__version__}, torch_cuda={torch.version.cuda}, "
        f"CUDA_HOME={os.environ.get('CUDA_HOME')}, transformer_engine={te_version})"
    )
PY
fi

# TRAIN_FILE should point at the skip-filtered *.train.parquet emitted by
# scripts/build_data.py r2e.

"${PYTHON_BIN}" - "${RAY_API_URL}" <<'PY'
import sys
import urllib.request

url = sys.argv[1].rstrip("/")
try:
    with urllib.request.urlopen(url + "/api/version", timeout=5) as response:
        print(f"[P2A] Ray dashboard reachable: {url} ({response.status})")
except Exception as exc:
    raise SystemExit(f"[P2A] Ray dashboard is not reachable at {url}: {exc}")
PY

# p2a.main wraps Uni-Agent's fully async trainer and stays vanilla when
# P2A_BONUS_MAP_DIR is unset.
"${RAY_BIN}" job submit --no-wait --address="${RAY_API_URL}" --runtime-env "${RUNTIME_ENV}" \
    -- env -C "${SRC_ROOT}" "${PYTHON_BIN}" -m p2a.main \
    'hydra.searchpath=[pkg://verl.trainer.config]' \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    data.return_raw_chat=True \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    +actor_rollout_ref.model.override_config.model_config.max_position_embeddings=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.model.use_fused_kernels=False \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_decay_style='constant' \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.optim.lr_decay_steps=${total_rollout_steps} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=${optimizer_offload_fraction} \
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True \
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True \
    actor_rollout_ref.actor.megatron.use_mbridge=$USE_MBRIDGE \
    actor_rollout_ref.actor.megatron.vanilla_mbridge=$VANILLA_MBRIDGE \
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=$USE_DIST_CKPT \
    actor_rollout_ref.actor.megatron.use_remove_padding=True \
    actor_rollout_ref.actor.megatron.param_offload=${offload} \
    actor_rollout_ref.actor.megatron.grad_offload=${offload} \
    actor_rollout_ref.actor.megatron.optimizer_offload=${offload} \
    actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp} \
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type="alltoall" \
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32 \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full \
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1 \
    actor_rollout_ref.actor.router_replay.mode="${router_replay_mode}" \
    actor_rollout_ref.rollout.enable_rollout_routing_replay=${enable_routing_replay} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.prompt_length=${max_prompt_length} \
    actor_rollout_ref.rollout.response_length=${max_response_length} \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1 \
    actor_rollout_ref.rollout.agent.num_workers=${NUM_AGENT_WORKERS:-128} \
    actor_rollout_ref.rollout.agent.agent_loop_config_path=${AGENT_CONFIG_PATH} \
    actor_rollout_ref.rollout.agent.default_agent_loop=swe_agent \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.expert_parallel_size=${infer_ep} \
    actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length)) \
    ${rollout_engine_params[@]+"${rollout_engine_params[@]}"} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${val_resp_per_prompt} \
    actor_rollout_ref.rollout.name=${rollout_name} \
    actor_rollout_ref.rollout.mode=${rollout_mode} \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.nccl_timeout=${nccl_timeout} \
    actor_rollout_ref.hybrid_engine=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${update_weights_bucket_megabytes} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=${USE_DIST_CKPT} \
    actor_rollout_ref.ref.megatron.param_offload=${offload} \
    actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp} \
    actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp} \
    actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp} \
    actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep} \
    actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp} \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.val_before_train=${val_before_train} \
    trainer.save_freq=10 \
    trainer.total_epochs=20 \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.nnodes="${NNODES_TRAIN}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    rollout.nnodes="${NNODES_ROLLOUT}" \
    rollout.n_gpus_per_node="${NGPUS_PER_NODE}" \
    rollout.total_rollout_steps="${total_rollout_steps}" \
    trainer.test_freq="${test_freq}" \
    async_training.staleness_threshold="${staleness_threshold}" \
    async_training.trigger_parameter_sync_step="${trigger_parameter_sync_step}" \
    async_training.require_batches="${require_batches}" \
    async_training.partial_rollout="${partial_rollout}"
