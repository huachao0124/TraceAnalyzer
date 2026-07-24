#!/usr/bin/env bash
# Vanilla GSPO launcher for the Uni-Agent SWE/R2E workload.
#
# The default trainer runs in lockstep sample -> update mode while the
# multi-turn AgentLoop uses the asynchronous vLLM server interface. A wrapper
# may select another V1 trainer mode while reusing the same Nexus integration.
set -xeuo pipefail

SCRIPT_SRC_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
P2A_STAGE_LOCAL_RUNTIME="${P2A_STAGE_LOCAL_RUNTIME:-0}"
P2A_VENV_DIR="${P2A_VENV_DIR:-.venv-container}"
export P2A_STAGE_LOCAL_RUNTIME P2A_VENV_DIR

source "${SCRIPT_SRC_ROOT}/scripts/lib.sh"
p2a_source_local_env "${SCRIPT_SRC_ROOT}"

export UV_PROJECT_ENVIRONMENT="${SCRIPT_SRC_ROOT}/${P2A_VENV_DIR}"
export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:/usr/local/cuda/bin:${PATH}"
p2a_source_runtime_profile "${UV_PROJECT_ENVIRONMENT}"
p2a_stage_local_runtime "${SCRIPT_SRC_ROOT}"

SRC_ROOT="${P2A_RUNTIME_SRC_ROOT}"
UV_PROJECT_ENVIRONMENT="${SRC_ROOT}/${P2A_VENV_DIR}"
export UV_PROJECT_ENVIRONMENT
export VIRTUAL_ENV="${UV_PROJECT_ENVIRONMENT}"
export PATH="${UV_PROJECT_ENVIRONMENT}/bin:/usr/local/cuda/bin:${PATH}"
p2a_source_runtime_profile "${UV_PROJECT_ENVIRONMENT}"
cd "${SRC_ROOT}"

# This launcher is deliberately vanilla. P2A advantage reshaping currently
# targets the fully asynchronous trainer and is not enabled here.
unset P2A_BONUS_MAP_DIR P2A_M_MAX P2A_TRACKING_MODE P2A_CREDIT_GRANULARITY
unset P2A_EVAL_BONUS_MAP_DIR P2A_EVAL_DETAILS_DIR P2A_EVAL_NEAR_THRESHOLD
unset UNI_AGENT_P2A_TRACE

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export CUDA_PATH="${CUDA_PATH:-${CUDA_HOME}}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-/usr/local/cuda/lib64}"
export NVTE_FRAMEWORK="${NVTE_FRAMEWORK:-pytorch}"
export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export VLLM_USE_DEEP_GEMM="${VLLM_USE_DEEP_GEMM:-0}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-bond1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-bond1}"
export NCCL_CUMEM_ENABLE="${NCCL_CUMEM_ENABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-1}"
export NCCL_MNNVL_ENABLE="${NCCL_MNNVL_ENABLE:-1}"
export VLLM_USE_NCCL_SYMM_MEM="${VLLM_USE_NCCL_SYMM_MEM:-1}"

RAY_DATA_HOME="${RAY_DATA_HOME:-/apdcephfs_sgfd/share_303735497/yixianliu/arimazhu/verl}"
project_name="${PROJECT_NAME:-P2A-SWE-Agent}"
exp_name="${EXPERIMENT_NAME:-Vanilla-GSPO-R2E-Sync-32GPU}"

MODEL_PATH="${MODEL_PATH:-$(default_model_path)}"
TRAIN_FILE="${TRAIN_FILE:-${SRC_ROOT}/../../datasets/p2a/r2e_gym_subset_p2a.train.parquet}"
TEST_FILE="${TEST_FILE:-${SRC_ROOT}/../../datasets/p2a/swe_bench_verified_hard.parquet}"
MODEL_PATH="$(resolve_shared_path "${MODEL_PATH}")"
TRAIN_FILE="$(resolve_shared_path "${TRAIN_FILE}")"
TEST_FILE="$(resolve_shared_path "${TEST_FILE}")"

CKPTS_DIR="${CKPTS_DIR:-${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}}"
WANDB_DIR="${WANDB_DIR:-${RAY_DATA_HOME}/wandb}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR

RUNTIME_ENV_SOURCE="${RUNTIME_ENV_SOURCE:-${RAY_DATA_HOME}/data/swe_agent/runtime_env_arl.yaml}"
RUNTIME_ENV="${RUNTIME_ENV:-${RAY_DATA_HOME}/data/swe_agent/runtime_env_sync.yaml}"
TASK_CONFIG_PATH="${TASK_CONFIG_PATH:-${AGENT_CONFIG_PATH:-${SRC_ROOT}/env/agent_config_nexus.yaml}}"
GATEWAY_COUNT="${GATEWAY_COUNT:-8}"
AGENT_CONCURRENCY="${AGENT_CONCURRENCY:-32}"
AGENT_LOG_DIR="${AGENT_LOG_DIR:-${RAY_DATA_HOME}/logs/${project_name}/${exp_name}}"
PYTHON_BIN="${PYTHON_BIN:-${UV_PROJECT_ENVIRONMENT}/bin/python}"
RAY_BIN="${RAY_BIN:-${UV_PROJECT_ENVIRONMENT}/bin/ray}"
RAY_API_URL="${RAY_API_SERVER_ADDRESS:-http://127.0.0.1:8265}"
export RAY_API_SERVER_ADDRESS="${RAY_API_URL}"

if [[ ! -x "${PYTHON_BIN}" || ! -x "${RAY_BIN}" ]]; then
    echo "[baseline] Missing ${PYTHON_BIN} or ${RAY_BIN}" >&2
    exit 2
fi
for path in "${MODEL_PATH}/config.json" "${TRAIN_FILE}" "${TEST_FILE}" "${TASK_CONFIG_PATH}"; do
    if [[ ! -e "${path}" ]]; then
        echo "[baseline] Required path does not exist: ${path}" >&2
        exit 2
    fi
done

mkdir -p "$(dirname "${RUNTIME_ENV}")" "${CKPTS_DIR}" "${WANDB_DIR}" "${AGENT_LOG_DIR}"
if [[ ! -f "${RUNTIME_ENV}" ]]; then
    cp "${SRC_ROOT}/config/runtime_env.yaml" "${RUNTIME_ENV}"
fi

# Reuse the configured Nexus credential without echoing it into launcher logs.
set +x
if [[ -z "${GONGFENG_TOKEN:-}" && -f "${RUNTIME_ENV_SOURCE}" ]]; then
    GONGFENG_TOKEN="$(
        awk -F': ' '
            /^  GONGFENG_TOKEN:/ {
                value=$2
                sub(/^"/, "", value)
                sub(/"$/, "", value)
                print value
                exit
            }
        ' "${RUNTIME_ENV_SOURCE}"
    )"
    export GONGFENG_TOKEN
fi
if [[ -z "${GONGFENG_TOKEN:-}" ]]; then
    echo "[baseline] GONGFENG_TOKEN is required for the Nexus deployment." >&2
    exit 2
fi
set -x

"${PYTHON_BIN}" -m p2a.runtime_env \
    "${RUNTIME_ENV}" \
    --src-root "${SRC_ROOT}" \
    --drop-working-dir

RAY_API_HOST="${RAY_API_URL#*://}"
RAY_API_HOST="${RAY_API_HOST%%/*}"
RAY_API_HOST="${RAY_API_HOST%%:*}"
RAY_NO_PROXY="localhost,127.0.0.1,::1"
if [[ -n "${RAY_API_HOST}" ]]; then
    RAY_NO_PROXY="${RAY_NO_PROXY},${RAY_API_HOST}"
fi
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${RAY_NO_PROXY}"
export no_proxy="${no_proxy:+${no_proxy},}${RAY_NO_PROXY}"

if [[ "${SYNC_DRY_RUN:-0}" != "1" ]]; then
    curl --fail --silent --show-error "${RAY_API_URL%/}/api/version" >/dev/null
fi

rollout_name="vllm"
adv_estimator="grpo"
loss_mode="${P2A_LOSS_MODE:-gspo}"
loss_agg_mode="${LOSS_AGG_MODE:-token-mean}"
trainer_mode="${BASELINE_TRAINER_MODE:-sync}"
num_warmup_batches="${NUM_WARMUP_BATCHES:-1}"

max_prompt_length="${MAX_PROMPT_LENGTH:-4096}"
max_response_length="${MAX_RESPONSE_LENGTH:-65536}"
n_resp_per_prompt="${N_RESP_PER_PROMPT:-8}"
train_prompt_bsz="${SYNC_TRAIN_PROMPT_BSZ:-16}"
train_prompt_mini_bsz="${TRAIN_PROMPT_MINI_BSZ:-16}"
use_fused_kernels="${USE_FUSED_KERNELS:-False}"

gen_tp="${GEN_TP:-4}"
train_tp="${TRAIN_TP:-4}"
train_pp="${TRAIN_PP:-1}"
train_cp="${TRAIN_CP:-4}"
train_ep="${TRAIN_EP:-8}"
train_etp="${TRAIN_ETP:-1}"
infer_ep="${INFER_EP:-1}"
actor_ppo_max_token_len="$(((max_prompt_length + max_response_length) / train_cp))"
infer_ppo_max_token_len="${INFER_PPO_MAX_TOKEN_LEN:-${actor_ppo_max_token_len}}"

NNODES="${NNODES:-4}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"
NUM_AGENT_WORKERS="${NUM_AGENT_WORKERS:-32}"
val_before_train="${VAL_BEFORE_TRAIN:-False}"
test_freq="${TEST_FREQ:-100}"
save_freq="${SAVE_FREQ:-10}"
total_epochs="${TOTAL_EPOCHS:-20}"
gpu_memory_utilization="${GPU_MEMORY_UTILIZATION:-0.7}"
update_weights_bucket_megabytes="${P2A_UPDATE_WEIGHTS_BUCKET_MB:-2048}"
nccl_timeout="${P2A_NCCL_TIMEOUT:-9600}"
lr_decay_steps="${LR_DECAY_STEPS:-2000}"
rollout_is="${ROLLOUT_IS:-null}"
rollout_is_threshold="${ROLLOUT_IS_THRESHOLD:-2.0}"
rollout_is_batch_normalize="${ROLLOUT_IS_BATCH_NORMALIZE:-False}"
rollout_rs="${ROLLOUT_RS:-null}"
rollout_rs_threshold="${ROLLOUT_RS_THRESHOLD:-0.999_1.001}"
bypass_mode="${BYPASS_MODE:-False}"
disable_log_stats="${DISABLE_LOG_STATS:-False}"
checkpoint_save_contents="${CHECKPOINT_SAVE_CONTENTS:-}"

# Megatron builds separate rank decompositions for non-expert and expert
# layers. EP combines with ETP and PP; it is not multiplied on top of TP x CP.
world_size=$((NNODES * NGPUS_PER_NODE))
non_expert_model_parallel_size=$((train_tp * train_pp * train_cp))
expert_model_parallel_size=$((train_etp * train_ep * train_pp))
if ((world_size % non_expert_model_parallel_size != 0)); then
    echo "[baseline] world_size=${world_size} is not divisible by TP*PP*CP=${non_expert_model_parallel_size}." >&2
    exit 2
fi
if ((world_size % expert_model_parallel_size != 0)); then
    echo "[baseline] world_size=${world_size} is not divisible by ETP*EP*PP=${expert_model_parallel_size}." >&2
    exit 2
fi
non_expert_data_parallel_size=$((world_size / non_expert_model_parallel_size))
expert_data_parallel_size=$((world_size / expert_model_parallel_size))

case "${trainer_mode}" in
    sync|colocate_async) ;;
    *)
        echo "[baseline] Unsupported BASELINE_TRAINER_MODE=${trainer_mode}; expected sync or colocate_async." >&2
        exit 2
        ;;
esac

trainer_mode_args=(
    trainer.use_v1=True
    "trainer.v1.trainer_mode=${trainer_mode}"
    transfer_queue.enable=True
)
if [[ "${trainer_mode}" == "colocate_async" ]]; then
    trainer_mode_args+=(
        "trainer.v1.colocate_async.num_warmup_batches=${num_warmup_batches}"
    )
fi

checkpoint_args=()
if [[ -n "${checkpoint_save_contents}" ]]; then
    checkpoint_args+=(
        "+actor_rollout_ref.actor.checkpoint.save_contents=${checkpoint_save_contents}"
    )
fi

submit_cmd=(
    "${RAY_BIN}" job submit
    --no-wait
    --address="${RAY_API_URL}"
    --runtime-env "${RUNTIME_ENV}"
    --
    env -C "${SRC_ROOT}"
    "${PYTHON_BIN}" -m verl.trainer.main_ppo
    --config-name=ppo_megatron_trainer.yaml
    'hydra.searchpath=[pkg://verl.trainer.config]'
    "${trainer_mode_args[@]}"
    "data.train_files=${TRAIN_FILE}"
    "data.val_files=${TEST_FILE}"
    data.prompt_key=prompt
    data.filter_overlong_prompts=True
    data.truncation=error
    "data.max_prompt_length=${max_prompt_length}"
    "data.max_response_length=${max_response_length}"
    "data.train_batch_size=${train_prompt_bsz}"
    data.return_raw_chat=True
    "actor_rollout_ref.rollout.n=${n_resp_per_prompt}"
    "actor_rollout_ref.actor.policy_loss.loss_mode=${loss_mode}"
    "actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode}"
    "algorithm.adv_estimator=${adv_estimator}"
    "algorithm.rollout_correction.bypass_mode=${bypass_mode}"
    "algorithm.rollout_correction.rollout_is=${rollout_is}"
    "algorithm.rollout_correction.rollout_is_threshold=${rollout_is_threshold}"
    "algorithm.rollout_correction.rollout_is_batch_normalize=${rollout_is_batch_normalize}"
    "algorithm.rollout_correction.rollout_rs=${rollout_rs}"
    "algorithm.rollout_correction.rollout_rs_threshold=${rollout_rs_threshold}"
    algorithm.use_kl_in_reward=False
    algorithm.kl_ctrl.kl_coef=0.0
    "actor_rollout_ref.model.path=${MODEL_PATH}"
    actor_rollout_ref.model.trust_remote_code=True
    actor_rollout_ref.model.use_remove_padding=True
    "actor_rollout_ref.model.use_fused_kernels=${use_fused_kernels}"
    "+actor_rollout_ref.model.override_config.model_config.max_position_embeddings=$((max_prompt_length + max_response_length))"
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.kl_loss_coef=0.0
    actor_rollout_ref.actor.clip_ratio_low=4e-4
    actor_rollout_ref.actor.clip_ratio_high=4e-4
    actor_rollout_ref.actor.clip_ratio_c=10.0
    actor_rollout_ref.actor.use_dynamic_bsz=True
    "actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz}"
    "actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len}"
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.optim.lr_decay_style=constant
    actor_rollout_ref.actor.optim.weight_decay=0.1
    "actor_rollout_ref.actor.optim.lr_decay_steps=${lr_decay_steps}"
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_offload_fraction=1.0
    +actor_rollout_ref.actor.optim.override_optimizer_config.overlap_cpu_optimizer_d2h_h2d=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.use_precision_aware_optimizer=True
    +actor_rollout_ref.actor.optim.override_optimizer_config.optimizer_cpu_offload=True
    actor_rollout_ref.actor.megatron.use_mbridge=True
    actor_rollout_ref.actor.megatron.vanilla_mbridge=True
    actor_rollout_ref.actor.megatron.use_dist_checkpointing=False
    actor_rollout_ref.actor.megatron.use_remove_padding=True
    actor_rollout_ref.actor.megatron.param_offload=True
    actor_rollout_ref.actor.megatron.grad_offload=True
    actor_rollout_ref.actor.megatron.optimizer_offload=True
    "actor_rollout_ref.actor.megatron.tensor_model_parallel_size=${train_tp}"
    "actor_rollout_ref.actor.megatron.pipeline_model_parallel_size=${train_pp}"
    "actor_rollout_ref.actor.megatron.context_parallel_size=${train_cp}"
    "actor_rollout_ref.actor.megatron.expert_model_parallel_size=${train_ep}"
    "actor_rollout_ref.actor.megatron.expert_tensor_parallel_size=${train_etp}"
    +actor_rollout_ref.actor.megatron.override_transformer_config.apply_rope_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.masked_softmax_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_activation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.bias_dropout_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.gradient_accumulation_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.deallocate_pipeline_outputs=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.persist_layer_norm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_grouped_gemm=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_permute_fusion=True
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_token_dispatcher_type=alltoall
    +actor_rollout_ref.actor.megatron.override_transformer_config.moe_router_dtype=fp32
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_method=uniform
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_granularity=full
    +actor_rollout_ref.actor.megatron.override_transformer_config.recompute_num_layers=1
    actor_rollout_ref.actor.router_replay.mode=R3
    actor_rollout_ref.rollout.enable_rollout_routing_replay=True
    actor_rollout_ref.actor.entropy_coeff=0
    "${checkpoint_args[@]}"
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True
    "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}"
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True
    "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len}"
    "actor_rollout_ref.rollout.prompt_length=${max_prompt_length}"
    "actor_rollout_ref.rollout.response_length=${max_response_length}"
    actor_rollout_ref.rollout.multi_turn.enable=True
    actor_rollout_ref.rollout.multi_turn.max_parallel_calls=1
    ++actor_rollout_ref.rollout.multi_turn.format=qwen3_coder
    "actor_rollout_ref.rollout.agent.num_workers=${NUM_AGENT_WORKERS}"
    ++actor_rollout_ref.rollout.agent.agent_loop_manager_class=uni_agent.framework.entry.AgentFrameworkRolloutAdapter
    ++actor_rollout_ref.rollout.custom.agent_framework.framework_class_fqn=env.framework.P2AAgentFramework
    "++actor_rollout_ref.rollout.custom.agent_framework.gateway_count=${GATEWAY_COUNT}"
    "++actor_rollout_ref.rollout.custom.agent_framework.log_dir=${AGENT_LOG_DIR}"
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_fqn=env.task_runner.run_task
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.dispatch_mode=inline_async
    "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.max_concurrent_sessions=${AGENT_CONCURRENCY}"
    "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.task_config_path=${TASK_CONFIG_PATH}"
    "++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.model_name=$(basename "${MODEL_PATH}")"
    ++actor_rollout_ref.rollout.custom.agent_framework.agent_runners.task.runner_kwargs.report_reward=True
    ++actor_rollout_ref.rollout.custom.agent_framework.use_reward_loop_worker=False
    "actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_memory_utilization}"
    "actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp}"
    "actor_rollout_ref.rollout.expert_parallel_size=${infer_ep}"
    actor_rollout_ref.rollout.enable_chunked_prefill=True
    "actor_rollout_ref.rollout.max_model_len=$((max_prompt_length + max_response_length))"
    "actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length))"
    actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.top_p=1.0
    actor_rollout_ref.rollout.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95
    actor_rollout_ref.rollout.val_kwargs.top_k=-1
    actor_rollout_ref.rollout.val_kwargs.do_sample=True
    actor_rollout_ref.rollout.val_kwargs.n=1
    "actor_rollout_ref.rollout.name=${rollout_name}"
    actor_rollout_ref.rollout.mode=async
    actor_rollout_ref.rollout.calculate_log_probs=True
    actor_rollout_ref.hybrid_engine=True
    "actor_rollout_ref.nccl_timeout=${nccl_timeout}"
    actor_rollout_ref.rollout.enforce_eager=False
    actor_rollout_ref.rollout.free_cache_engine=True
    "actor_rollout_ref.rollout.disable_log_stats=${disable_log_stats}"
    "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${update_weights_bucket_megabytes}"
    actor_rollout_ref.ref.megatron.use_dist_checkpointing=False
    actor_rollout_ref.ref.megatron.param_offload=True
    "actor_rollout_ref.ref.megatron.tensor_model_parallel_size=${train_tp}"
    "actor_rollout_ref.ref.megatron.pipeline_model_parallel_size=${train_pp}"
    "actor_rollout_ref.ref.megatron.context_parallel_size=${train_cp}"
    "actor_rollout_ref.ref.megatron.expert_model_parallel_size=${train_ep}"
    "actor_rollout_ref.ref.megatron.expert_tensor_parallel_size=${train_etp}"
    reward.reward_manager.name=dapo
    +reward.reward_kwargs.overlong_buffer_cfg.enable=False
    +reward.reward_kwargs.overlong_buffer_cfg.len=4096
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0
    +reward.reward_kwargs.overlong_buffer_cfg.log=False
    "+reward.reward_kwargs.max_resp_len=${max_response_length}"
    "trainer.logger=['console','wandb']"
    "trainer.project_name=${project_name}"
    "trainer.experiment_name=${exp_name}"
    "trainer.val_before_train=${val_before_train}"
    "trainer.save_freq=${save_freq}"
    "trainer.test_freq=${test_freq}"
    "trainer.total_epochs=${total_epochs}"
    trainer.resume_mode=auto
    trainer.log_val_generations=10
    "trainer.default_local_dir=${CKPTS_DIR}"
    "trainer.nnodes=${NNODES}"
    "trainer.n_gpus_per_node=${NGPUS_PER_NODE}"
)

echo "[baseline] Ray endpoint: ${RAY_API_URL}"
echo "[baseline] nodes=${NNODES}, gpus_per_node=${NGPUS_PER_NODE}"
echo "[baseline] trainer_mode=${trainer_mode}, prompts=${train_prompt_bsz}, responses_per_prompt=${n_resp_per_prompt}"
echo "[baseline] parallelism=TP${train_tp}/PP${train_pp}/CP${train_cp}/EP${train_ep}/ETP${train_etp}, non_expert_dp=${non_expert_data_parallel_size}, expert_dp=${expert_data_parallel_size}"
echo "[baseline] train=${TRAIN_FILE}"
echo "[baseline] val=${TEST_FILE}"
echo "[baseline] model=${MODEL_PATH}"
echo "[baseline] checkpoints=${CKPTS_DIR}"
echo "[baseline] task_config=${TASK_CONFIG_PATH}, gateway_count=${GATEWAY_COUNT}, concurrency=${AGENT_CONCURRENCY}"

if [[ "${SYNC_DRY_RUN:-0}" == "1" ]]; then
    printf '[baseline] command:'
    printf ' %q' "${submit_cmd[@]}"
    printf '\n'
    exit 0
fi

"${submit_cmd[@]}"
