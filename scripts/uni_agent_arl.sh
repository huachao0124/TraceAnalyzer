#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: src/scripts/uni_agent_arl.sh prepare|data|smoke|debug|all

prepare  Write ARL runtime_env.yaml and task_config.yaml into $RAY_DATA_HOME.
data     Generate ARL-backed R2E-Gym-Subset train parquet.
smoke    Boot one ARL sandbox and verify SDK runtime persistence/upload.
debug    Run the ARL-aligned train_p2a.sh launcher.
all      Run prepare, data, then debug.
EOF
}

SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC_ROOT="${SRC_DIR}"
source "${SRC_DIR}/scripts/lib.sh"
UNI_AGENT_DIR="${SRC_DIR}/uni-agent"
RAY_DATA_HOME="${RAY_DATA_HOME:-${HOME}/verl}"
DATA_DIR="${RAY_DATA_HOME}/data/swe_agent"
RUNTIME_ENV="${RUNTIME_ENV:-${DATA_DIR}/runtime_env_arl.yaml}"
TASK_CONFIG_PATH="${TASK_CONFIG_PATH:-${AGENT_CONFIG_PATH:-${DATA_DIR}/task_config_arl.yaml}}"
TRAIN_FILE="${TRAIN_FILE:-${DATA_DIR}/r2e_gym_subset_p2a.train.parquet}"
TEST_FILE="${TEST_FILE:-${DATA_DIR}/r2e_gym_subset_p2a.train.parquet}"

require_uni_agent() {
  if [[ ! -d "${UNI_AGENT_DIR}/uni_agent" || ! -d "${UNI_AGENT_DIR}/verl" ]]; then
    echo "Missing Uni-Agent or nested verl under ${UNI_AGENT_DIR}" >&2
    echo "Run: git -C src submodule update --init --recursive" >&2
    exit 1
  fi
}

prepare() {
  require_uni_agent
  mkdir -p "$DATA_DIR"
  if [[ ! -f "$RUNTIME_ENV" ]]; then
    cp "${SRC_DIR}/config/runtime_env.yaml" "$RUNTIME_ENV"
  fi
  cp "${SRC_DIR}/env/agent_config_arl.yaml" "$TASK_CONFIG_PATH"
  (cd "$SRC_DIR" && uv run python -m p2a.runtime_env "$RUNTIME_ENV" --src-root "$SRC_DIR" --drop-working-dir --env-profile arl)

  local debug_concurrency="${UNI_AGENT_DEBUG_CONCURRENCY:-4}"
  local debug_max_turns="${UNI_AGENT_DEBUG_MAX_TURNS:-10}"
  local debug_action_timeout="${UNI_AGENT_DEBUG_ACTION_TIMEOUT:-120}"
  local debug_eval_timeout="${UNI_AGENT_DEBUG_EVAL_TIMEOUT:-300}"

  sed -i -E "s/^    max_steps: .*/    max_steps: ${debug_max_turns}/" "$TASK_CONFIG_PATH"
  sed -i -E "s/^    action_timeout: .*/    action_timeout: ${debug_action_timeout}/" "$TASK_CONFIG_PATH"
  sed -i -E "s/^  eval_timeout: .*/  eval_timeout: ${debug_eval_timeout}/" "$TASK_CONFIG_PATH"

  cat <<EOF
Prepared Uni-Agent ARL config:
  RUNTIME_ENV=${RUNTIME_ENV}
  TASK_CONFIG_PATH=${TASK_CONFIG_PATH}
  AGENT_CONCURRENCY=${debug_concurrency}
  TRAIN_FILE=${TRAIN_FILE}
  TEST_FILE=${TEST_FILE}
EOF
}

data() {
  require_uni_agent
  mkdir -p "$DATA_DIR"
  # Writes the full R2E parquet plus the skip-filtered *.train.parquet used by RL.
  cd "$SRC_DIR"
  PYTHONPATH=.:uni-agent:uni-agent/examples/data_preprocess \
    uv run python scripts/build_data.py r2e --out "${DATA_DIR}/r2e_gym_subset_p2a.parquet"
}

smoke() {
  require_uni_agent
  cd "$SRC_DIR"
  # Prefer a pair-diag image from the built R2E parquet; otherwise require ARL_SMOKE_IMAGE.
  local image="${ARL_SMOKE_IMAGE:-}"
  if [[ -z "$image" && -f "$TRAIN_FILE" ]]; then
    image="$(uv run python -c "
import json, pandas as pd
ei = pd.read_parquet('$TRAIN_FILE')['extra_info'].iloc[0]
ei = json.loads(ei) if isinstance(ei, str) else ei
print(ei['tools_kwargs']['env']['deployment']['image'])
" 2>/dev/null || true)"
  fi
  if [[ -z "$image" ]]; then
    echo "Set ARL_SMOKE_IMAGE to a pair-diag image, or run '${BASH_SOURCE[0]} data' first to build ${TRAIN_FILE}." >&2
    exit 1
  fi
  uv run python -m env.smoke --image "$image"
}

debug() {
  require_uni_agent
  if [[ ! -f "$RUNTIME_ENV" ]]; then
    echo "Missing $RUNTIME_ENV; run prepare first." >&2
    exit 1
  fi
  if [[ ! -f "$TASK_CONFIG_PATH" ]]; then
    echo "Missing $TASK_CONFIG_PATH; run prepare first." >&2
    exit 1
  fi

  cd "$SRC_DIR"

  # TRAIN_FILE and the default debug TEST_FILE use the skip-filtered R2E train parquet.

  export RAY_DATA_HOME
  export TRAIN_FILE
  export TEST_FILE
  export RUNTIME_ENV
  export TASK_CONFIG_PATH
  export AGENT_CONCURRENCY="${UNI_AGENT_DEBUG_CONCURRENCY:-4}"
  if [[ -n "${P2A_BONUS_MAP_DIR:-}" ]]; then
    echo "P2A_BONUS_MAP_DIR is set; debug will run P2A advantage reshape, not pure baseline." >&2
  fi
  bash scripts/train_p2a.sh
}

cmd="${1:-}"
case "$cmd" in
  prepare)
    prepare
    ;;
  data)
    data
    ;;
  smoke)
    smoke
    ;;
  debug)
    debug
    ;;
  all)
    prepare
    data
    debug
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
