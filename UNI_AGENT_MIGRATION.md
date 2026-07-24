# Uni-Agent Migration

TraceAnalyzer keeps Uni-Agent and its nested verl checkout pristine. P2A-specific
training, sandbox, task, data, and analysis behavior lives in this repository.

## Pinned Stack

| Component | Pin |
|---|---|
| Uni-Agent | `3d4e40140db0455c7a3c669a68c58fc7c0ae8095` |
| verl | `78bba31d1a6b95084d83f8bddd59c190ca704144` |
| verl package version | `0.9.0.dev0` |

Initialize both levels after cloning:

```bash
git submodule update --init --recursive
uv sync --locked --extra train --extra dev
```

All Python commands from this directory must use the locked environment:

```bash
PYTHONPATH=uni-agent/verl:uni-agent:. uv run python -c \
  "import uni_agent, verl, env.sandbox, env.tasks"
```

## Agent Framework Integration

The pinned Uni-Agent uses Agent Framework Task Configs instead of the removed
`AgentEnv`/agent-loop interfaces.

- `env/sandbox.py` registers `arl` and `nexus` Sandbox providers.
- `env/tasks.py` registers the local `r2e_gym` and `swe_bench` tasks.
- `env/task_runner.py` resolves native Task Config rows and translates existing
  `tools_kwargs.env` plus `tools_kwargs.reward` rows at runtime.
- `env/framework.py` attaches instance IDs and decoded model responses to rollout
  records for P2A scoring and dashboard inspection.
- `env/agent_config_*.yaml` are Task Config files despite their retained
  compatibility filenames.
- `config/runtime_env.yaml` is the repository-owned Ray runtime template.

The local providers delegate lifecycle and file/command operations to
`env.deployment` and `env.runtime`. ARL connects through `arl-env`; Nexus uses the
Nexus runtime API. Neither path modifies the Uni-Agent submodule.

## Dataset Schema

`scripts/build_data.py` writes a native Task Config at
`extra_info.tools_kwargs.task`:

```text
task:
  name: r2e_gym | swe_bench
  prompt: [...]
  sandbox:
    image: ...
    sandbox_kwargs: ...
  metadata:
    instance_id: ...
```

The builder also writes explicit `env` and `reward` aliases so existing bonus-map
and dashboard artifacts remain readable. Launchers can consume both old and new
parquets through `env.task_runner.run_task`.

Generate datasets with:

```bash
PYTHONPATH=uni-agent/verl:uni-agent:. uv run python scripts/build_data.py r2e \
  --out ../../datasets/p2a/r2e_gym_subset_p2a.parquet

PYTHONPATH=uni-agent/verl:uni-agent:. uv run python scripts/build_data.py swebench-verified \
  --out ../../datasets/p2a/swe_bench_verified.parquet
```

## Four-Node Official Baseline

`scripts/train_qwen3_moe_official_4node.sh` adapts the current upstream
Qwen3-Coder MoE recipe to 4 nodes / 32 GPUs:

- `colocate_async`
- 64 prompts x 8 responses per step
- TP4 / PP1 / CP4 / EP8 / ETP1
- 8 Agent Framework gateways
- 512 maximum in-flight task sessions
- Nexus `purpose=test`

PP is 1 rather than the upstream 8-node recipe's PP2. With 32 GPUs this keeps
the non-expert decomposition at TP4 x PP1 x CP4 = 16 and gives data parallel
degree 2. Expert decomposition is ETP1 x EP8 x PP1 = 8 and gives expert data
parallel degree 4.

Validate the command without submitting a Ray job:

```bash
SYNC_DRY_RUN=1 \
MODEL_PATH=../../models/Qwen3-Coder-30B-A3B-Instruct \
TRAIN_FILE=../../datasets/p2a/r2e_gym_subset_p2a.train.parquet \
TEST_FILE=../../datasets/p2a/swe_bench_verified.parquet \
bash scripts/train_qwen3_moe_official_4node.sh
```

Run it only after the 4-node Ray cluster and GPU runtime are ready:

```bash
bash scripts/train_qwen3_moe_official_4node.sh
```

## Fully Async P2A

`scripts/train_p2a.sh` remains the fully async P2A launcher. It uses
`p2a.main`, the pinned verl fully-async policy modules, and the same Agent
Framework task runner.

Leave `P2A_BONUS_MAP_DIR` unset for a vanilla fully-async baseline. Set it to a
matching map directory to enable P2A advantage reshaping:

```bash
export P2A_BONUS_MAP_DIR="$PWD/data/bonus_maps/r2e-gym-subset"
bash scripts/train_p2a.sh
```

## ARL Helper

Provide credentials and endpoints through the environment or `.secrets/ips.sh`;
never commit them:

```bash
source .secrets/ips.sh
: "${ARL_GATEWAY_URL:?set ARL_GATEWAY_URL}"
```

The helper prepares repository-owned runtime and Task Config files:

```bash
bash scripts/uni_agent_arl.sh prepare
bash scripts/uni_agent_arl.sh smoke
bash scripts/uni_agent_arl.sh data
bash scripts/uni_agent_arl.sh debug
```

`smoke` is the live gate before an ARL launch. It verifies sandbox startup,
persistent shell state, and file upload.

## Bonus-Map Precompute

Dynamic precompute uses the same registered Sandbox providers:

```bash
PYTHONPATH=uni-agent/verl:uni-agent:. P2A_DEPLOYMENT=arl \
  uv run python -m p2a.precompute.precompute_bonus_maps \
  ../../datasets/p2a/r2e_gym_subset_p2a.train.parquet \
  --output_dir data/bonus_maps/r2e-gym-subset \
  --mode dynamic \
  --sandbox_backend uni_agent \
  --n_parallel 4 \
  --save_trace_sidecars
```

Start with low parallelism because each item creates a sandbox.

## Verification

CPU-side checks:

```bash
PYTHONPATH=uni-agent/verl:uni-agent:. uv run pytest -q tests
PYTHONPATH=uni-agent/verl:uni-agent:. uv run ruff check \
  env p2a scripts tests
bash -n scripts/train_sync_baseline.sh \
  scripts/train_qwen3_moe_official_4node.sh \
  scripts/train_p2a.sh scripts/uni_agent_arl.sh
```

GPU imports and an actual Ray launch still require the cluster's CUDA,
TransformerEngine, mbridge, vLLM, NCCL, Nexus/ARL credentials, and shared model
assets. CPU validation does not replace that live gate.
