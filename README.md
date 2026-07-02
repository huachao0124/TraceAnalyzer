# P2A on Uni-Agent + ARL

Program-Analysis-based Process Advantage (P2A) for SWE agentic RL, implemented on
the **Uni-Agent** training stack with our **ARL** cluster as the sandbox backend.
P2A reshapes the per-step RL advantage using a precomputed **bonus map**: a
captured **Graph** from failing-test execution, a rewardable non-test Graph
slice from the first post-test node to the terminal patched root cause, and a
diagnostic **Path** from the issue symptom to that root cause when a symptom
anchor can be matched. Steps whose agent actions land on rewardable Graph nodes
get a larger advantage.

Everything is **self-contained**: data comes from HuggingFace, images from the
pair-diag mirror of the original R2E images. There is **no dependency on the old
`src-backup` fork**.

> **Terminology**: the repo-root [`GLOSSARY.md`](../GLOSSARY.md) is the canonical
> term table for the **Graph / Path / Trace** triad and the full role/metric
> vocabulary used in this README, the code, and the dashboard. Match it.

## Bonus-map anchors

Dynamic bonus maps keep observed Graph nodes, distances, roles, rewardability,
and callable source as bonus-map data. Target semantics are:
`test_harness -> test_adapter -> symptom -> intermediate/fix_adapter -> root_cause`.
Only `test_harness` is excluded from `hop_max`, source capture, and read
matching; every non-test node is rewardable. `test_adapter` names rewardable
non-test frames before the selected issue symptom anchor;
`fix_adapter` names golden-patch-modified callables that sit upstream of the
terminal patched root cause. Legacy artifacts may still use `pre_symptom` as an
alias for `test_adapter`. The training ground-truth anchor is the first
non-test node after the test harness, not the issue symptom. The issue symptom
anchor only defines the diagnostic Path and Path metrics; if no high-confidence
issue symptom anchor matches the captured Graph, Path metrics are unavailable
but Graph reward remains defined over all non-test F2P→terminal-root frames.

> **ARL is the sandbox, not a "remote".** The `arl-env` SDK connects directly to the
> ARL Gateway (`ARL_GATEWAY_URL`) to boot a per-instance container sandbox where tests
> and P2A instrumentation run (bonus-map precompute, training rollouts); it is reachable
> directly from CPU hosts. This is separate from VRC's `remote` facility, which targets
> the **GPU server** for command debugging — ARL gateway reachability is independent of
> `vrc remote`.

Default HuggingFace datasets and models are shared across sibling projects.
TraceAnalyzer-generated artifacts stay inside this checkout under `data/`.

| Asset | Default location from `src/` | Override |
|---|---|---|
| Models | `../../models/<repo-name>` | `MODEL_PATH` / `P2A_MODELS_DIR` / `P2A_MODEL_REPO` |
| Datasets and generated parquets | `../../datasets`, with P2A parquets in `../../datasets/p2a` | `P2A_DATASETS_DIR` / `DATA` / `P2A_SHARED_ROOT` |
| Project artifacts | `data/` | `P2A_ARTIFACTS_DIR` |

If a dataset/model is already present in the shared location, scripts read it
directly. If it is missing, the script downloads it from HuggingFace and saves
it there. Bonus maps, SQLite caches, rollout dumps, analysis reports, eval
details, and dashboard snapshots are project artifacts and default to `data/`.

## Git-metadata sanitization (anti-leak, issue #29)

Model-facing rollout/eval sandboxes must not expose the original `/testbed` (or
`/app`) `.git`: its history reaches the future/fixed commits, remote/PR refs, and
reflogs baked into the image, so a shell-capable agent can recover the gold patch with
`git log --all` / `git show` instead of solving the task. `scripts/build_data.py`
appends a two-phase tail (`p2a/sandbox_git.py`) to each model-facing row's
`post_setup_cmd`:

1. restore the intended code state (R2E checks out the buggy commit; SWE-bench checks
   out `base_commit`);
2. replace `.git` with a single neutral `baseline` commit of the worktree.

`git diff HEAD` and `git apply` keep working against the baseline, so patch extraction
and reward/eval are unaffected (SWE-bench test-file reset compares against `HEAD`). The
tail is wrapped in `# >>>P2A_GIT_SANITIZE>>>` sentinels; the **precompute** path strips
it (`strip_sanitize_block`) because bonus-map instrumentation needs the real history.

Canary (issue acceptance): `scripts/canary_git_sanitize.py offline` checks the
command-generation and strip invariants anywhere; `... arl --parquet <p> --n 1` boots a
real model-facing sandbox and asserts `git log --all` shows only the baseline, no
remote/tag refs remain, and `git diff HEAD` still captures a model edit.

## Current capabilities

This repo now has four mostly independent surfaces:

| Surface | Use it for | Primary entry points |
|---|---|---|
| Data setup | Build R2E-Gym subset, SWE-bench Verified, the default SWE-bench hard validation split, and the optional SWE-Bench-Pro Python eval subset. | `scripts/setup.sh data ...`, `scripts/build_data.py` |
| P2A training | Run Uni-Agent baseline or P2A advantage reshaping on ARL/Ray. | `scripts/main.sh`, `scripts/train_p2a.sh` |
| Graph diagnostics | Precompute dynamic/static bonus maps and score whether rollouts read the fault-propagation graph. | `scripts/precompute_eval_bonus_maps.sh`, `p2a.eval_fault_localization`, live `val-p2a/*` validation metrics |
| Third-party baselines | Run an OpenAI-compatible external model through the same Uni-Agent + ARL SWE/R2E environment and produce rollout/localization artifacts. | `scripts/main_3rd.sh`, `scripts/third_party_eval.sh`, `config/third_party_eval.deepseek.example.yaml` |

The shell scripts are layered deliberately: `scripts/setup.sh` owns idempotent
data/dependency/map setup; `main.sh` and `main_3rd.sh` are high-level wrappers;
the remaining scripts are lower-level runners or runtime checks for explicit
control.

## Directory map

```
src/
  config/                     # ALL json/yaml config lives here
    startup_fixups.json       #   per-repo test-startup fixups (source patches + dep pins)
    bad_instances.json        #   R2E instances to exclude from training (skip-list)
    third_party_eval.deepseek.example.yaml # third-party OpenAI-compatible model config template
  scripts/
    build_data.py             # SINGLE data builder: r2e | swebench-verified | swebench-hard | swebench-pro | skip-list
    uni_agent_arl.sh          # prepare/data/smoke/debug launcher (ARL config)
    setup.sh                  # idempotent data/dependency/eval-map setup helpers
    ray_setup.sh              # bring up Ray and smoke-check Ray Jobs
    lib.sh                    # sourced helpers: HF/path resolution, local-env, node-local staging
    main.sh                   # one-shot baseline launcher
    main_3rd.sh               # one-shot third-party OpenAI-compatible rollout baseline
    train_p2a.sh              # training launcher (baseline OR P2A)
    third_party_eval.sh       # lower-level third-party rollout pass-through
    precompute_eval_bonus_maps.sh # eval-map helper for validation diagnostics
    check_deps_cpu.sh         # CPU dependency/import smoke check
    check_uni_agent_runtime.py # GPU/runtime import smoke check
  p2a/
    core.py                   # bonus-map load + read->Graph match + m(d)=m_max^(1-d) multiplier
    trainer.py                # apply_p2a_reshape: capture agent reads -> reshape advantage
    main.py                   # training entry; P2AFullyAsyncTrainer (vanilla if P2A_BONUS_MAP_DIR unset)
    rollouter.py              # validation rollouter wrapper for live graph diagnostics
    validation_metrics.py     # aggregate val-p2a/* localization metrics
    third_party_eval.py       # OpenAI-compatible external model rollout harness
    test_setup.py             # startup_fixup_command(repo) — loads config/startup_fixups.json
    sandbox_git.py            # git-metadata sanitizer for model-facing rollout/eval (issue #29)
    trace.py                  # Graph-capture instrumentation + legacy call_graph artifact build
    eval_fault_localization.py # offline eval rollout read->fault-localization metrics
    hf_assets.py              # shared HuggingFace model/dataset path helpers
    runtime_env.py            # Ray runtime-env path normalization helpers
    precompute/precompute_bonus_maps.py  # build bonus maps on the ARL backend
    skip_cases.py             # load_skip_ids() from config/bad_instances.json
  env/                        # ARL deployment + runtime (implements swe-rex AbstractRuntime; no swe-rex server)
  uni-agent/                  # UNMODIFIED Uni-Agent submodule (swe-rex is its runtime interface)
```

## Pipeline

### Step 0. Prepare the shared checkout and venv

Clone the repo onto the shared disk first. Run all following commands from the
repo root:

```bash
git clone git@github.com:Siyuexi/TraceAnalyzer.git
cd TraceAnalyzer
git submodule update --init --recursive
export UV_PYTHON_INSTALL_DIR=$PWD/.uv-python
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_PATH=$CUDA_HOME
uv python install --managed-python 3.11
"$(uv python find --managed-python --no-project 3.11)" -m venv --clear --copies .venv
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv sync --locked --extra train --extra gpu
```

The `uv` path above is also the fused Megatron / mbridge training runtime.
`scripts/main.sh`, `scripts/train_p2a.sh`, and `scripts/ray_setup.sh` default to
`.venv` and native CUDA 13.0. Verify the runtime before launching a cluster job:

```bash
cd TraceAnalyzer
git submodule update --init --recursive
export CUDA_HOME=/usr/local/cuda-13.0
export CUDA_PATH=$CUDA_HOME
UV_PROJECT_ENVIRONMENT=$PWD/.venv uv run --no-sync python scripts/check_uni_agent_runtime.py
```

The locked GPU stack is:

| Component | Version / source |
|---|---|
| CUDA toolkit | `/usr/local/cuda-13.0` |
| PyTorch | CUDA 13 wheels from `https://download.pytorch.org/whl/cu130` |
| vLLM | `vllm==0.11.2` |
| cupy | `cupy-cuda13x==13.6.0` |
| TransformerEngine / Megatron | resolved by `uv sync --locked --extra train --extra gpu` |
| mbridge | `git+https://github.com/ISEEKYAN/mbridge.git` |

`scripts/ray_setup.sh` stages the selected venv to each node-local runtime path,
so head and workers run the same Python stack.

### Step 1. Set common paths

Set these once per shell:

```bash
export UV_PROJECT_ENVIRONMENT="${UV_PROJECT_ENVIRONMENT:-$PWD/.venv}"
export PATH="$UV_PROJECT_ENVIRONMENT/bin:$PATH"
export RAY_DATA_HOME="${RAY_DATA_HOME:-$HOME/verl}"
export DATA="${DATA:-../../datasets/p2a}"
export MODEL="${MODEL:-../../models/Qwen3-Coder-30B-A3B-Instruct}"
# Private endpoints stay outside git.
source .secrets/ips.sh
: "${ARL_GATEWAY_URL:?set ARL_GATEWAY_URL or create .secrets/ips.sh}"
# If submitting from the Ray head node, the local dashboard endpoint is enough.
export RAY_API_SERVER_ADDRESS="${RAY_API_SERVER_ADDRESS:-http://localhost:8265}"
# Ray cluster ports. 6379 is Ray GCS; 8265 is Ray dashboard / Jobs.
export RAY_GCS_PORT="${RAY_GCS_PORT:-6379}"
export RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
export RAY_SSH_OPTS="${RAY_SSH_OPTS:--p 36000 -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8}"
# Ray 2.55 prestarts one Python worker per advertised CPU. Limit this on
# large GPU nodes so dashboard-agent / Jobs startup is not blocked by hundreds
# of shared-venv worker imports.
export NUM_CPUS="${NUM_CPUS:-64}"
```

### Step 2. Build the training and validation parquets

```bash
PYTHONPATH=.:uni-agent:uni-agent/verl:uni-agent/examples/data_preprocess \
  uv run python scripts/build_data.py r2e --out $DATA/r2e_gym_subset_p2a.parquet

PYTHONPATH=.:uni-agent:uni-agent/examples/data_preprocess \
  uv run python scripts/build_data.py swebench-verified --out $DATA/swe_bench_verified.parquet

PYTHONPATH=.:uni-agent:uni-agent/examples/data_preprocess \
  uv run python scripts/build_data.py swebench-hard --out $DATA/swe_bench_verified_hard.parquet

# Phase 1 SWE-Bench-Pro eval subset: Python repos only, never training.
export P2A_SWEBENCH_PRO_SCRIPTS_DIR=/path/to/SWE-bench_Pro-os/run_scripts
PYTHONPATH=.:uni-agent:uni-agent/verl:uni-agent/examples/data_preprocess \
  uv run python scripts/build_data.py swebench-pro \
    --out $DATA/swe_bench_pro.parquet \
    --scripts-dir "$P2A_SWEBENCH_PRO_SCRIPTS_DIR"
```

`swebench-hard` is a filtered parquet, not a separate HuggingFace cache directory.
The two upstream cache directories are expected: `SWE-Bench-Verified/` carries
R2E-Gym eval rows, while `SWE-bench_Verified/` carries Princeton difficulty labels.
`swebench-pro` stores the upstream `repo_language`, normalized `FAIL_TO_PASS` /
`PASS_TO_PASS`, the mirrored `jefzda/sweap-images:{dockerhub_tag}` image, and
the per-instance SWE-Bench-Pro run script/parser from the official open-source
`run_scripts/` checkout. `--scripts-dir` is required because the HuggingFace
dataset does not carry `run_script.sh` or `parser.py`; setup rejects existing
SWE-Bench-Pro parquets with empty script fields. Pro images use `/app` as the
repository root, so the generated setup and verifier metadata keep that path
separate from SWE-bench Verified's `/testbed`. Dynamic ARL precompute also
requires the corresponding `sweap-images` tags to be present in the pair-diag
mirror; missing tags fail during sandbox startup as `ImagePullBackOff`.
Use the image preflight helper to generate or check the Phase-1 mirror list:

```bash
uv run python scripts/swebench_pro_images.py $DATA/swe_bench_pro.parquet --limit 5
uv run python scripts/swebench_pro_images.py $DATA/swe_bench_pro.parquet \
  --limit 5 --check-manifests --fail-on-missing-mirror
uv run python scripts/swebench_pro_images.py $DATA/swe_bench_pro.parquet \
  --emit-mirror-script > /tmp/swebench_pro_mirror_python.sh
```

### Step 3. Precompute training bonus maps for P2A

```bash
PYTHONPATH=.:uni-agent:uni-agent/verl P2A_DEPLOYMENT=arl \
  uv run python p2a/precompute/precompute_bonus_maps.py \
    $DATA/r2e_gym_subset_p2a.parquet \
    --output_dir data/bonus_maps/r2e-gym-subset \
    --mode dynamic --n_parallel 64
```

The setup wrapper also writes maps to `data/bonus_maps/<dataset>` by default:

```bash
bash scripts/setup.sh maps r2e-gym-subset
```

Set `P2A_BONUS_MAP_DIR` to the matching directory when enabling P2A training.

Each generated map includes Graph nodes and edges under the legacy artifact keys
`call_graph_nodes` and `call_graph_edges`, plus a per-node `source` snippet when
the sandbox file can be read. The edge list is diagnostic schema for topology
views; training still reshapes from node distances.

Skip this step for a pure baseline run. P2A training reads these maps through
`P2A_BONUS_MAP_DIR`.

### Step 4. Precompute eval maps for validation graph metrics

```bash
TEST_FILE=$DATA/swe_bench_verified_hard.parquet bash scripts/precompute_eval_bonus_maps.sh

EVAL_DATASET=swebench-pro \
  TEST_FILE=$DATA/swe_bench_pro.parquet \
  bash scripts/precompute_eval_bonus_maps.sh
```

Eval maps default to `data/bonus_maps/<dataset>`, such as
`data/bonus_maps/swebench-hard` or `data/bonus_maps/swebench-pro`. They are
diagnostic only: validation logging reads them, but the training reshape should
use the training split's `data/bonus_maps/r2e-gym-subset` directory.

### Step 5. Configure logging and GPU layout

Configure Weights & Biases before submitting training. Use online logging:

```bash
wandb login
# or export WANDB_API_KEY=...
```

Or use local offline logs:

```bash
export WANDB_MODE=offline
```

The launcher defaults to the Uni-Agent Qwen3 30B MoE 64-GPU shape. For a
4-node x 8-GPU cluster, override it with this 32-GPU starter profile:

```bash
export NNODES_TRAIN=2
export NNODES_ROLLOUT=2
export NGPUS_PER_NODE=8
```

This allocates 16 GPUs to trainer and 16 GPUs to rollout. Do not set
`NNODES_TRAIN=4 NNODES_ROLLOUT=4 NGPUS_PER_NODE=8` on a 32-GPU cluster; that
requests 64 GPUs.

### Step 6. Optional manual Ray restart

`scripts/main.sh` restarts Ray by default before submitting training. To restart
Ray manually instead, run this from the head node. It stops workers first, then
the head, then starts the head, starts workers, and finally submits a tiny Ray
Jobs smoke task:

```bash
HEAD_IP=<HEAD_IP>
export RAY_WORKER_HOSTS="<WORKER_IP_1> <WORKER_IP_2> <WORKER_IP_3>"
export P2A_LOCAL_ROOT=/tmp/p2a-traceanalyzer
export RAY_SSH_OPTS="${RAY_SSH_OPTS:--p 36000 -o BatchMode=yes -o StrictHostKeyChecking=no -o ConnectTimeout=8}"
bash scripts/ray_setup.sh "$HEAD_IP" restart-cluster
```

The script keeps the shared checkout as the source of truth, stages repo code,
`.uv-python`, and `.venv` to `$P2A_LOCAL_ROOT/TraceAnalyzer` on each node,
rewrites copied venv paths from the shared source to the local runtime path,
then starts Ray from the local venv. `P2A_STAGE_LOCAL_RUNTIME=1` is the default.
Data, models, checkpoints, and W&B output remain controlled by `DATA`,
`MODEL_PATH`/`MODEL`, `RAY_DATA_HOME`, and `WANDB_DIR`; keep those on shared
storage unless their access pattern becomes a measured bottleneck. The staging
path mainly reduces Ray control-plane / worker-import pressure during startup;
it is not expected to materially speed up steady-state training unless training
was blocked on shared-disk Python package reads.

The script starts `--head` on the head node and joins workers to that head over
ssh. It intentionally ignores the generic `PORT` environment variable; use
`RAY_GCS_PORT` if you need to override the Ray cluster port. `8080` is not a Ray
port in the VRC remote-debug setup. In staged mode, `NUM_CPUS` defaults to
`nproc`; in direct shared-disk mode it defaults to 64 to avoid prestarting
hundreds of workers from the shared venv.

To bypass local staging and run directly from the shared checkout:

```bash
export P2A_STAGE_LOCAL_RUNTIME=0
bash scripts/ray_setup.sh "$HEAD_IP" restart-cluster
```

If you cannot ssh from the head to workers, run the same script manually in this
order:

```bash
HEAD_IP=<HEAD_IP>
# on every worker first
bash scripts/ray_setup.sh "$HEAD_IP" stop
# on the head
bash scripts/ray_setup.sh "$HEAD_IP" start
# on every worker
bash scripts/ray_setup.sh "$HEAD_IP" start
# on the head
bash scripts/ray_setup.sh "$HEAD_IP" smoke
```

The `smoke` command does not restart Ray. It waits for the dashboard and submits
a tiny Ray Jobs task. If you submit training from the head node,
`RAY_API_SERVER_ADDRESS` can stay at `http://localhost:8265`; otherwise set it
to the Ray head dashboard URL.

### Step 7. Submit a baseline or P2A run

One-shot baseline from the Ray head:

```bash
bash scripts/main.sh
```

`scripts/main.sh` stages code/runtime locally like the other launchers, calls
`scripts/setup.sh` for idempotent dependency and data setup, restarts Ray through
`scripts/ray_setup.sh`, and keeps default `DATA` / `MODEL` paths anchored at the
shared checkout (`../../datasets/p2a` and `../../models/...`) instead of under
`/tmp`. It reads cluster-local endpoints from `.secrets/ips.sh` or existing
environment variables (`HEAD_IP`/`RAY_HEAD_IP`, `RAY_WORKER_HOSTS`,
`RAY_GCS_PORT`, `RAY_SSH_OPTS`). Override those env vars if the allocation
changes. To submit to an already-running Ray cluster without a restart, set
`P2A_RESTART_RAY=0`.

Baseline:

```bash
TRAIN_FILE=$DATA/r2e_gym_subset_p2a.train.parquet \
  TEST_FILE=$DATA/swe_bench_verified_hard.parquet \
  MODEL_PATH=$MODEL \
  P2A_EVAL_BONUS_MAP_DIR=data/bonus_maps/swebench-hard \
  P2A_EVAL_DETAILS_DIR=data/eval_details \
  bash scripts/train_p2a.sh
```

P2A:

```bash
TRAIN_FILE=$DATA/r2e_gym_subset_p2a.train.parquet \
  TEST_FILE=$DATA/swe_bench_verified_hard.parquet \
  MODEL_PATH=$MODEL \
  P2A_EVAL_BONUS_MAP_DIR=data/bonus_maps/swebench-hard \
  P2A_EVAL_DETAILS_DIR=data/eval_details \
  P2A_BONUS_MAP_DIR=data/bonus_maps/r2e-gym-subset \
  P2A_M_MAX=3.0 \
  P2A_CREDIT_GRANULARITY=step \
  bash scripts/train_p2a.sh
```

`P2A_CREDIT_GRANULARITY=step` is the default and preserves per-step reshape
behavior. Set `P2A_CREDIT_GRANULARITY=block` to group adjacent same-purpose
steps and apply credit to read blocks that touch the Graph.

### Step 8. Optional offline analysis

For an already-dumped rollout file or dump directory:

```bash
uv run python -m p2a.eval_fault_localization $ROLLOUT_JSONL \
  --bonus-map-dir data/bonus_maps/swebench-hard \
  --summary-out data/eval_faultloc_summary.json \
  --details-out data/eval_faultloc_details.jsonl
```

To open the unified HTML dashboard over an already-dumped rollout file or dump
directory:

```bash
uv run python scripts/p2a_dashboard.py $ROLLOUT_JSONL \
  --bonus-map-dir data/bonus_maps/swebench-hard \
  --port 8766
```

To write a portable static snapshot of the same HTML dashboard:

```bash
uv run python scripts/p2a_dashboard.py $ROLLOUT_JSONL \
  --bonus-map-dir data/bonus_maps/swebench-hard \
  --out-dir data/p2a_dashboard
```

The dashboard can also read scored validation details, Uni-Agent run directories,
and the third-party eval SQLite cache:

```bash
uv run python scripts/p2a_dashboard.py \
  --details data/eval_details \
  --bonus-map-dir data/bonus_maps/swebench-hard

uv run python scripts/p2a_dashboard.py \
  --log-dir /tmp/swebench_qwen3_coder \
  --bonus-map-dir data/bonus_maps/swebench-hard

uv run python scripts/p2a_dashboard.py \
  --db data/evals/traces.sqlite \
  --build-db data/evals/traces.dashboard.sqlite \
  --experiment-id public-swebench-hard-demo \
  --dataset swebench-hard \
  --bonus-map-dir data/bonus_maps/swebench-hard
```

If `--build-db` is omitted, the dashboard derives it next to the raw DB as
`<raw-stem>.dashboard.sqlite`, for example
`data/evals/traces.dashboard.sqlite`.

Server startup and page refreshes do not rebuild or migrate dashboard cache
data. To migrate legacy raw-DB dashboard caches into the build DB, run the
explicit one-shot command:

```bash
uv run python scripts/p2a_dashboard.py \
  --db data/evals/traces.sqlite \
  --build-db data/evals/traces.dashboard.sqlite \
  --bonus-map-dir data/bonus_maps/swebench-hard \
  --migrate-build-db
```

After confirming the build DB is populated, the raw DB can be slimmed explicitly.
This removes migrated dashboard detail/cache payloads from `quantitative_metrics`
and compacts `raw_rollouts.rollout_json` into metadata only; raw trace content
remains in the structured `messages_json`, `trajectory_json`, and
`p2a_step_traces_json` columns.

```bash
uv run python scripts/p2a_dashboard.py \
  --db data/evals/traces.sqlite \
  --build-db data/evals/traces.dashboard.sqlite \
  --slim-raw-db \
  --vacuum
```

Live DB dashboards keep a process-local snapshot cache and reuse it while the
underlying raw/build DB update token is unchanged. To enable admin-only
deletion, put the password in `.secrets/dashboard_admin.txt` or pass
`--admin-secret .secrets/dashboard_admin.txt`; without that file the dashboard
remains read-only and open for normal viewing. Admin deletion previews the DB
blast radius and removes only DB rows (`run_cells` plus cascaded rollouts and
metrics, and matching `experiments`). On-disk rollout artifacts are not deleted.
After admin login, the Admin panel can rebuild all or selected eval cells. A
rebuild reads raw rollouts from the raw eval DB and overwrites the corresponding
dashboard build DB rows.

The Overview tab is the dataset and eval-cell registry. Dataset-level
distributions count unique instances in a dataset/split, so five model runs over
the 45-instance `swebench-hard` split still show a distribution population of 45,
not 225 trajectories. Eval cells are the model/checkpoint/API-run comparison
unit: source kind (`local_training`, `local_inference`, or `third_party_api`),
experiment id, provider, dataset, and model label. Select a dataset first, then
select an eval cell/model before inspecting trajectories.

The Metrics tab is the model-level analysis surface for the selected dataset.
It renders one comparison table grouped by semantics: Graph metrics
(`Graph P.`, `Graph R.`, `Graph F1`) score agent reads against the real
dependency graph captured by instrumentation/failing-test execution; Outcome
metrics report task success and symptom/root-cause hits; Path metrics score the
issue symptom-to-root-cause subgraph/path with `Path P.`, `Path R.`, and
`Path F1`; Pattern, purpose-block, and efficiency/cost
metrics come after them. Cache-write metrics are hidden
until populated. In user-facing terminology, Graph means the captured
dependency graph, Path means the symptom-to-root-cause subgraph/path, and Trace
means the model/agent execution trajectory.
Dashboard DB persistence is split into two stores. The raw eval DB is written
during rollout and contains raw rollout JSON/messages/trajectories, issue
descriptions, golden patches, token/runtime data, run status, and artifact
references. The dashboard build DB is the dashboard-bound materialized cache: it
contains per-rollout pattern/detail rows and eval-cell metrics rows, but no raw
trace payloads. The raw DB stores raw trace content once in structured
`messages_json`, `trajectory_json`, and `p2a_step_traces_json` columns;
`rollout_json` is slim metadata and must not duplicate the full trace. Data
flows one way from raw DB to build DB only during admin rebuild, explicit
build-DB migration, or incremental rollout caching. Structured rollout/system
failures update only `run_cells.status/error` for overview and rerun
eligibility; they do not write `raw_rollouts`, `quantitative_metrics`, or
dashboard build-DB detail/metric rows. Dashboard static reads use the build DB
when all completed non-error rollouts in an eval cell have valid materialized
rows. If any completed non-error rollout in that cell is missing valid build
data, Metrics shows only basic progress, pass/resolved, token/cost, and length
metrics; symptom/root, Path, pattern, order, miracle, and purpose-block KPIs
stay empty until the cell is fully materialized. The Traces tab lists only
rollouts that actually have raw trace content; planned-but-unrolled cells are
omitted, and traces without materialized detail render without pattern tags. If
`--bonus-map-dir` is omitted,
the dashboard tries `data/bonus_maps/<dataset>` under the artifact root and uses
it only when it contains matching instance maps. If old DB rows do not carry
issue descriptions or golden patches, the dashboard fills them from
`--data-file` or the standard local dataset parquet for the selected dataset.
Node Source is bonus-map data: the dashboard reads full callable source from the
explicit or inferred P2A bonus-map directory, and DB `source_preview` fields are
only stale artifact fallbacks.
The Logs tab explains artifact/log-producing executions and only
mixes runs into a selected eval cell when they carry explicit eval-cell links;
unlinked logs are shown separately. The Traces tab is the micro-analysis
surface: narrow instance list on the left, graph plus purpose-block/step
timeline in the middle, and a wide right panel with parsed tool/action details,
separate reasoning/chat text, collapsible raw action/observation payloads, and
inline edit diffs when write actions provide old/new text. When an eval cell has
repeated rollouts for the same instance, the left instance row is split into
per-rollout success/failure color segments and the selected instance title
offers a rollout selector. Pattern tag toggles filter the instance list and the
rollout selector to rollouts matching all selected tags; undefined miracle,
reverse, or Path-hit markers do not match their tag. Copy-link controls produce
URL-hash permalinks for experiment, instance, rollout, and exact step locations,
and pasted links clear conflicting filters before navigating. Step colors and trace
markers come from P2A parser/scorer fields: reads, writes, execution errors,
root-cause edits, symptom/root-cause hits, and Path hits are computed
in `p2a/core.py`, `p2a/eval_fault_localization.py`, and
`p2a/dashboard_adapter.py`, then rendered by the frontend. Execution-failure
marks use structured tool status, nonzero exit codes, explicit error fields, or
traceback/command-failure output; source code that merely contains words such as
`Error` is not a failed step. Miracle means root-cause access before the
symptom/anchor evidence, or before intermediate dependency evidence; dashboard
miracle/reverse rates are shares of the currently filtered latent traces with
defined Pattern semantics, matching the trace-list pattern markers. `latent`
means a formerly standard case with a clean, ordered Path denominator: selected
anchors exist, root causes do not overlap those anchors, and reward Path edges
with distinct distances exist. `exposed` is the complementary formerly standard
case where that clean Pattern denominator is collapsed or unavailable. If one read step observes the
symptom, intermediate nodes, and root cause together, that simultaneous
observation is not a miracle. Unlicensed arrival marks the first touch of a
rewardable Graph node whose file path, file basename, or callable name had no
antecedent in the issue text or any strictly earlier tool observation; the
per-trace marker, the trace-list filter tag, and the `Unlicensed
arrival`/`Unlicensed trace` KPI columns all read the
`p2a/eval_fault_localization.py` license fields (`license_evaluable`,
`n_graph_arrivals`, `n_unlicensed_arrivals`, `unlicensed_arrival_nodes`).
Empirically most arrivals are licensed, so these KPIs act as a rarely-firing
provenance check rather than a reward-shaping signal. Step colors split into equal role segments when a
single step hits multiple map roles; a callable that is both symptom and root
cause uses a diagonal split so it is visually distinct from a multi-node step
hit. Node Source uses the full captured callable source when the bonus map
provides it. The
global case filters keep Overview as the full dataset registry while restricting
Metrics and Traces to the checked `direct`, `latent`, `exposed`, or `other` case
types.

The offline `summary-out` and `details-out` files are post-hoc artifacts for
inspecting dumped rollouts. Training and validation do not read them; live
validation scoring uses `P2A_EVAL_BONUS_MAP_DIR`, and the HTML dashboard reads
`P2A_EVAL_DETAILS_DIR` when per-case validation details are enabled.

### Step 9. Optional third-party model rollout baseline

For a main-style smoke/default run, set the API key and run the wrapper. It
defaults to `swebench-hard`, builds the parquet if missing, precomputes matching
Graph/Path bonus maps, and writes rollout + fault-localization artifacts
under `data/third_party/<dataset>/<model>/`:

```bash
export P2A_THIRD_PARTY_API_KEY=...
export P2A_THIRD_PARTY_BASE_URL=https://apic1.ohmycdn.com/v1
export P2A_THIRD_PARTY_MODEL=deepseek-v4-flash

bash scripts/main_3rd.sh
```

To run a pattern-focused subset without creating subtype-specific dataset
parquets, keep the source dataset and filter through bonus-map metadata:

```bash
P2A_THIRD_PARTY_CASE_TYPES=latent bash scripts/main_3rd.sh
```

Batch configs support the same scope as `bonus_map_instance_filter.case_type:
latent`. Training-time validation uses `P2A_EVAL_CASE_TYPES=latent` with
`P2A_EVAL_BONUS_MAP_DIR`; the launcher writes a derived eval parquet plus a
`.scope.json` metadata file while leaving the canonical dataset parquet intact.

For API batches, put a non-secret config under `config/` or a private one under
`.secrets/`, then run the same entry point in batch mode:

```bash
bash scripts/main_3rd.sh --batch config/third_party_batch.example.yaml
bash scripts/main_3rd.sh --batch .secrets/internal_api_batch.yaml
```

Batch configs explicitly choose `provider.source` (`openai_compatible` or
`internal_api`) and `models[]`. The committed example uses dummy model names
only. The internal adapter is tracked in `p2a/`; the private internal API client,
tokens, and model lists stay ignored under `.secrets/internal_api_eval.py` (or
the path set by `provider.api_module` / `P2A_INTERNAL_API_MODULE`). If that
private module is missing, batch mode fails before launching cells.
Batch results are upserted into the unified SQLite cache configured by
`storage.db` (default `data/evals/traces.sqlite`, resolved under
`P2A_ARTIFACTS_DIR`, which defaults to this checkout's `data/`) and can be
watched live in the unified HTML dashboard:

```bash
uv run python scripts/p2a_dashboard.py \
  --db data/evals/traces.sqlite \
  --experiment-id public-swebench-hard-demo \
  --dataset swebench-hard \
  --bonus-map-dir data/bonus_maps/swebench-hard
```

The old terminal/TUI batch watcher has been folded into this HTML dashboard.
The Metrics tab keeps per-model progress and efficiency/cost/cache visibility,
but the primary diagnostic metrics are the shared Python scorer outputs:
graph P./R./F1, path P./R./F1,
anchor/root hit rates, order/reverse-order,
miracle, purpose-block achieved/wasted/loop rates, and Path pattern flags.

If the smoke phase records only system errors such as ARL gateway or interactive
shell failures, batch mode stops before the full phase and reports the structured
error kind in the rollout artifacts.

Switch datasets with `THIRD_PARTY_DATASET=swebench-verified`,
`THIRD_PARTY_DATASET=swebench-pro`, or `THIRD_PARTY_DATASET=r2e-gym-subset`.
For `swebench-pro`, set `P2A_SWEBENCH_PRO_SCRIPTS_DIR` before the setup phase so
the parquet embeds the per-instance verifier scripts. Keep
`P2A_THIRD_PARTY_LIMIT` small for smoke tests; set it higher, or to `all`, for a
real baseline. The wrapper does not sync dependencies by default so it does not
prune a shared training `.venv`; set `P2A_THIRD_PARTY_SYNC_DEPS=1` only when you
intentionally want a core CPU sync in the active environment.

The lower-level pass-through remains available when you want explicit control
over every path and CLI flag:

```bash
export P2A_THIRD_PARTY_API_KEY=...
export P2A_THIRD_PARTY_BASE_URL=https://apic1.ohmycdn.com/v1
export P2A_THIRD_PARTY_MODEL=deepseek-v4-flash

timeout 15m bash scripts/third_party_eval.sh \
  --config config/third_party_eval.deepseek.example.yaml \
  --data $DATA/swe_bench_verified_hard.parquet \
  --out data/third_party/deepseek_v4_flash_rollouts.jsonl \
  --limit 1 \
  --max-turns 3 \
  --max-tokens 1024 \
  --tool-install-timeout 300 \
  --skip-tool-install str_replace_editor \
  --bonus-map-dir data/bonus_maps/swebench-hard
```

The harness uses Uni-Agent's `OpenAICompatibleChatModel`, the local ARL
deployment adapter, and the same SWE/R2E/SWE-Bench-Pro reward specs as training.
It writes `p2a_third_party_rollout_v1` JSONL with `messages`, structured tool calls,
`p2a_step_traces`, reward details, and termination status. When
`--bonus-map-dir` is set it also writes scorer details, a summary JSON, and a
short Markdown localization baseline report. For smoke tests, `--max-turns`,
`--max-tokens`, `--tool-install-timeout`, `--skip-tool-install`, the timeout
overrides, and an outer shell `timeout` can bound the ARL/model spend without
editing the checked-in config.

## What you configure yourself

These are knobs you set; the repo does not pin them:

| What | Where |
|---|---|
| Model | `MODEL_PATH` env var; default is `../../models/Qwen3-Coder-30B-A3B-Instruct` from `Qwen/Qwen3-Coder-30B-A3B-Instruct` |
| Shared dataset/parquet root | `DATA`, conventionally `../../datasets/p2a` |
| Project artifact root | `P2A_ARTIFACTS_DIR`, default `data/`; compatibility alias `P2A_PROJECT_DATA_DIR` |
| Train / val data | `TRAIN_FILE` / `TEST_FILE` env vars (point at the parquets built above) |
| Ray job target | `RAY_API_SERVER_ADDRESS` (Ray Jobs endpoint, usually `http://<ray-head-ip>:8265`) |
| GPU / CPU layout | `NNODES_TRAIN` / `NNODES_ROLLOUT` / `NGPUS_PER_NODE` (32-GPU 4×8 starter: `2 / 2 / 8`); `NUM_CPUS` is the Ray CPU resource advertised per node, default 64 |
| Bonus maps (read + write) | `P2A_BONUS_MAP_DIR`; setup/eval/third-party helpers default to `data/bonus_maps/<dataset>`. Training treats it as the P2A on/off switch (unset = baseline). `P2A_M_MAX` sets strength. `P2A_CREDIT_GRANULARITY=step|block` selects per-step or purpose-block credit. |
| Eval fault-localization diagnostics | `P2A_EVAL_BONUS_MAP_DIR`, `P2A_EVAL_NEAR_THRESHOLD`, `P2A_EVAL_DETAILS_DIR`, `P2A_EVAL_BONUS_N_PARALLEL`, `P2A_EVAL_BONUS_LIMIT`, `P2A_EVAL_BONUS_OFFSET`; defaults belong under `data/` |
| Third-party provider baseline | `config/third_party_eval.deepseek.example.yaml` plus `P2A_THIRD_PARTY_BASE_URL`, `P2A_THIRD_PARTY_API_KEY`, `P2A_THIRD_PARTY_MODEL`; API keys stay in env vars, not git |
| Third-party run scope | `THIRD_PARTY_DATASET`, `THIRD_PARTY_DATA_FILE`, `P2A_THIRD_PARTY_LIMIT`, `P2A_THIRD_PARTY_N_PARALLEL`, `P2A_THIRD_PARTY_RUN_TIMEOUT`, `P2A_THIRD_PARTY_BONUS_*` |
| ARL gateway | `ARL_GATEWAY_URL` |
| Hard-subset criterion | `--difficulties` flag of `build_data.py swebench-hard` (default = the `1-4 hours` / `>4 hours` difficulty set) |
| SWE-Bench-Pro scripts | `P2A_SWEBENCH_PRO_SCRIPTS_DIR` / `SWEBENCH_PRO_SCRIPTS_DIR`, pointing at the official `SWE-bench_Pro-os/run_scripts` checkout |
| R2E bad-case policy | `config/bad_instances.json` (pair-diag ARL gate evidence) |

Hydra training overrides live in `scripts/train_p2a.sh`; if you move any to a json/yaml
config, put it under `config/`.

## Eval Fault-Localization Metrics

`scripts/precompute_eval_bonus_maps.sh` reuses the same dynamic precompute path as
training bonus maps, but points it at `TEST_FILE` / `EVAL_FILE`. The resulting
maps live under the same artifact family (`data/bonus_maps/<dataset>`), but use
a split-specific directory. During training, `P2A_BONUS_MAP_DIR` should point at
the training split and `P2A_EVAL_BONUS_MAP_DIR` at the validation split.

`p2a.eval_fault_localization` accepts rollout dumps in `.jsonl`, `.json`, or
`.parquet` format.  It first reads `p2a_step_traces`, then structured
`tool_calls`, then response text / assistant messages, and reports:

| Metric | Meaning |
|---|---|
| `bonus_map_coverage` | Fraction of rollout rows with a matching eval bonus map. |
| `call_graph_coverage` | Fraction with a bonus map that contains Graph nodes; the metric name is a legacy storage/API key. |
| `read_rate` | Fraction of rows where file-viewing actions were recovered. |
| `graph_hit_rate_over_call_graphs` | Fraction whose reads hit any node in the eval Graph; the suffix is legacy naming. |
| `ground_truth_hit_rate_over_call_graphs` | Fraction whose reads hit a patched callable (`distance == 0`). |
| `near_hit_rate_over_call_graphs` | Fraction whose best read distance is `<= --near-threshold` (default `0.5`). |
| `avg_read_precision` / `avg_node_recall` / `avg_hit_f1` | Graph P., Graph R., and Graph F1 across scored rollouts. |
| `avg_path_node_precision` / `avg_path_node_recall` / `avg_path_node_f1` | Dashboard Path P., Path R., and Path F1 over deduplicated Path/context node hits; legacy `avg_chain_*` aliases are kept for old artifacts. |
| `path_node_recall` / `path_read_precision` | CLI summary aliases for Path node recall and read-level Path hit share; legacy `chain_*` aliases are kept for old artifacts. |
| `avg_order_score` / `reverse_order_rate` | Kendall-style agreement between read order and movement from tests toward patched callables. |
| `miracle_rate_over_gt_hits` | Fraction of Pattern-evaluable latent ground-truth hits that jump directly to patched code before reading intermediate graph levels. |
| `avg_block_order_score` / `block_miracle_rate_over_gt_hits` | Same order and miracle diagnostics after purpose-block segmentation. |
| `block_achieve_rate` / `block_waste_rate` / `block_loop_rate` | Purpose-block outcomes, including repeated same-action loop blocks. |
| `avg_block_efficiency_steps` | Average steps to first Graph hit inside achieving read blocks. |
| `achieving_block_step_share` / `wasted_block_step_share` / `loop_block_step_share` | Share of block-covered steps spent in each block outcome. |
| `bad_pattern_trace_rate` / `error_spiral_rate` | Trace-level loop and repeated-error flags. Step-level execution errors are parsed from tool results, command status, and traceback/error text. |
| `avg_min_distance_on_hits` | Lower is better; `0` means the model read the edited callable. |
| `avg_best_positive_multiplier_on_hits` | The diagnostic P2A multiplier implied by the best read distance. |

For live training analysis, set `P2A_EVAL_BONUS_MAP_DIR` and
`P2A_EVAL_DETAILS_DIR` when launching `scripts/train_p2a.sh`. The local
`P2AFullyAsyncRollouter` keeps the validation path otherwise unchanged, scores
validation rollouts against those eval maps, writes per-case details when
`P2A_EVAL_DETAILS_DIR` is set, and returns the same aggregate signals to the
trainer logger at each validation step.

Run the unified HTML dashboard against the details directory while training is
running:

```bash
uv run python scripts/p2a_dashboard.py \
  --details data/eval_details \
  --bonus-map-dir data/bonus_maps/swebench-hard \
  --port 8766
```

For the hard split built by this repo, the W&B/console keys are:

```
val-p2a/swebench-hard/bonus_map_coverage
val-p2a/swebench-hard/call_graph_coverage
val-p2a/swebench-hard/read_rate
val-p2a/swebench-hard/graph_hit_rate_over_call_graphs
val-p2a/swebench-hard/ground_truth_hit_rate_over_call_graphs
val-p2a/swebench-hard/near_hit_rate_over_call_graphs
val-p2a/swebench-hard/avg_node_recall
val-p2a/swebench-hard/avg_read_precision
val-p2a/swebench-hard/avg_hit_f1
val-p2a/swebench-hard/order_defined_rate
val-p2a/swebench-hard/reverse_order_rate
val-p2a/swebench-hard/miracle_rate_over_gt_hits
val-p2a/swebench-hard/block_order_defined_rate
val-p2a/swebench-hard/block_reverse_order_rate
val-p2a/swebench-hard/block_miracle_rate_over_gt_hits
val-p2a/swebench-hard/avg_blocks_per_trace
val-p2a/swebench-hard/block_achieve_rate
val-p2a/swebench-hard/block_waste_rate
val-p2a/swebench-hard/block_loop_rate
val-p2a/swebench-hard/achieving_block_step_share
val-p2a/swebench-hard/wasted_block_step_share
val-p2a/swebench-hard/loop_block_step_share
val-p2a/swebench-hard/bad_pattern_trace_rate
val-p2a/swebench-hard/error_spiral_rate
val-p2a/swebench-hard/avg_min_distance_on_hits
val-p2a/swebench-hard/avg_best_positive_multiplier_on_hits
val-p2a/swebench-hard/avg_order_score
val-p2a/swebench-hard/avg_block_order_score
val-p2a/swebench-hard/avg_block_efficiency_steps
```

`P2A_EVAL_DETAILS_DIR` writes per-case JSONL files named by validation step for
debugging individual instances and for the unified HTML dashboard. The logger
metrics above are still returned directly from validation and do not depend on
`summary-out` / `details-out` from the offline CLI.

Historical SWE-bench Verified eval-map sanity check, before the `standard` split
into `latent` and `exposed`, after the targeted F2P,
trace-capture, unittest-description F2P, zero-test runner, and F2P collection
guards, is:

| Split | Rows | Dynamic (`legacy standard+direct`) | legacy `standard` | `direct` | `newly_created` | `no_callable` | `no_f2p` | `instrumentation_failed` | `signature_mismatch` | `all_pass` | `no_trace` | `no_gt` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| hard validation | 45 | 39 (86.7%) | 32 | 7 | 4 | 0 | 1 | 1 | 0 | 0 | 0 | 0 |
| test rest | 455 | 389 (85.5%) | 258 | 131 | 35 | 18 | 2 | 2 | 7 | 2 | 0 | 0 |
| full Verified | 500 | 428 (85.6%) | 290 | 138 | 39 | 18 | 3 | 3 | 7 | 2 | 0 | 0 |

The full run used `cache/eval_bonus_verified500_f2p_targeted_20260608_220312/bonus_maps/`;
the 13 former `no_f2p` Django cases were rerun under
`cache/swe_no_f2p_rerun_20260609/maps/`, recovering 11 dynamic maps.
The 13 former `all_pass` cases were rerun after the issue #6/#7/#8 merges under
`cache/swe_allpass_after_merge_20260609_131500/maps/`, recovering 10 dynamic
maps. The remaining residuals are 2 deterministic `all_pass` Django cases and 1
`no_f2p` SymPy case.
`no_trace=0` is the build-quality gate.  Non-dynamic buckets now mean:

| Bucket | Count | Meaning |
|---|---:|---|
| `newly_created` | 39 | The patched callable is absent in the buggy tree, so a function-body tracer cannot observe it dynamically. |
| `no_callable` | 18 | The patch has no callable-level Python change for the bonus-map extractor. |
| `signature_mismatch` | 7 | The F2P test fails during Python argument binding before entering the patched callable body. |
| `instrumentation_failed` | 3 | Static callable extraction found candidates, but sandbox instrumentation produced no instrumented callable. |
| `no_f2p` | 3 | F2P failures remain unaligned with traces after description-to-method recovery. |
| `all_pass` | 2 | Buggy F2P tests exit 0 after checkout/test-selection verification, so the bug does not reproduce locally. |

After a buggy F2P run fails and enters trace classification, trace parsing
covers all captured tracer JSONL lines by default. Clean buggy runs are
classified as `all_pass` before trace parsing. Set `P2A_TRACE_PARSE_MAX_LINES`
only as an explicit debugging cap; bonus-map metadata records both
`trace_parse_line_cap_reached` and `trace_event_cap_reached`.  The runtime
tracer still keeps a finite `P2A_TRACE_MAX_EVENTS` guard, defaulting to
`10000`; when a trace cap is reached before missing F2P evidence can be proven,
the map is classified as `trace_cap_inconclusive` instead of a confident
`no_f2p`.

## Training Smoke Check

Before a full P2A run, do a small smoke run and confirm validation logs include
the `val-p2a/swebench-hard/*` metrics above. If `P2A_EVAL_DETAILS_DIR` is set,
check that it writes per-step JSONL files for the validation cases.
