# AGENTS.md - P2A source repository

This `src/` directory is the source repository for the Uni-Agent migration. It has
a GitHub remote (`origin` = `git@github.com:Siyuexi/TraceAnalyzer.git`); development
is on `main`.

## Layout

- `uni-agent/` - pristine Uni-Agent upstream-mirror submodule. Use this for vanilla baseline code and upstream Uni-Agent docs.
- `env/` - local Uni-Agent environment glue for ARL SDK deployment, image routing, agent-loop adapter, and smoke/data helpers. The external `arl-env` SDK owns the `arl` import name.
- `p2a/` - P2A trainer wrapper, advantage reshape code, and bonus-map precompute utilities.
- `scripts/` - local launch helpers for preparing data/config and running Uni-Agent baseline checks.
- `UNI_AGENT_MIGRATION.md` - current migration notes and tomorrow's baseline commands.

## Asset and artifact paths

- Shared datasets live outside this checkout under `../../datasets` by default.
  Generated P2A parquets use `DATA`, conventionally `../../datasets/p2a`.
- Shared model checkpoints live outside this checkout under `../../models` by
  default. Use `MODEL_PATH`, `MODEL`, or `P2A_MODELS_DIR` to override.
- Project artifacts live inside this checkout under `data/` by default. This
  includes bonus maps, validation details, SQLite eval caches, rollout dumps,
  analysis reports, and dashboard snapshots. Use `P2A_ARTIFACTS_DIR` only when
  the whole artifact root must move.
- Temporary local batch/eval configs that need secrets or private endpoints
  belong under `.secrets/tmp/`. Do not place one-off experiment YAMLs directly
  under `.secrets/`; keep the top-level secret config directory for curated,
  intentionally reusable configs.
- For temporary experiments, start from the closest existing config for the same
  dataset and change only the requested experiment scope, model list, rollout
  range, or output id. Keep adapters, provider settings, concurrency, and other
  execution defaults unchanged unless the controller explicitly asks to change
  them.
- Do not put public/reusable datasets or model checkpoints under `src/data`.
  Do not put TraceAnalyzer-specific run artifacts under `../../datasets` or
  `../../models` by default.
- SWE-Bench-Pro is eval-only. Phase 1 supports only Python repos and writes
  `swe_bench_pro.parquet`; never route it into RL training. The official Pro
  images use `/app` as the repository root, not SWE-bench Verified's `/testbed`.
  ARL runs require the corresponding `jefzda/sweap-images:{dockerhub_tag}` tags
  to be mirrored under `pair-diag-cn-guangzhou.cr.volces.com/code/sweap-images`;
  Docker Hub direct pull is not assumed available from ARL.

## Research concept docs

- The root `proposal.md` is the proposal source of truth, root
  `proposal.html` is the canonical web rendering, and root
  `report/proposal.html` is the VRC-hosted copy. Keep all three synchronized
  when proposal text changes.
- The root `report/2026-06-18_traceanalyzer-logic-map.html` is the living
  implementation/concept map. New P2A concepts should be recorded there with
  stable fully-qualified symbol anchors (`module::function` or
  `module::Class.method`), not brittle file-line references.
- The root `report/2026-06-09_p2a-bonus-map-pipeline.html` is the companion
  reference for bonus-map taxonomy, capture, and classification semantics.
- If a code change introduces a concept that changes research claims, method
  semantics, experiment definitions, or public terminology, update root
  `proposal.md`, `proposal.html`, and `report/proposal.html` in the same unit.

## Semantic consistency rules

- Do not let dashboard-only terminology drift from P2A functionality. Trace
  labels, KPI names, legend text, reports, and README explanations must use the
  same semantics as `p2a/core.py`, `p2a/eval_fault_localization.py`, and
  `p2a/dashboard_adapter.py`.
- Use Graph / Path / Trace terminology consistently. **Graph** means the real
  dependency graph captured from instrumentation/failing-test execution.
  **Path** means the issue symptom-to-root-cause subgraph/path. **Trace** means
  the model/agent execution trajectory. Do not use "Trace" for the captured
  dependency graph in user-facing labels or research text.
- Treat `call_graph_*`, `chain_*`, `dynamic_traceable_*`, and historical
  instrumentation filenames as legacy storage/API vocabulary. New helpers,
  variables, comments, UI labels, README/proposal text, and issue descriptions
  should use Graph, Path, and Trace directly; when old keys are required for
  backward compatibility, isolate them behind explicit alias/normalization
  helpers and label them as legacy.
- Use Graph node roles consistently. Only `test_harness` is non-rewardable.
  `test_adapter` is the rewardable non-test frame before the selected issue
  symptom anchor; legacy `pre_symptom` is only an artifact alias for
  `test_adapter`. `fix_adapter` is a rewardable golden-patch-modified callable
  upstream of the terminal patched root cause. `root_cause` is reserved for
  terminal patched callables/components. The training ground-truth anchor is
  the first non-test node after the test harness; the issue symptom anchor is a
  diagnostic/visual Path anchor, not the reward boundary.
- If a dashboard feature depends on read/write/error/root-cause semantics, add
  or reuse the corresponding parser/scorer fields in P2A source first, then
  render those fields in the frontend. Avoid frontend-only inference for
  metrics or trace status unless it is a compatibility fallback for old
  artifacts.
- Keep dashboard persistence split into a raw eval DB and a dashboard build DB.
  The raw eval DB is the rollout-time store: it should contain raw rollout
  content, issue descriptions, golden patches, token/runtime data, run status,
  and artifact references. It should not carry dashboard pattern/detail payloads
  or pattern-derived metrics, and it should store each raw trace only once in
  structured columns (`messages_json`, `trajectory_json`, and
  `p2a_step_traces_json`). `rollout_json` is a slim metadata compatibility field,
  not a second full trace copy. The dashboard build DB is the dashboard-bound
  materialized cache: it contains per-rollout pattern/detail rows and eval-cell
  metrics rows, but no raw trace payloads. Data flows one way from raw DB to
  build DB only during admin rebuild, explicit build-DB migration, or
  incremental rollout caching; dashboard read paths treat the build DB as
  read-only. Starting or restarting the dashboard server must not migrate,
  rebuild, or otherwise materialize build DB data.
- Structured rollout/system failures are rerunnable error states, not cached
  traces. Error cases should update only `run_cells.status/error` for overview
  and rerun eligibility; they must not write `raw_rollouts`,
  `quantitative_metrics`, or dashboard build-DB detail/metric rows. Dashboard
  cache readiness is counted over completed non-error rollouts only.
- For dashboard DB reads, use build DB materialized data when it is complete and
  fingerprint-valid for every completed non-error rollout in an eval cell. If
  any such rollout is missing valid build data, expose only basic progress,
  pass/resolved, token/cost, and length metrics for that cell. Symptom/root,
  Path, pattern, order, miracle, and purpose-block KPIs remain null/hidden until
  the cell is fully materialized. The Traces view should list only rollouts that
  actually have raw trace content; planned-but-unrolled cells are not useful
  trace rows, and traces without materialized detail should render without
  pattern tags.
- Treat node source code as bonus-map data. Dashboard Node Source must read full
  callable source from the inferred or explicit P2A bonus-map directory; DB
  `source_preview` fields are only compatibility fallbacks for old artifacts.
- Execution failure must be inferred from structured tool/runtime signals such
  as `status`, `error`, nonzero exit code, or traceback/command-failure output,
  not from broad keyword scans over source code or successful read observations.
- Keep the unified dashboard compatible with local training, local inference,
  and third-party API inference artifacts. New trace fields should degrade
  cleanly when older artifacts do not contain them.
- When public-facing semantics change, update the README and, where research
  claims are affected, keep the proposal/report documents synchronized.

## Precompute ↔ rollout alignment (INVIOLABLE — root `FACTS.md` FACT-001)

The bonus map / golden call graph is built by the **precompute** path; its process
supervision is applied to the **rollout** path. Both MUST observe identical
execution semantics — the *same* command-execution interface and the *same* sandbox
environment (PATH / venv / cwd / shell init) — or the supervision no longer matches
what the agent actually executed (P2A soundness breaks).

- Current ARL rollout execution uses `ArlRuntime.run_in_session` over
  `ManagedSession.execute`, seeds the configured repo cwd (`session_cwd`, `/testbed`
  for R2E and `/app` for SWE-Bench-Pro), preserves `cd`/exported env through
  runtime state files, and strips terminal control sequences from observations.
- `InteractiveShellClient` is a human/debug readline PTY and must not be used for
  model/tool command execution. ARL rollout command execution must stay on the same
  `execute` interface that precompute uses.
- Any change to one path's exec interface or environment MUST be mirrored to the
  other. Do not introduce a precompute-only or rollout-only execution nuance.

## ARL is a sandbox, not a VRC remote

ARL is the containerized compute backend. The `arl-env` SDK connects directly to the
ARL Gateway (`ARL_GATEWAY_URL`) to boot a per-instance
sandbox where tests + P2A instrumentation run (bonus-map precompute, training rollouts),
and it is reachable directly from CPU hosts. This has nothing to do with VRC's `remote`
facility: `vrc remote` is the debug proxy that targets the **GPU server** when the local
host has no GPU. An ARL gateway being reachable or not is independent of `vrc remote
health` — do not infer one from the other.

## Dashboard service deployment

- This environment is a server. Do not present dashboard preview URLs bound to
  `127.0.0.1` or `localhost` as user-accessible links.
- For user-accessible dashboard services, bind the server to `0.0.0.0` and give
  the user the server public IP plus port, for example
  `http://<server-public-ip>:8770`.
- When the user says they will manage the service themselves, provide the exact
  start/stop commands only; do not start or keep the service running for them.

## Python Rules

Every Python invocation inside `src/` MUST be prefixed with `uv run` (the repo pins
`uv.lock`). Bare `python`/`pip`/`pytest` either fail (deps not on PATH) or silently
pick a different environment than the lock — `uv run python ...`, `uv run pytest ...`,
`uv run ruff ...` are the only correct forms.

- Keep local P2A imports available with `PYTHONPATH=uni-agent/verl:uni-agent:.` when running from this `src/` directory.
- `UV_CACHE_DIR=/tmp/uv-cache` is a Codex sandbox workaround only: use it when
  Codex cannot write to `~/.cache/uv`, and run through this repo's `uv run` /
  `src/.venv` interpreter rather than another Python. It is not a project
  requirement. Commands written for the user should stay in the normal form
  they can run from `src/`, such as `bash scripts/...` or `uv run ...`, without
  Codex-only cache prefixes.
- For ARL runs, use `scripts/uni_agent_arl.sh`; it keeps `uni-agent/` unmodified and routes runtime startup through `env.agent_loop.ArlUniAgentLoop`.

## Unit Test Policy

Do not write or modify unit tests unless they are strictly necessary and the user
has explicitly approved that test work first.

## Comment Hygiene

Comments describe the present design and the non-obvious WHY — never the code's
history. Do NOT write changelog/defensive comments ("previously did X, it was wrong,
changed to Y", "reverted the … switch", dated attributions, PR/issue numbers as
narrative). Git log is the history; the code is not. Delete such comments or rewrite
them present-tense.

The same rule applies to documents (FACTS.md, GLOSSARY.md, README, reports): entries
state the current semantics plainly — no "(v2, supersedes …)", no "finalized/updated
on <date>", no "previously X, now Y" narrations. When semantics change, rewrite the
entry; git log is the only history book.

## Uni-Agent / ARL Docs

When working with the Uni-Agent training stack, use the official Uni-Agent
documentation as a primary reference. ARL-specific runtime behavior lives in
this repository under `env/` and `scripts/uni_agent_arl.sh`.
https://uni-agent.readthedocs.io/en/latest/index.html

## Git Rules

- `src/` is a git repo with a GitHub remote (`origin` = `git@github.com:Siyuexi/TraceAnalyzer.git`). Open PRs against `main`; the controller merges. Never self-merge.
- `uni-agent/` is a nested submodule pointing at the pristine fork mirror `git@github.com:Siyuexi/uni-agent.git`.
- Do not modify `uni-agent/`; put P2A behavior in `p2a/`, `env/`, `scripts/`, or `config/`.
- Treat GitHub read access, git-over-SSH push access, and authenticated
  issue/PR comment writes as separate capabilities. Do not infer a valid `gh`
  token from successful GitHub reads or `git push`; before claiming GitHub
  synchronization is blocked, identify the exact operation and access path.
