# CLAUDE.md — P2A on Uni-Agent + ARL (code-level)

Controller requirements gathered during the 2026-06 migration. Read before
touching this tree. The research-level `CLAUDE.md` is at the repo root.

## Hard rules (controller stated these repeatedly)

1. **Self-contained — never depend on `src-backup`** (code or data). All data is
   sourced from HuggingFace:
   - R2E training: `R2E-Gym/R2E-Gym-Subset` (instances + parsed_commit_content +
     relevant_files). Cases that genuinely fail F2P/P2P on the pair-diag ARL images
     are recorded in `config/bad_instances.json` and excluded from training; the
     `dyyyyyyyy/r2e-gym-subset-filtered` set is not used (its filtering was built for
     a different enterprise registry).
   - SWE-bench eval: `R2E-Gym/SWE-Bench-Verified` (rows + eval fields), with the
     `difficulty` label cross-indexed from `princeton-nlp/SWE-bench_Verified`.
   Old code may be **read for reference only**, never imported/run.
2. **No throwaway / one-off scripts as the pipeline, no post-hoc enrichment.** 不要写冗余代码.
   In particular, **ALL "build data" jobs are subcommands of the single
   `scripts/build_data.py`** (`r2e` / `swebench-verified` / `swebench-hard` / `skip-list`). Do NOT add a
   separate `build_*.py` per dataset — building the hard subset, the skip-list, etc.
   are all "build data", so they go in `build_data.py`. 举一反三.
   **Likewise, EVERY runnable entry/launcher script (`*.sh`, training/data launchers)
   lives in `scripts/`, never in `p2a/`/`env/`** (module code stays there; `p2a/main.py`
   is a `python -m p2a.main` module entry, invoked by `scripts/train_p2a.sh`).
   `train_p2a.sh` was moved `p2a/`→`scripts/` on 2026-06-08.
3. **All json/yaml config goes under `config/`** (`startup_fixups.json`,
   `bad_instances.json`, and any future training config).
4. **Images = pair-diag mirror**, not enterprise. `env/images.py` routes to
   `pair-diag-cn-guangzhou.cr.volces.com/code/...`; the build wrapper writes the
   pair-diag ref into each row. pair-diag is the mirror of the original R2E
   `namanjain12` images and is the reference the old bonus-map report reproduces on.
5. **`uni-agent/` is an UNMODIFIED submodule.** All our glue lives in `p2a/`,
   `env/`, `scripts/`, `config/`. Reuse Uni-Agent's prompt/schema constants by
   import, do not copy. `swe-rex` is Uni-Agent's own runtime interface — required,
   not removable (we implement its `AbstractRuntime` for ARL; we do not run a
   swe-rex server).
6. **Training runtime = uv-managed `.venv` on native CUDA 13.0.** Launchers default
   to `/usr/local/cuda-13.0` and the locked cu130 stack; do not add a parallel
   pip-managed runtime path.
7. **Precompute and rollout MUST share execution semantics (INVIOLABLE — see root
   `FACTS.md` FACT-001).** The bonus map / golden call graph is built by the
   precompute path; its supervision is applied to the rollout path. Both MUST use
   the *same* command-execution interface and the *same* sandbox env (PATH / venv /
   cwd / shell init), or the supervision no longer matches what the agent ran.
   Current ARL rollout execution uses `ArlRuntime.run_in_session` over
   `ManagedSession.execute`, seeds the configured repo cwd (`session_cwd`, `/testbed`
   for R2E and `/app` for SWE-Bench-Pro), preserves `cd`/exported env through
   runtime state files, and strips terminal control sequences from observations.
   `InteractiveShellClient` is a human/debug readline PTY and must not be used for
   model/tool command execution. Any change to one path's exec interface or env
   MUST be mirrored to the other; do not add a precompute-only or rollout-only
   execution nuance.

## Asset and artifact paths

- `../../datasets` and `../../models` are the shared roots for reusable datasets
  and model checkpoints. Generated P2A parquets use `DATA`, conventionally
  `../../datasets/p2a`; override the model root with `MODEL_PATH`, `MODEL`, or
  `P2A_MODELS_DIR`.
- `src/data` is the default artifact root for TraceAnalyzer-generated outputs:
  bonus maps, validation details, SQLite eval caches, rollout dumps, analysis
  reports, and dashboard snapshots. Override the root with `P2A_ARTIFACTS_DIR`
  only when needed.
- Keep public datasets and reusable checkpoints out of `src/data`; keep
  project-specific artifacts out of the shared datasets/models roots by default.

## Fixups & skip-list

- `config/startup_fixups.json` = the faithful, behavior-equivalent port of the old
  `test_startup_fixups` (source patches like aiohttp `asyncio.async(`→`create_task(`,
  numpy `is ()`→`== ()`, coveragepy py3.10 opcode, orange3 numpy pin, + dep installs).
  **Do not trim** — removing "redundant-looking" fixups already broke classification
  once. Add new per-repo fixups here; any removal needs a full-gate ablation.
- `config/bad_instances.json` = instances whose F2P/P2P cannot be reproduced even
  with fixups. Excluded from ALL ARL training (RL data, P2A bonus precompute, P2A
  training). Unfixable cases: mark here first; later rebuild correct images and push
  to pair-diag to recover them.

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

## Unit Test Policy

Do not write or modify unit tests unless they are strictly necessary and the user
has explicitly approved that test work first.

## Dashboard service deployment

- This environment is a server. Do not present dashboard preview URLs bound to
  `127.0.0.1` or `localhost` as user-accessible links.
- For user-accessible dashboard services, bind the server to `0.0.0.0` and give
  the user the server public IP plus port, for example
  `http://<server-public-ip>:8770`.
- When the user says they will manage the service themselves, provide the exact
  start/stop commands only; do not start or keep the service running for them.

## P2A advantage — verify before trusting (TODO)

`p2a/trainer.py::apply_p2a_reshape` + `p2a/core.py` implement and wire the reshape
(capture agent reads → match to call graph → `m(d)=m_max^(1-d)` multiplier). It has
**never run end-to-end at training**. Before a real run, do a small demo smoke test
proving P2A works on the Uni-Agent tool set and actually captures actions
(non-empty `reads`). See README "TODO".

## Reminders

- **ARL is the sandbox backend, not a VRC remote.** The `arl-env` SDK connects
  directly to the ARL Gateway (`ARL_GATEWAY_URL`)
  to boot a per-instance container sandbox where tests + P2A instrumentation run
  (bonus-map precompute, training rollouts); it is reachable directly from CPU hosts.
  This is unrelated to VRC's `remote` facility — `vrc remote` targets the **GPU
  server** for command debugging. ARL gateway reachability is independent of
  `vrc remote health`; do not infer one from the other.
- Every Python invocation inside `src/` uses `uv run` (pinned `uv.lock`). Keep
  local P2A imports resolvable with `PYTHONPATH=uni-agent/verl:uni-agent:.` when
  running from this `src/` directory.
- **Comments describe the present design, not the code's history.** Do NOT write
  changelog/defensive comments ("previously did X, it was a bug, changed to Y",
  "reverted the … switch", dated attributions, PR/issue numbers as narrative). Git
  log is the history; the code is not your history book. Keep present-tense WHY
  comments; delete or rewrite the rest.
- Migration validated on a 26-case stratified sample: dynamic signal (standard/direct)
  reproduces the old report 8/8; full case_type ~24/26 (residual = non-dynamic edges).
