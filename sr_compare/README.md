# sr_compare — Sparse vs Dense attention on WebArena

Reproduces our **full** WebArena experiment on **Qwen3-VL-32B**: dense (full attention) vs a sweep of
sparse-attention methods, measuring pass rate + sparse-selection metrics, saving every LLM trajectory.

This is a **faithful copy of the currently running experiment** — same configs, same sites (including
the **self-hosted** WebArena map on `:13000`), same scoring. If you just run it, you reproduce our setup.

## Configurations compared
| dir | method | key config |
|---|---|---|
| `dense` | full attention | Qwen3-VL-32B, triton backend |
| `tsa_tk128/96/64/48/32` | TreeSparseAttention | top_k sweep, page_size 64 |
| `vortex` | vortex **block** (official) | topk=30, chunk=32, ratio=0 |
| `quest_rec` | vortex **quest** (official) | topk=61, chunk=64, ratio=0.0625 |

Agent: `browser-use` fork, `use_vision=False`, temperature 0, max 30 steps, max_tokens 4096.
Staged over the 812 WebArena tasks in blocks: **100 → 200 → 400 → 600 → 812**, scoring cumulatively
after each block. 16-way concurrency, 8 HTTP proxies, self-healing re-runs of infra-failed tasks.

Sites (self-hosted WebArena, identical to the main run): shopping, shopping_admin, reddit, gitlab,
wikipedia, **map (`:13000`)**, across two hosts (`WA_HOST2`=hilbit2, `WA_HOST1`=hilbit1).

## Layout
```
sr_compare/
├── config.env                 # EDIT THIS: venvs, model paths, results dir, site hosts
├── run_staged.sh              # main driver (dense + 5 TSA + 2 vortex), all stages
├── serve/
│   ├── serve_dense_4x.sh      # dense Qwen3-VL, :18005-18008
│   ├── serve_tsa_4x.sh        # + serve_tsa_new.sh (TSA, needs TSA repo built)
│   ├── serve_vortex_v59_4x.sh # vortex (sglang v0.5.9 + vortex_torch)
│   └── serve_judge.sh         # Llama-3.3-70B fuzzy_match judge, :18000
└── wa_exp/
    ├── wa_config.py           # site config (map = self-hosted :13000; WA_MAP_URL to override)
    ├── run_batch.py           # site-aware scheduler (1 task/mutating replica; RO_CAP/ro replica)
    ├── run_task.py            # single-task agent runner (Playwright via HTTP proxy)
    ├── launch_vortex_textonly.py
    ├── score_batch.py / score_one.py   # official WebArena scoring -> SCORES.json
    ├── rescore_lenient.py     # lenient re-score
    ├── agg_sparse_metrics.py  # aggregate retained-KV % per config
    ├── stage_summary.py / per_site_stage.py / final_table.py   # report tables
    └── gen_auth.py            # generate site login cookies
```

## Prerequisites
- **4× ~80GB GPUs** (one model replica per GPU; judge uses 2 GPUs, run after the configs).
- **Model**: `Qwen/Qwen3-VL-32B-Instruct`. **Judge**: `Llama-3.3-70B-Instruct`.
- **Three Python envs** (kept separate — different sglang versions):
  1. `BENCH_PY` — sglang (Qwen3-VL support, >=0.5.x): driver, run_batch, scoring, dense, judge.
  2. `TSA_PY` — the TreeSparseAttention repo env (build its CUDA ext: `TSA_REPO/build`).
  3. `VORTEX_PY` — `vortex_torch` + **sglang v0.5.9** (see vortex_torch README; install order matters).
- **Agent**: `pip install browser-use` (matching fork) + `playwright install chromium`.
- **WebArena sites**: your running shopping/shopping_admin/reddit/gitlab/wikipedia/**map** servers (set
  `WA_HOST1`/`WA_HOST2` in config.env), site login cookies under `wa_exp/auth/` (`gen_auth.py`), and an
  HTTP proxy the agent reaches them through (`run_task.py` ports 18900-18907).

## 0. Clone & build the dependency repos
All repos live under the `agentic-browser-project` org. Pick a workspace dir `WS=/path/to/ws`.
You build **three separate venvs** (different sglang builds can't share one env).

```bash
# --- this repo (contains sr_compare) ---
git clone https://github.com/agentic-browser-project/webarena.git "$WS/webarena"
cd "$WS/webarena/sr_compare"        # everything below is run from the bundle

# --- A. bench env: sglang 0.5.9 + browser-use agent + scorer  => BENCH_PY ---
python -m venv ~/venvs/bench && source ~/venvs/bench/bin/activate
pip install "sglang[all]==0.5.9"                       # dense serve + judge + run_batch + scoring
git clone https://github.com/agentic-browser-project/browser-use.git "$WS/browser-use"
pip install -e "$WS/browser-use"                       # the agent fork (browser_use 0.13.x)
playwright install chromium
deactivate                                            # => BENCH_PY=~/venvs/bench/bin/python

# --- B. TreeSparseAttention  => TSA_PY + TSA_REPO ---
git clone https://github.com/agentic-browser-project/TreeSparseAttention.git "$WS/TreeSparseAttention"
cd "$WS/TreeSparseAttention"
python -m venv ~/venvs/tsa && source ~/venvs/tsa/bin/activate
bash setup.sh                                          # builds the CUDA ext into ./build (see its README)
deactivate                                            # => TSA_PY=~/venvs/tsa/bin/python, TSA_REPO=$WS/TreeSparseAttention

# --- C. vortex_torch: sglang v0.5.9 + vortex  => VORTEX_PY ---
git clone https://github.com/agentic-browser-project/vortex_torch.git "$WS/vortex_torch"
cd "$WS/vortex_torch"                                  # sglang v0.5.9 ships under third_party/sglang/v0.5.9
python -m venv ~/venvs/vortex59 && source ~/venvs/vortex59/bin/activate
pip install -e third_party/sglang/v0.5.9/sglang/python --no-deps
pip install sgl-kernel==0.3.21 --no-deps
pip install torch==2.9.1 torchvision torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu128
pip install gguf xgrammar==0.1.27 pybase64 nvidia-cudnn-cu12==9.16.0.29 flashinfer-python
pip install -e .                                       # install vortex_torch itself
deactivate                                            # => VORTEX_PY=~/venvs/vortex59/bin/python

# --- Models (HuggingFace) ---
#   Qwen/Qwen3-VL-32B-Instruct  -> MODEL_PATH
#   meta-llama/Llama-3.3-70B-Instruct -> JUDGE_MODEL
```
> The WebArena **sites** (shopping/admin/reddit/gitlab/wikipedia/map) are a separate deployment —
> stand them up per the upstream WebArena docker instructions and point `WA_HOST1/2` at them.

## Setup
```bash
cd "$WS/webarena/sr_compare"
$EDITOR config.env          # point BENCH_PY/TSA_PY/VORTEX_PY, MODEL_PATH, JUDGE_MODEL,
                            # TSA_REPO, WA_HOST1/2 at your boxes
source config.env
# one-time: generate site auth cookies for the mutating sites
$BENCH_PY wa_exp/gen_auth.py     # writes wa_exp/auth/*.json
```

## Run everything
```bash
cd /home/cc/temp/webarena/sr_compare
nohup bash run_staged.sh &       # logs -> sr_run.log
tail -f sr_run.log
```
Per task it writes `results/<cfg>/task_<id>/task_<id>.json` (final answer, steps, final_url, n_steps),
`llm_calls.jsonl` (full per-step input + output + token usage), `input.json` (resolved task + eval),
plus `SCORES.json` per config and runtime sparse-metrics jsonl. After each block the driver prints a
cumulative pass-rate table (`stage_summary.py`).

### Run a single config / report manually
```bash
source config.env
# serve one config (example: vortex-quest), wait for :18005-18008, then:
METRICS_DIR=$WA_RESULTS/quest_rec VORTEX_MODULE=gqa_quest_sparse_attention \
  VORTEX_TOPK=61 VORTEX_CHUNK=64 VORTEX_RATIO=0.0625 bash serve/serve_vortex_v59_4x.sh &
$BENCH_PY wa_exp/run_batch.py --out-dir $WA_RESULTS/quest_rec --base-urls "$LLM_URLS" \
  --model qwen3vl-vortex --start 0 --end 812 --max-concurrency 16 --num-proxies 8 \
  --retries 2 --max-steps 30 --task-timeout 3000 --skip-done
# score (judge must be up on :18000):
$BENCH_PY wa_exp/score_batch.py --results-dir $WA_RESULTS/quest_rec --judge-url $JUDGE_URL --judge-model llama-judge
# tables:
$BENCH_PY wa_exp/stage_summary.py 812      # cumulative pass-rate per config
$BENCH_PY wa_exp/per_site_stage.py 812     # per-site breakdown
$BENCH_PY wa_exp/final_table.py            # unified: retained-KV% | chunks | success
```

---

## Task ID record (results)

Three experiments were run with this bundle. For every method, the **passed / failed /
execution-errored task-id lists** are stored as JSON under `task_outcomes/`:

| table | experiment | per-method task-id JSONs |
|---|---|---|
| 1 | self-hosted WebArena, all sites, tasks 0–399 | `task_outcomes/self_hosted_map/<method>.json` |
| 2 | block1: `map` switched to real `openstreetmap.org`, tasks id<100 (51 map tasks) | `task_outcomes/map_osm_block1/<method>.json` |
| 3 | block2: `map` on real `openstreetmap.org`, tasks id≥100 (58 map tasks) | `task_outcomes/map_osm_block2/<method>.json` |

Each `<method>.json` contains:
`lenient` = {`n_pass`, `passed_ids`, `failed_ids`}, `official` = {`n_pass`, `passed_ids`, `failed_ids`},
and `execution_errored_ids`. (`passed_ids + failed_ids` = every task that actually ran.)
See `task_outcomes/README.md` for the exact file format.

### Terms
- **official** — strict WebArena scoring: `string_match` / `url_match` / `program_html`, with
  `fuzzy_match` graded by the Llama-3.3-70B judge. A task passes iff `score >= 1`.
- **lenient** — passes official **OR** the answer states the correct reference value but only failed
  strict scoring on phrasing/formatting (re-judged by the LLM judge, `rescore_lenient.py`). This is the
  **primary success rate** used below (real-OSM tasks especially: strict refs were annotated on the
  self-hosted snapshot, so `official` is low by construction).
- **execution error** — task that did not finish cleanly (connection timeout / crash); a subset of the
  failed ids, recorded separately in `execution_errored_ids` (so you can tell "ran but wrong" from
  "didn't finish").
- **retained_KV** — measured fraction of the KV cache the sparse method actually attends to
  (dense = 100%; lower = more aggressive sparsity).
- **chunk_kept** — fraction of chunks/blocks kept by the sparse selector.

### Why input/output length, retained_KV and chunk_kept are mostly blank
- **Tables 1 & 2** were produced on **other machines**. Only the pass/fail task-id artifacts were copied
  into this repo (plus block1's measured `retained_KV`). The raw trajectories (`llm_calls.jsonl`) and
  the runtime sparse-metrics files stayed **local to those machines**, so input/output token lengths and
  chunk% are not recorded here.
- **Table 3** (this machine): the serving endpoints returned `prompt_tokens = 0` (input length not
  reported) → **in_len blank**; and the TSA server build used here writes no sparse-metrics file →
  **retained_KV / chunk_kept not recorded**. `out_len` (completion tokens, avg ± 3σ) is well-sampled
  only for **dense** (779 logged calls); the trajectory logger silently drops calls whose response
  format differs, so the TSA `out_len` `[n]` is small and biased.

### Table 1 — self-hosted WebArena (all sites), first 400 tasks
Cells = **lenient** pass / ran (rate). `wikipedia` has no single-site tasks in 0–399 (those are
cross-site, counted under `cross/other`).

| method | shopping | shopping_admin | reddit | gitlab | wikipedia | map | cross/other | **TOTAL** | exec_err | in_len | out_len | retained_KV | chunk_kept |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| dense | 49/124 (40%) | 24/89 (27%) | 4/10 (40%) | 20/72 (28%) | — | 7/100 (7%) | 0/5 (0%) | **104/400 (26%)** | 0 in 0–399 (**15 in full 812**) |  |  |  |  |
| tsa_tk128 | 44/124 (35%) | 15/89 (17%) | 4/10 (40%) | 25/72 (35%) | — | 11/100 (11%) | 0/5 (0%) | **99/400 (25%)** | 0 |  |  |  |  |
| tsa_tk96 | 39/124 (31%) | 20/89 (22%) | 3/10 (30%) | 17/72 (24%) | — | 9/100 (9%) | 0/5 (0%) | **88/400 (22%)** | 0 |  |  |  |  |
| tsa_tk64 | 34/124 (27%) | 15/89 (17%) | 4/10 (40%) | 18/72 (25%) | — | 10/100 (10%) | 1/5 (20%) | **82/400 (20%)** | 0 |  |  |  |  |
| tsa_tk48 | 29/124 (23%) | 12/89 (13%) | 2/10 (20%) | 15/72 (21%) | — | 10/100 (10%) | 0/5 (0%) | **68/400 (17%)** | 0 |  |  |  |  |
| tsa_tk32 | 34/124 (27%) | 12/89 (13%) | 4/10 (40%) | 11/72 (15%) | — | 8/100 (8%) | 1/5 (20%) | **70/400 (18%)** | 0 |  |  |  |  |
| vortex_block | 37/124 (30%) | 14/89 (16%) | 2/10 (20%) | 18/72 (25%) | — | 10/100 (10%) | 1/5 (20%) | **82/400 (20%)** | 0 |  |  |  |  |
| vortex_quest | 34/124 (27%) | 15/89 (17%) | 5/10 (50%) | 22/72 (31%) | — | 10/100 (10%) | 0/5 (0%) | **86/400 (22%)** | 0 |  |  |  |  |

> ⚠️ **Execution errors — read before comparing.** Within the **first 400 tasks every method has 0
> execution errors**, so the Table-1 comparison is clean. **dense's full 812-task run, however, has 15
> execution errors** — task ids **436–440, 506–510, 585–589** — all in the 400–811 range (outside this
> table). They are **`ConnectTimeout`s** (the agent's headless browser couldn't reach the WebArena site
> during a brief replica/proxy outage), **not** a model/attention failure, and they were **never
> retried** (dense was a pre-existing reference run outside the self-healing loop). They have
> `score = null` → counted as **fails**, so they slightly lower dense's full-812 number (200/812 = 24.6%;
> excluding them, 200/797 = 25.1%). They do **not** affect the 0–399 comparison above. The sparse
> configs each ran 0–399 with **0** execution errors.

Task-id lists: `task_outcomes/self_hosted_map/<method>.json` (`lenient`/`official` `passed_ids`/`failed_ids`,
and **`execution_errored_ids`** — dense's is `[436–440, 506–510, 585–589]`). Methods: dense,
tsa_tk128/96/64/48/32, vortex_block, vortex_quest.

### Table 2 — block1: map on real openstreetmap.org, tasks id<100 (51 map tasks)
map is the only site, so per-site = total. `retained_KV` here is the value **measured on the machine
that ran block1** (recorded in `task_outcomes/README.md`).

| method | map (= total) lenient | official | in_len | out_len | retained_KV | chunk_kept |
|---|---|---|---|---|---|---|
| dense | 10/51 (20%) | 4/51 (8%) |  |  | 100% |  |
| tsa_tk128 | 14/51 (27%) | 6/51 (12%) |  |  | 92.6% |  |
| tsa_tk64 | 14/51 (27%) | 7/51 (14%) |  |  | 72.7% |  |
| tsa_tk32 | 11/51 (22%) | 6/51 (12%) |  |  | 33.9% |  |
| vortex_block | 11/51 (22%) | 7/51 (14%) |  |  | 6.3% |  |
| vortex_quest | 16/51 (31%) | 9/51 (18%) |  |  | 12.0% |  |

Task-id lists: `task_outcomes/map_osm_block1/<method>.json`.

**Reproduce Table 2:** `bash run_block1.sh` (task ids in `osm_block1_ids.txt` = the 51 single-site
`map` tasks with id<100). It forces `WA_MAP_URL=https://www.openstreetmap.org`, serves each of the 6
configs in turn, runs the 51 tasks, then brings up the judge and writes per-method task-id JSONs +
`summary.md` under `task_outcomes/map_osm_block1_repro/`. Prereqs: the 3 venvs + models + a proxy with
**public-internet egress** (real OSM) per the README setup. vortex needs a CUDA-12.8 GPU driver
(>= ~570); on older drivers the script logs + skips the two vortex configs and still finishes dense+TSA.

### Table 3 — block2: map on real openstreetmap.org, tasks id≥100 (58 map tasks, this machine)
map is the only site. `out_len` = completion tokens, avg ± 3σ, `[n logged calls]`.
`vortex_block`/`vortex_quest` are pending: vortex's cu128/NCCL stack needs a CUDA-12.8 GPU driver
(this box was upgraded 560→570; the values land after a reboot). See
`task_outcomes/map_osm_block2/summary.md` for the latest.

| method | map (= total) lenient | official | in_len | out_len (avg±3σ) [n] | retained_KV | chunk_kept |
|---|---|---|---|---|---|---|
| dense | 21/58 (36%) | 17/58 (29%) |  | 411 ± 964 [779] |  |  |
| tsa_tk128 | 6/58 (10%) | 4/58 (7%) |  | 442 ± 216 [35] |  |  |
| tsa_tk64 | 4/58 (7%) | 3/58 (5%) |  | 437 ± 264 [35] |  |  |
| tsa_tk32 | 6/58 (10%) | 4/58 (7%) |  | 355 ± 248 [53] |  |  |
| vortex_block | — (pending) | — |  |  |  |  |
| vortex_quest | — (pending) | — |  |  |  |  |

Task-id lists: `task_outcomes/map_osm_block2/<method>.json`.
