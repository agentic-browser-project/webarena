# sr_compare — Sparse vs Dense attention on WebArena

Reproduces our **full** WebArena experiment on **Qwen3-VL-32B**: dense (full attention) vs a sweep of
sparse-attention methods, measuring pass rate + sparse-selection metrics, saving every LLM trajectory.

This is a **faithful copy of the currently running experiment** — same configs, same sites (including
the **self-hosted** WebArena map on `:13000`), same scoring. If you just run it, you reproduce our setup.

> An **optional** switch to run map on the **real `https://www.openstreetmap.org`** is documented at the
> bottom (["Optional: map on real OpenStreetMap"](#optional-map-on-real-openstreetmap)) — **off by
> default** because it makes map scores incomparable and gets rate-limited/blocked. Read that section
> before flipping it.

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

## Optional: map on real OpenStreetMap

By default `map` is the **self-hosted** WebArena OSM (`:13000`), exactly like the main experiment. You
can instead point it at the real site:
```bash
export WA_MAP_URL=https://www.openstreetmap.org   # (or uncomment it in config.env)
```
**Do not do this casually.** It changes results and is likely to get blocked. The risks and what to do
about them:

### Risk 1 — Reference answers won't match (results become incomparable)
WebArena's map answers (driving/walking times, addresses, search hits) were annotated against the
*self-hosted* OSM snapshot with its own routing engine. Real OSM uses different data + routers
(OSRM/GraphHopper) → different results, so route-time and address tasks will be scored **wrong even when
the agent does the right thing**. Map scores then are **not comparable** to the benchmark or to a
self-hosted run.
- **Mitigation**: treat map separately, re-judge by hand/semantics, or restrict to the knowledge-style
  map tasks (e.g. "which states border X", scored by `must_include`) that don't depend on OSM data.
- For a real apples-to-apples gain, **self-host Nominatim + OSRM** from an OSM extract (that *is*
  WebArena's `:13000`) and keep `WA_MAP_URL` unset — reproducible, reference-matching, no rate limits.

### Risk 2 — Connectivity: the proxy needs internet egress
The agent reaches every site **through an HTTP proxy** (`run_task.py` → Playwright `proxy.server`,
ports 18900-18907). For real OSM that proxy **must have public-internet egress**.
- Our in-house proxy only routes the *internal* WebArena network and **cannot reach the internet** — so
  map-on-OSM will **not** work through it unchanged. Use a proxy with internet access, or run on a host
  that can reach both your WebArena sites and the internet.

### Risk 3 — Anti-scraping / rate limits (expect throttling and IP bans)
At our concurrency, real OSM will block us:
- **Nominatim search** (`nominatim.openstreetmap.org`, the search-box backend): hard policy of
  **≤ 1 request/second**, a valid identifying `User-Agent`/`Referer` required, **no bulk/automated
  use** — violators are throttled then **IP-banned**. Our agent searches many times; 16-way concurrent
  browsing breaches this immediately.
- **Routing** ("Directions"): external demo routers (OSRM/GraphHopper), also rate-limited / may refuse
  automated traffic.
- **Tiles / website**: governed by the
  [tile usage policy](https://operations.osmfoundation.org/policies/tiles/); bulk/automated access is
  disallowed and bot-detected (headless Chromium can trip it).
- Expect 429/403/captcha/empty results well before 812 tasks finish — and it's abusive of a free
  community service to push hard.

**If you must run it, do so gently and legitimately:**
- Set **`RO_CAP=1`** in `wa_exp/run_batch.py` and add a per-request delay so the *aggregate* Nominatim
  rate stays **≤ 1 req/s** across ALL workers.
- Set a real, identifying **User-Agent** (with contact) on the browser.
- Prefer **self-hosting** Nominatim+OSRM (Risk 1 mitigation) over hitting the public service.

---

## What we found on these map tasks (heads-up)
On the map subset the per-site numbers are **noise-dominated**: small n, several tasks are actually
knowledge questions or unachievable (ref `N/A`), and the OSM site often returns "not found" for
*everyone* (same site state). Pass/fail hinges on whether the agent **falls back to its own knowledge**
and on **answer phrasing**, not on attention quality. Compare `llm_calls.jsonl` inputs across configs to
confirm the **page content is the same**; the differences are in the agent's behavior, not the site.
