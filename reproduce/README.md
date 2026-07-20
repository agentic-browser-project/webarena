# WebArena 0-99 — Sparse Attention Results (sparsity + per-site breakdown)

## Experiment configuration

| Item | Value |
|---|---|
| **Model under test** | **Qwen3-VL-32B-Instruct** (`MODEL_PATH`); all five methods share the same model, only the attention implementation changes |
| **Scoring judge** | Llama-3.3-70B-Instruct (vLLM, `--served-model-name llama-judge`) |
| **Vision / screenshots** | ⚠️ **Disabled throughout** (see below) |
| Sampling | `temperature 0` (deterministic), `max_tokens 4096` |
| Agent | browser-use, `max_steps 30` |
| Tasks | WebArena tasks **0-99** (100 per method); the map site points at the **real openstreetmap.org** |

> ⚠️ **On vision: the model is a vision-language model, but screenshots and visual analysis were disabled for the entire experiment.**
> The agent receives **no page screenshots** — only the page's **accessibility tree as text**. Two switches are active: `use_vision=False` in `run_task.py`, and the chromium flag `--blink-settings=imagesEnabled=false` (images are not even loaded).
>
> Why: (1) vortex/quest does not handle mrope and emits garbage with images, so **quest could not participate in the comparison with vision on**; (2) with vision off, all five methods receive an identical input modality, which is what makes the sparse-attention comparison clean (and why the median `in_len` below is nearly identical across methods).
>
> **All numbers here are therefore "text-only agent" results** and do not represent the model's ceiling with vision enabled.

## Results (sparsity + lengths + per-site)

| Method | top_k | %chunks | %pages | **%selected tokens** | **median in_len** | **median out_len** | **official** | shopping | admin | gitlab | reddit | **non-map subtotal** | map | **TOTAL (lenient)** |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **dense** (full attention) | — | 100% | 100% | **100%** | 8230 | 402 | 23% | 5/12 | 7/24 | 2/3 | 5/9 | **19/48 (39%)** | 13/51 | **33/100 (33%)** |
| **quest** | 61 pages | — | 10.8% | **10.8%** | 8111 | 363 | 26% | 5/12 | 8/24 | 1/3 | 5/9 | **19/48 (39%)** | 13/51 | **33/100 (33%)** |
| **TSA-minmax tk64** | 64 chunks | 53.6% | — | **78.3%** | 8091 | 397 | 19% | 5/12 | 7/24 | 1/3 | 4/9 | **17/48 (35%)** | 16/51 | **34/100 (34%)** |
| **TSA-centroid tk64** | 64 chunks | 54.8% | — | **57.5%** | 8092 | 377 | 14% | 5/12 | 6/24 | 1/3 | 2/9 | **14/48 (29%)** | 14/51 | **28/100 (28%)** |
| **TSA-centroid tk32** | 32 chunks | 30.6% | — | **26.6%** | 8306 | 338 | 9% | 4/12 | 6/24 | 0/3 | 1/9 | **11/48 (22%)** | 15/51 | **26/100 (26%)** |

Per-site cells are `passed / tasks on that site` (lenient); TOTAL is lenient, the `official` column is the strict pass rate.

`%selected tokens` is a **full-model measurement** (fraction of KV the selected chunks actually cover, using real decode queries); for quest it is computed exactly. **This is the only directly comparable axis between quest and TSA** — `%chunks` and `%pages` are different units.

`in_len` / `out_len` are median token counts **per LLM call**, measured consistently: inputs are re-tokenized from the prompt actually sent (the TSA server does not return `prompt_tokens`, so all methods are re-tokenized), outputs use `completion_tokens`. Means are in 8373–8592 / out 391–483, over 1219–1458 calls per method. Input lengths are near-identical across methods (median 8.1k–8.3k), confirming the workloads are comparable.

## Data availability (LLM trajectories)

The **full LLM call trajectories** for all five methods (100 tasks each) are published as a
Hugging Face dataset:

**https://huggingface.co/datasets/Sean1999/webarena**

The dataset is **private** — request access from **Xian Wang (wangxian@engineering.upenn.edu)**.
The easiest way is through the Hugging Face web page itself: open the dataset link while signed in
and use the access-request button there; the request is reviewed and granted from that page. Emailing directly also works if you prefer.

It contains, per method and per task: `llm_calls.jsonl` (every LLM call with the full input
messages, output, token usage and latency), `task_<id>.json` (final answer, step count,
`final_url`, timings), `input.json` (resolved task definition and `eval` ground truth) and
`run.log` (browser-use agent log), plus `SCORES.json` / `SCORES_adjusted.json` per method.

This repository keeps only the aggregated results and the pass/fail id lists (`task_ids/`);
the raw trajectories are too large to version here.

## Layout and scripts

```
reproduce/
├── config.env        ← the only file to edit when moving machines (external paths)
├── harness/          required for reproduction: scheduler / task runner / auth generation
├── scripts/          the drivers and measurement scripts actually used in this experiment
├── scoring/          scoring (official + lenient)
├── serve_reference/  serving scripts (REFERENCE ONLY, see below)
├── build_sm90/       sm100 → sm90 port patches
└── task_ids/         per-method pass/fail task ids
```

**Self-contained / portable**: paths between `harness/` `scripts/` `scoring/` are resolved from each script's **own location** (`__file__` for Python, `$(dirname "${BASH_SOURCE[0]}")` for shell), so the whole `reproduce/` directory can be copied anywhere and used as-is — it does not depend on any external directory layout.

**All external dependencies live in `config.env`**; edit only that file when moving machines. Every entry uses `${VAR:-default}` form so it can also be overridden by an environment variable:

| Variable | Description |
|---|---|
| `WA_REPO` | WebArena repo (`evaluation_harness` / `browser_env` / `config_files`) |
| `TSA_REPO` / `VORTEX_REPO` | TreeSparseAttention (with `build_sm90/` patches applied) / vortex_torch |
| `MODEL_PATH` / `JUDGE_MODEL_PATH` | model under test / scoring judge |
| `AGENT_PY` / `SERVE_PY` / `QUEST_PY` / `UTIL_PY` | four Python environments (`SERVE_PY`'s bin dir must contain `ninja`) |
| `RESULTS_DIR` | output directory |
| `WA_FORCE_PROXY` / `SCORE_PROXY` | proxies for the agent / for scoring-time navigation (both must point at a working SOCKS proxy) |
| `WA_MAP_URL` | map site (real OSM in this experiment) |
| `TS_CUDA_ARCHS` | GPU arch: H100 = `90`, B200 = `100` |

```bash
source config.env    # scripts source it automatically; source once before running commands by hand
```

**`harness/` (required for reproduction)** — these determine experimental correctness, especially login isolation:

| File | Role |
|---|---|
| `run_batch.py` | site-aware scheduler: the "1 task per replica for state-mutating sites" isolation logic, `RO_CAP` for read-only sites, retry on connection failure, round-robin over LLM endpoints |
| `run_task.py` | single-task runner: injects the anti-public-internet preamble, `use_vision=False`, merges the login cookies of the replicas named in `replica_map`, raised browser event timeouts, per-call LLM logging |
| `gen_auth.py` | logs into every (site, replica) pair and saves a `storage_state` |
| `wa_config.py` | site/replica URLs and accounts (one persistently unhealthy gitlab replica removed) |

> These are the **exact versions used for this experiment**, including two necessary modifications: `run_task.py` honours `WA_FORCE_PROXY` to force SOCKS (upstream only probes HTTP proxies), and `wa_config.py` drops the dead replica.

**`scripts/` (used for this experiment)**

| File | Role |
|---|---|
| `run_exp2.sh` | main driver: method list, concurrency 12, TSA timeout 2400s, auto-skip of already-finished methods |
| `serve_tsa4.sh` | 4× single-GPU TSA servers (`--scoring-mode centroid\|minmax`, `--top-k`, `TS_CUDA_ARCHS=90`) |
| `serve_dense_tp4.sh` | dense: vLLM tp=4 |
| `serve_quest.sh` | quest: sglang + vortex (topk=61, page 16, chunk 64, ratio 0.0625) |
| `rescore_all.sh` | scoring driver, **includes the `ninja` PATH fix** (without it the judge fails to start silently) |
| `metrics2.py` | input/output lengths, %chunks, quest %pages |
| `tok_cov.sh` | real TSA selected-token coverage (full-model `eval_recall`) |

**End-to-end reproduction** (`run_exp2.sh` chains the first three steps and loops over methods automatically):

```bash
source config.env

# 0) one-time: ensure ninja is on PATH, nltk punkt is downloaded, replicas are healthy, SOCKS is up
export PATH=$(dirname $SERVE_PY):$CUDA_BIN:$PATH

# 1) per method: reset the sites (see "Site reset") -> regenerate login cookies
AUTH_PROXY=$WA_FORCE_PROXY AUTH_FORCE=1 \
  $AGENT_PY harness/gen_auth.py --sites shopping shopping_admin reddit gitlab --replicas 10

# 2) start the servers for that method (pick one)
bash scripts/serve_dense_tp4.sh                                  # dense
SCORING=centroid TOPK=64 MAXB=8 bash scripts/serve_tsa4.sh       # TSA (centroid|minmax, top_k)
QUEST_TP=4 QUEST_PORT=30000 bash scripts/serve_quest.sh          # quest

# 3) run the tasks (main driver: concurrency/timeout policy, resumable)
bash scripts/run_exp2.sh

# 4) score: official + lenient (starts the judge automatically)
bash scripts/rescore_all.sh

# 5) metrics
$UTIL_PY scripts/metrics2.py <method result dir> tsa|quest|dense [top_k]   # lengths / %chunks / %pages
bash scripts/tok_cov.sh                                                    # real TSA selected-token coverage
```

**`serve_reference/` (reference only, NOT used in this experiment)** — serving scripts from the original sr_compare bundle (`serve_dense_4x.sh`, `serve_tsa_4x.sh`, `serve_tsa_new.sh`, `serve_vortex_v59_4x.sh`, `serve_judge.sh`, `launch_vortex_textonly.py`). They target **a different machine setup** (different venv paths, HTTP proxies, 4 single-GPU dense replicas, …) and were **not used here**; they are kept only as a parameter reference (e.g. the full vortex/quest parameter set, the judge's tp configuration). To reproduce, use the versions under `scripts/`.

## Task ID details

`task_ids/` holds four files per method, one task id per line:

```
<method>.lenient_pass.txt    <method>.lenient_fail.txt
<method>.official_pass.txt   <method>.official_fail.txt
```

method ∈ {`dense`, `quest`, `tsa_minmax_tk64`, `tsa_centroid_tk64`, `tsa_centroid_tk32`}

| Method | lenient pass/fail | official pass/fail |
|---|---|---|
| dense | 33 / 67 | 23 / 77 |
| quest | 33 / 67 | 26 / 74 |
| tsa_minmax_tk64 | 34 / 66 | 19 / 81 |
| tsa_centroid_tk64 | 28 / 72 | 14 / 86 |
| tsa_centroid_tk32 | 26 / 74 | 9 / 91 |

## official vs lenient

**official (strict)** = the standard WebArena evaluator. Depending on each task's `eval_types` it runs three kinds of checks: `string_match` (`exact_match` exact string / `must_include` must contain all key strings / `fuzzy_match` semantic equivalence judged by an LLM), `url_match` (compare the agent's final URL), and `program_html` (open the page with playwright and run DOM assertions). A task passes only if `score == 1`.

**lenient** = on top of official, **only the tasks official marked as failed** are re-judged once by an LLM, specifically to recover false negatives where the answer states the correct value but fails strict string matching on phrasing/formatting. The judge prompt marks CORRECT if the answer **clearly states the reference value** (ignoring surrounding words, capitalization, punctuation, symbols like ™); it marks INCORRECT if the answer gives a wrong value, misses required items, is vague/uncertain, says it could not find it, or lists several guesses without committing. `url_match` / `program_html` tasks require real navigation and are **skipped, not re-judged** (official result kept).

So `lenient ≥ official` always holds, and the gap is the number of recovered formatting false-negatives. In this experiment the map tasks run against the **real openstreetmap.org** while the reference answers were annotated on a self-hosted snapshot, so official is systematically low — **lenient is therefore the primary metric**.

## Scoring scripts

Under `scoring/`:

| File | Role |
|---|---|
| `score_batch.py` | official scoring entry point; dispatches `score_one.py` per task concurrently, aggregates into `SCORES.json` |
| `score_one.py` | official evaluation of a single task (all three evaluator kinds; `fuzzy_match` calls the LLM judge) |
| `rescore_lenient.py` | lenient re-judging; reads `SCORES.json`, writes `SCORES_adjusted.json` |

`wa_config.py` (site/replica URLs and accounts) lives in `harness/` and is imported from there — a single copy, to avoid divergence.

```bash
source config.env
# one-shot: start the judge and run official + lenient for every method (recommended; includes the ninja PATH fix)
bash scripts/rescore_all.sh

# or score a single method directory:
SCORE_PROXY=$SCORE_PROXY JUDGE_BASE_URL=http://127.0.0.1:18000/v1 JUDGE_MODEL=llama-judge \
  $AGENT_PY scoring/score_batch.py --results-dir <method result dir> \
  --judge-url http://127.0.0.1:18000/v1 --judge-model llama-judge --concurrency 8
$AGENT_PY scoring/rescore_lenient.py <method result dir>   # lenient; requires SCORES.json first
```

The judge here is Llama-3.3-70B-Instruct (vLLM, `--served-model-name llama-judge`). Note that the judge URL and model name are **hardcoded** inside `rescore_lenient.py` (`127.0.0.1:18000`, `llama-judge`).

## Concurrency / batch limits (tied to login correctness and GPU utilization)

Three layers, outermost first:

1. **Global concurrency** `--max-concurrency 12` (same for all five methods) — the cap on simultaneously running tasks.
2. **Per-site replica slots** (the real physical limit, and **the key to login working at all**): **state-mutating sites** (shopping / shopping_admin / gitlab / reddit) are limited to **1 task per replica**. With ~9-10 replicas per site across the two machines, that site can run at most ~9-10 concurrent tasks. If two tasks share the same replica *and* account, the later login evicts the earlier session and the agent hits a login wall. **Read-only sites** (map / wikipedia / homepage) need no isolation: `RO_CAP = 16` per replica.
3. **Server-side batch**: each TSA server uses `--max-batch-size` 8 (centroid) / 4 (minmax), across 4 single-GPU servers; dense / quest use tp=4 continuous batching.

Concurrency 12 was tuned: at 8, **2 of the 4 GPUs sat idle** (agents spend roughly half their time waiting on page loads, so 2 tasks per server cannot keep a GPU busy); at 12 the four GPUs ran at 65-96%.

### Why TSA's batch cannot be raised (GPU memory + arrival rate)

TSA's `serve.py` is already a **long-running online server** (with an 80 ms `batch_collect_ms` collection window) waiting for requests, so the bottleneck is not "not mounted". There are two real limits:

**(a) A hard memory ceiling — again rooted in the lack of tensor parallelism**

| | Weights | Free per GPU | KV per request (@9k) | Theoretical max concurrency |
|---|---|---|---|---|
| TSA (single GPU) | **66 GB / 93 GB** | **27 GB** | 2.36 GB | **~11** (before activations / workspace / graph buffers) |
| dense (tp=4) | 16 GB/GPU | 77 GB/GPU | 2.36 GB | 306 GB aggregate KV headroom across 4 GPUs |

32B bf16 weights are 66 GB, consuming 71% of a single H100 and leaving only 27 GB for KV. Each 9k-context request needs `64 layers × 9000 × 8 kv_heads × 128 dim × 2 (K,V) × 2 bytes ≈ 2.36 GB`, so roughly a dozen fit at most. That is why `--max-batch-size` is 8 (centroid) / 4 (minmax). dense with tp=4 shards the weights to 16 GB per GPU with 77 GB free, giving far more KV room. **In other words, "no tensor parallelism" causes both the ~4× per-request slowdown and the memory-bound batch ceiling.**

**(b) In practice the tighter constraint is "not enough requests"**

Batch sizes actually observed in the server logs: `bs=1` 73 times, `bs=2` 72, `bs=3` 20, `bs=4` 22 — **most of the time only 1-2 requests were batched, never hitting the cap**. With 12 concurrent agents spread over 4 servers that is 3 per server, and agents spend nearly half their time waiting on page loads (through the SOCKS proxy; the real OSM is especially slow), so each server typically has only 1-2 live requests at any instant. **Raising `max-batch-size` would not help — there simply are not more requests to batch**; and raising concurrency to fill the batch is limited by the "1 task per replica" rule for mutating sites and by queue-tail latency pushing timeouts back up.

Also, **prefill is serialized** (logs show `Pre-allocated buffers for ... max_batch_size=1` and `Prefilling request i/n` one at a time); only decode is batched.

To make TSA throughput genuinely comparable, the fix is to **add tensor parallelism to `serve.py`**, not to raise the batch size.

> **Hardware caveat: everything above is measured on H100 (93 GB) and changes with the GPU.** The memory arithmetic depends entirely on "card capacity − 66 GB of weights": H100 leaves 27 GB → ~11 concurrent requests; on a **B200 (192 GB HBM3e)** about 126 GB would remain → theoretically **~53 concurrent requests**, largely removing the memory ceiling and allowing a much larger `max-batch-size`. B200 also has far higher memory bandwidth (~8 TB/s vs H100 ~3.35 TB/s), so per-request decode would be considerably faster too. The quantitative claims here ("batch only reaches 4/8", "TSA is ~4× slower") **hold only for this H100 setup** and should not be extrapolated to B200/GB200.
>
> Two things are GPU-independent, however: (1) as long as `serve.py` lacks tensor parallelism, a single request can only use one GPU's bandwidth, so the gap versus tp=4 remains; (2) at ~8k context, KV accounts for only a few percent of memory traffic, so **sparse attention cannot save much** — that is arithmetic, not hardware, and demonstrating the value of sparsity requires far longer contexts.

## Timeouts (tied to the speed differences between methods)

Two levels:

- **Per-LLM-call timeout, 300s** (`run_task.py` default). TSA's maximum observed call latency is 295-299s, meaning **some requests are genuinely being cut off at this ceiling**.
- **Per-task timeout `--task-timeout`**: dense / quest = **1500s**, **TSA = 2400s**.

Why only TSA was relaxed — look at the measured median task wall-clock time:

| Method | median LLM call latency | p90 | **median task wall time** |
|---|---|---|---|
| quest | 7.3s | 10.2s | **113s** |
| dense | 15.5s | 21.6s | **255s** |
| TSA-minmax tk64 | 32.5s | 127.1s | **1151s** |
| TSA-centroid tk32 | 29.0s | 166.9s | **1397s** |
| TSA-centroid tk64 | 39.5s | 218.5s | **1663s** |

The median TSA task takes 1151-1663s, i.e. **at or beyond the 1500s cap**; measured at 1500s, **63% of TSA tasks were marked as timeouts**, and timeouts count as failures, which corrupts the results (not a weak method, just an unfinished run). dense (255s) and quest (113s) have 6-13× headroom at 1500s and needed no change.

After raising it to 2400s, ~25% of TSA tasks still time out — almost all of them map tasks against the **real openstreetmap.org** (proxied, often 31 steps). Those are hard for every method and would fail anyway; the table lists map as its own column so the "non-map subtotal" can be read in isolation.

> Note: TSA is slow mainly because its `serve.py` **has no tensor parallelism** — the 32B model runs on a single GPU, whereas dense/quest use tp=4 (four GPUs serving one request). Decode is memory-bandwidth bound, so a single request is ~4× slower; and because the agent loop is sequential, a 2-4× per-call slowdown becomes a 4-6× slowdown in total task time. **This data therefore cannot support any speed/efficiency conclusions** — the parallelism is not matched. Accuracy conclusions are unaffected (timeouts were compensated with 2400s).

## Login / auth (regenerate for every method)

**This is the easiest thing to get wrong, and its symptoms are the most misleading.** If the login state is wrong, the agent stops at a login page and the final answer becomes "I don't have your login credentials", which is scored as a failure — it **looks like a weak method** when in fact the environment was misconfigured.

Key points:
1. **Generate cookies per replica.** Different replicas of the same site (different ports) are **independent container instances**; even with the same account, their sessions are independent. Use `gen_auth.py` to log in once per `(site, replica)` and save one `storage_state` each:
   ```bash
   source config.env
   AUTH_PROXY=$WA_FORCE_PROXY AUTH_FORCE=1 \
     $AGENT_PY harness/gen_auth.py --sites shopping shopping_admin reddit gitlab --replicas 10
   ```
   Cookies are written to `harness/auth/` by default (override with `WA_AUTH_DIR`).
2. **Use the cookie of the replica the task was assigned.** `run_task.py` reads the scheduler's `replica_map` and merges the cookies of that task's replicas into the browser. Using the wrong replica's cookie is equivalent to not being logged in.
3. **Regenerate for every method** (sessions expire over time and on reset). One round takes ~3-4 minutes (~39 replicas, sequential over SOCKS).
4. **Never let two concurrent tasks share the same replica *and* account** — the later login evicts the earlier session and both tasks are wasted. This is exactly why mutating sites are limited to 1 task per replica.

Sanity check: after a run, spot-check some tasks' `final_url` to confirm it lands on a post-login page (e.g. `/admin/customer/index/`), and grep the answers for "log in" / "credentials".

## Pitfalls and gotchas (checklist)

**Environment / build**
- **`ninja` must be on `PATH`** (e.g. `/home/cc/serve-venv/bin`). Both vLLM workers and `eval_recall` need JIT compilation; when it is missing the failure is **silent** — the judge server never starts, every scoring call returns `Connection refused`, and results show up as `score=None`, which counts as task failure and is easily misread as "the method is bad". **This bit us twice.** Sanity check: after scoring, confirm the number of `score is None` entries in `SCORES.json` is **0**.
- **nltk `punkt` / `punkt_tab` must be pre-downloaded.** `evaluation_harness/evaluators.py` uses `word_tokenize` for `must_include` / `exact_match`; a missing corpus raises and silently scores those tasks 0.
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce memory fragmentation at long contexts.
- A 32B model **crashes in flashinfer sampling when vLLM is started single-GPU**; use `--tensor-parallel-size 4` for dense and the judge.

**Proxy / sites**
- Everything goes through an ssh `-D` **SOCKS tunnel**. `run_task.py` by default probes a set of HTTP proxies (18900-18907), which do not exist in this environment — override with `WA_FORCE_PROXY=socks5://127.0.0.1:1080`.
- **Scoring has its own trap**: `score_one.py`'s `SCORE_PROXY` defaults to `http://127.0.0.1:18900` (dead). `url_match` / `program_html` need real navigation, so `SCORE_PROXY=socks5://127.0.0.1:1080` must be passed explicitly.
- **Health-check replicas before using them.** We found one gitlab replica persistently unhealthy; it must be removed from the pool or the scheduler will dispatch tasks to it and burn the retry budget.

**Agent configuration**
- **`max_tokens` must be at least 4096.** At 2048 browser-use raises `ModelOutputTruncatedError` (the structured output is truncated) and that step's action is lost.
- **Vision must be off** (`use_vision=False`, plus the chromium flag `--blink-settings=imagesEnabled=false`). vortex/quest does not handle mrope and emits garbage with images.
- **The "do not browse the public internet" preamble must be injected.** Otherwise the agent wanders off to google.com / github.com / the real reddit and burns steps. It is built into `run_task.py`.
- **Raise the browser event timeouts** (`TIMEOUT_NavigateToUrlEvent=90`, Click/Refresh 45, …). Sites load slowly through the proxy and the defaults cause many spurious navigation failures.

**TSA-specific**
- With the old implementation, **minmax at low top_k hits a CUDA illegal memory access when batched** (we were forced down to `batch=1`); with the GPU envelope kernel it batches normally.
- **The TSA server does not return `prompt_tokens`** (always 0); input lengths must be re-tokenized.
- **The envelope (min-max) scoring formula is wrong**: Quest's criterion takes the max **per channel and then sums**, whereas the current implementation sums first (and even averages over heads) before taking the max, which loses the upper-bound property. See `../../TSA_envelope_scoring_review.md`.
- **Envelope has a size bias on variable-length chunks**: at the same top_k, minmax actually retains 78.3% of tokens versus centroid's 57.5%, so the end-to-end comparison between the two is **not a matched-budget comparison**.

**Fairness warning for any speed claims**
- In this experiment **dense ran with `--enforce-eager` (CUDA graph disabled)** while quest / TSA had CUDA graph enabled, and TSA was single-GPU while dense/quest were tp=4. **This data may therefore be used only for accuracy comparisons, never for speed or throughput conclusions.**

## Site reset (mandatory before every method)

WebArena sites are **stateful**: while exploring, the agent may modify carts, orders, issues, etc. Without a reset, whichever method runs first leaves state behind for the next one, making the comparison unfair. **Every experiment must therefore reset the sites before each method.**

**Granularity: reset per *method*, not per *task*.** A single reset round takes 15-25 minutes, so per-task resets are infeasible (100 tasks = tens of hours); within one method all tasks share the same initial state, which is consistent for that method, and per-method granularity is enough to keep methods comparable.

**How (no sudo / root needed)**: the containers run under the `wangxian` account using its rootless docker socket, so simply ssh in as that account and call the existing reset script:

```bash
# run in parallel for each replica w (w = 0,1,2 ...)
timeout 1500 ssh -i <key> -o BatchMode=yes wangxian@hilbit2.cis.upenn.edu \
  "cd /z/wangcy07/webarena-repo && \
   PYTHONPATH=/z/wangcy07/webarena-repo /z/wangcy07/webarena-venv/bin/python \
   ~/wa_reset_wx.py shopping,shopping_admin,gitlab,reddit <w>"
```

**A health gate must follow the reset** — do not start immediately (containers, gitlab especially, boot slowly and there is an intermediate state where the port answers but the content is not restored). Poll each replica: shopping (`/catalogsearch/result/?q=bag` returns > 0 `product-item`), gitlab, admin and reddit returning 200/302; up to 20 polls × 15s; retry the whole round if still unhealthy.

**Timing and timeout**: the 4 sites × replicas run **in parallel**, taking about **15-25 minutes** per round, plus up to 5 minutes of health polling.

> ⚠️ **The timeout must be at least 1500 seconds.** An early value of 600s killed the process mid-restore and left sites half-broken (gitlab not starting, shopping returning no products) — worse than not resetting, and requiring another full reset round to fix.

Reset and login are two separate things: reset restores **site data**, after which login cookies must still be **regenerated** (see the login section). Both are required.

## sm100 → sm90 port (the original targets Blackwell; we run on H100)

TreeSparseAttention was written for **sm_100 / sm_120 (B200, consumer Blackwell)** — `python/jit_build.py` defaults to `_DEFAULT_TS_CUDA_ARCHS = "100;120"`, and `CMakeLists.txt` hardcodes `CMAKE_CUDA_ARCHITECTURES "100"` with a "force B200 only" comment. Our machine is **H100 (sm_90)**, where it does not compile as-is. Four changes were needed (patches in `build_sm90/`).

| Patch | File | What / why |
|---|---|---|
| `01-cmakelists-sm100-to-sm90.patch` | `CMakeLists.txt` | arch `100` → `90`: `CMAKE_CUDA_ARCHITECTURES`, `CMAKE_CUDA_ARCHITECTURES_ALL`, `--generate-code=arch=compute_100,code=sm_100` → `compute_90/sm_90`, and the per-target `CUDA_ARCHITECTURES` that overrides PyTorch's |
| `02-jit_build-sources-flags-ldflags.patch` | `python/jit_build.py` | (1) **add 7 missing `.cu` sources** (`ts_decode`, `ts_fused_qknorm_rope`, `ts_fused_page_select`, `ts_rms_norm`, `ts_kv_rearrange`, `ts_sparse_decode_tma`, `ts_sparse_decode_xqa`), otherwise the JIT build does not link all symbols; (2) add the `-U__CUDA_NO_HALF*_OPERATORS__` / `-U__CUDA_NO_BFLOAT16*` un-defines — torch's extension build disables half2/bf16 operator overloads that flashinfer relies on, breaking compilation under CUDA 13; (3) add `extra_ldflags=["-L/usr/lib/x86_64-linux-gnu","-lcuda"]` because the TMA decode kernels call the driver API `cuTensorMapEncodeTiled` and need `libcuda` linked explicitly |
| `03-flashinfer-cuda13-prefill-guards.patch` | `3rdparty/flashinfer`: `prefill.cuh`, `handler.cuh` | wrap the **host-side BatchPrefill wrappers that the decode path never uses** in `#if 0 … #endif` — their SWITCH macros fail to expand under CUDA 13's nvcc. Only unused code is disabled; the decode path we rely on is untouched |
| runtime env | — | always `export TS_CUDA_ARCHS=90` when building/serving (the default is `100;120`); the JIT cache is keyed per arch so stale binaries for another arch are not picked up |

Two more requirements, unrelated to the port but equally necessary: `ninja` must be on `PATH` (e.g. `/home/cc/serve-venv/bin`), otherwise the vLLM / `eval_recall` JIT workers fail silently; and `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` helps with memory fragmentation at long contexts.

```bash
# apply from the TreeSparseAttention repo root
git apply build_sm90/01-cmakelists-sm100-to-sm90.patch
git apply build_sm90/02-jit_build-sources-flags-ldflags.patch
git -C 3rdparty/flashinfer apply ../../build_sm90/03-flashinfer-cuda13-prefill-guards.patch
export TS_CUDA_ARCHS=90 PATH=/path/to/venv/bin:$PATH
```

> Note: switching to sm_90 only makes it **run** on H100 — no Hopper-specific performance tuning was done, and Blackwell-oriented paths in the original (e.g. the newer TMA/XQA kernels) may not take the optimal branch on sm_90.
