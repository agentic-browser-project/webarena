# Task outcomes — per-method passed/failed task IDs

Pass/fail task-ID breakdowns for the WebArena attention-sparsity runs on **Qwen3-VL-32B**.
There are **two runs**, in two subfolders:

| folder | run | map site | methods | tasks |
|---|---|---|---|---|
| `self_hosted_map/` | main experiment | **self-hosted** WebArena OSM (`:13000`) | 8 (dense + TSA 128/96/64/48/32 + vortex-block + vortex-quest) | dense=812, sparse=0–399 |
| `map_osm_block1/` | map re-run, block 1 | **real** `openstreetmap.org` | 6 (dense + TSA 128/64/32 + vortex-block + vortex-quest) | the 51 single-site **map** tasks in 0–100 |

Only tasks that were actually **run and scored** are listed; un-run tasks are excluded.

## File format (every `<method>.json`)
```json
{
  "config": "tsa_tk128",
  "n_run": 51,
  "lenient":  { "n_pass": 14, "passed_ids": [...], "failed_ids": [...] },
  "official": { "n_pass": 6,  "passed_ids": [...], "failed_ids": [...] },
  "execution_errored_ids": [...]
}
```
- **`lenient`** — pass/fail under the **lenient** criterion = strict pass **OR** the answer was correct but
  only failed strict scoring on output formatting. This is the **"success rate"** used in our tables.
  `passed_ids` + `failed_ids` == all run tasks.
- **`official`** — pass/fail under **strict** WebArena scoring (string_match / url_match / program_html;
  fuzzy_match judged by Llama-3.3-70B).
- **`execution_errored_ids`** — tasks that did **not execute cleanly** (timeout / connection error / crash);
  a subset of the failed IDs, listed separately so you can tell "ran but wrong answer" from "didn't finish".

## `self_hosted_map/` — summary
| method | n_run | lenient_pass | official_pass | exec_errors |
|---|---|---|---|---|
| dense | 812 | 200 (24.6%) | 178 (21.9%) | **15** |
| tsa_tk128 | 400 | 99 (24.8%) | 81 (20.2%) | 0 |
| tsa_tk96 | 400 | 88 (22.0%) | 68 (17.0%) | 0 |
| tsa_tk64 | 400 | 82 (20.5%) | 64 (16.0%) | 0 |
| tsa_tk48 | 400 | 68 (17.0%) | 53 (13.2%) | 0 |
| tsa_tk32 | 400 | 70 (17.5%) | 52 (13.0%) | 0 |
| vortex_block | 400 | 82 (20.5%) | 63 (15.8%) | 0 |
| vortex_quest | 400 | 86 (21.5%) | 71 (17.8%) | 0 |

> Compare methods on the **same task subset** (0–399, which all methods ran); dense's extra 400–811 has no sparse counterpart.

## `map_osm_block1/` — summary (51 map tasks, REAL OSM)
| method | retained KV | lenient_pass | official_pass | exec_errors |
|---|---|---|---|---|
| dense | 100% | 10 (20%) | 4 (8%) | 0 |
| tsa_tk128 | 92.6% | 14 (27%) | 6 (12%) | 0 |
| tsa_tk64 | 72.7% | 14 (27%) | 7 (14%) | 0 |
| tsa_tk32 | 33.9% | 11 (22%) | 6 (12%) | 0 |
| vortex_block | 6.3% | 11 (22%) | 7 (14%) | 0 |
| vortex_quest | 12.0% | **16 (31%)** | **9 (18%)** | 0 |

> **Caveats for the OSM block:** n=51 (noise-dominated); and the WebArena map **reference answers were
> annotated on the self-hosted OSM**, so real-OSM routes/addresses often don't match the reference →
> **`official` is low (8–18%) by construction, not because the agent failed**. Use these for *relative*
> method behavior on real OSM, not as comparable-to-benchmark accuracy. (All methods score higher than on
> the self-hosted map, where real-OSM search/routing being more functional helps.)

## What is dense's `execution_errored` (the 15 in `self_hosted_map/dense.json`)?
dense's 15 execution errors are all **`ConnectTimeout`** — the agent's headless browser **could not connect
to the WebArena site** during those tasks (a transient outage of specific site replicas / the proxy),
**not** an LLM or attention problem. The IDs cluster in three contiguous groups of five:
**436–440, 506–510, 585–589** — the signature of a few site replicas being briefly down during those windows.

- **Were they re-tested?** **No.** `dense` was a pre-existing reference run, *outside* the self-healing
  staged loop (which deletes errored/not-done tasks and re-runs them). So these 15 connection timeouts were
  never retried. (The sparse configs' timeouts *were* re-run — but those were a different cause: *task*
  timeouts from TSA being slow, fixed by raising the task budget to 5400 s.)
- **Counted as pass or fail?** They have `score = null` → treated as **failed** (our criterion is
  `(score or 0) >= 1`). So they sit in dense's denominator: 200/812 lenient (24.6%).
- **Impact on the comparison:** **all 15 are in 400–811**, i.e. **outside the 0–399 region where dense is
  compared against the sparse methods** → **zero effect on the apples-to-apples comparison**. They only
  marginally lower dense's full-812 number; excluding them gives 200/797 = 25.1% (vs 24.6%).
- A clean re-test of these 15 (sites are up now) is pending/optional and would only nudge dense's full-812 figure.
