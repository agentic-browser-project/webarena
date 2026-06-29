# Task outcomes — per-method passed/failed task IDs (self-hosted-map run)

Pass/fail task-ID breakdown for the **8 attention methods** from our WebArena run on
**Qwen3-VL-32B**, using the **original self-hosted map site** (`:13000`) — i.e. the run
**before** map was switched to real `openstreetmap.org`. (The map-on-OSM re-run is a separate
experiment and is **not** included here.)

Only tasks that were actually **run and scored** are listed; un-run tasks are excluded.

## Coverage
- `dense` ran the full **812** tasks (it's the reference).
- The 7 sparse methods ran tasks **0–399** (the subset completed before the map-site switch), i.e. ~400 each.

## Files
- `<method>.json` — one per method (see list below).
- `summary.md` — one-line-per-method counts table.

### Methods (file ↔ results dir)
| file | method | results dir | retained KV (page%) |
|---|---|---|---|
| `dense.json` | dense (full attention) | `qwenvl_full` | 100% |
| `tsa_tk128.json` | TreeSparseAttention top_k=128 | `tsa_tk128` | ~87% |
| `tsa_tk96.json` | TSA top_k=96 | `tsa_tk96` | ~78% |
| `tsa_tk64.json` | TSA top_k=64 | `tsa_tk64` | ~60% |
| `tsa_tk48.json` | TSA top_k=48 | `tsa_tk48` | ~46% |
| `tsa_tk32.json` | TSA top_k=32 | `tsa_tk32` | ~29% |
| `vortex_block.json` | vortex block-sparse (official, topk=30) | `vortex` | ~5.7% |
| `vortex_quest.json` | vortex quest (official, topk=61) | `quest_rec` | ~11% |

## Each `<method>.json`
```json
{
  "config": "tsa_tk128",
  "results_dir": "tsa_tk128",
  "scoring": "lenient = official WebArena pass OR format-only-fixed (matches 'success rate'); official = strict WebArena.",
  "n_run": 400,
  "lenient":  { "n_pass": 99, "passed_ids": [...], "failed_ids": [...] },
  "official": { "n_pass": 81, "passed_ids": [...], "failed_ids": [...] },
  "execution_errored_ids": [...]
}
```
Field meanings:
- **`n_run`** — number of tasks this method actually ran & scored (un-run tasks excluded).
- **`lenient`** — pass/fail under the **lenient** criterion = official pass **OR** the answer was correct
  but only failed strict scoring on output formatting (this is the "success rate" used in our tables).
  `passed_ids` + `failed_ids` together == all run tasks.
- **`official`** — pass/fail under **strict** WebArena scoring (string_match / url_match / program_html;
  fuzzy_match judged by Llama-3.3-70B).
- **`execution_errored_ids`** — tasks that did **not execute cleanly** (timeout / crash); these are a
  subset of the failed IDs, listed separately so you can tell "ran but wrong answer" from "didn't finish".

## Quick reference (from `summary.md`)
| method | n_run | lenient_pass | official_pass | exec_errors |
|---|---|---|---|---|
| dense | 812 | 200 (24.6%) | 178 (21.9%) | 15 |
| tsa_tk128 | 400 | 99 (24.8%) | 81 (20.2%) | 0 |
| tsa_tk96 | 400 | 88 (22.0%) | 68 (17.0%) | 0 |
| tsa_tk64 | 400 | 82 (20.5%) | 64 (16.0%) | 0 |
| tsa_tk48 | 400 | 68 (17.0%) | 53 (13.2%) | 0 |
| tsa_tk32 | 400 | 70 (17.5%) | 52 (13.0%) | 0 |
| vortex_block | 400 | 82 (20.5%) | 63 (15.8%) | 0 |
| vortex_quest | 400 | 86 (21.5%) | 71 (17.8%) | 0 |

> Note: to compare methods apples-to-apples, intersect on the **same task subset** (e.g. tasks 0–399,
> which all methods ran). `dense`'s extra tasks (400–811) have no sparse counterpart yet.
