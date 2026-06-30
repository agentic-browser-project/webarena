"""Build a task_outcomes/<method>.json in block1 format from a scored results dir.
Reads SCORES.json (official) + SCORES_adjusted.json (lenient) + per-task json (exec errors).
Usage: python build_block2_outcome.py <results_dir> <config_name> <out_json>
"""
import json, os, sys

res, cfg, out = sys.argv[1], sys.argv[2], sys.argv[3]
scores = json.load(open(os.path.join(res, "SCORES.json")))
adj = json.load(open(os.path.join(res, "SCORES_adjusted.json")))

official_pass, official_fail, exec_err = [], [], []
for t in scores["tasks"]:
    tid = t["task_id"]
    (official_pass if t.get("score") == 1.0 else official_fail).append(tid)
    # exec error = the agent run did not finish cleanly (exception/timeout/crash)
    try:
        r = json.load(open(os.path.join(res, f"task_{tid}", f"task_{tid}.json")))
        if r.get("error"):
            exec_err.append(tid)
    except Exception:
        exec_err.append(tid)

lenient_pass = sorted(set(official_pass) | set(adj.get("format_fixed_ids", [])))
all_ids = sorted(t["task_id"] for t in scores["tasks"])
lenient_fail = sorted(set(all_ids) - set(lenient_pass))

doc = {
    "config": cfg,
    "n_run": len(all_ids),
    "lenient":  {"n_pass": len(lenient_pass), "passed_ids": lenient_pass, "failed_ids": lenient_fail},
    "official": {"n_pass": len(official_pass), "passed_ids": sorted(official_pass), "failed_ids": sorted(official_fail)},
    "execution_errored_ids": sorted(exec_err),
}
json.dump(doc, open(out, "w"), indent=2)
print(f"{cfg}: n_run={doc['n_run']} lenient={len(lenient_pass)} official={len(official_pass)} exec_err={len(exec_err)} -> {out}")
