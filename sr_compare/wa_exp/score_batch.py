"""Score all completed tasks in a results dir and report pass rate.

Spawns score_one.py per task (own subprocess) with the task's replica URLs in the env so
program_html func: helpers resolve. Aggregates pass rate overall and per-site.
"""
import os, sys, json, argparse, subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, "/home/cc/temp/webarena/sr_compare/wa_exp")
import wa_config as C

ENV_VAR = {"shopping": "SHOPPING", "shopping_admin": "SHOPPING_ADMIN",
           "reddit": "REDDIT", "gitlab": "GITLAB", "wikipedia": "WIKIPEDIA", "map": "MAP"}


def score_dir(task_dir, args):
    tid = int(os.path.basename(task_dir).replace("task_", ""))
    task = json.load(open(os.path.join(task_dir, "input.json")))
    rmap = task.get("replica_map", {})
    env = dict(os.environ)
    # set every site URL to this task's replica (others default to primary)
    for site, var in ENV_VAR.items():
        ri = rmap.get(site, 0)
        env[var] = C.base_url(site, ri)
    env["HOMEPAGE"] = C.base_url("homepage", 0)
    env["JUDGE_BASE_URL"] = args.judge_url
    env["JUDGE_MODEL"] = args.judge_model
    try:
        p = subprocess.run([sys.executable, "/home/cc/temp/webarena/sr_compare/wa_exp/score_one.py",
                            "--task-dir", task_dir], env=env, capture_output=True,
                           text=True, timeout=args.timeout)
        line = [l for l in p.stdout.strip().splitlines() if l.startswith("{")]
        if line:
            return json.loads(line[-1])
        return {"task_id": tid, "score": None, "error": f"no_output rc={p.returncode}: {p.stderr[-200:]}"}
    except subprocess.TimeoutExpired:
        return {"task_id": tid, "score": None, "error": "scorer_timeout"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", required=True)
    ap.add_argument("--judge-url", default="http://127.0.0.1:18000/v1")
    ap.add_argument("--judge-model", default="qwen3vl-dense")
    ap.add_argument("--concurrency", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=300)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    dirs = sorted([os.path.join(args.results_dir, d) for d in os.listdir(args.results_dir)
                   if d.startswith("task_") and
                   os.path.exists(os.path.join(args.results_dir, d, "input.json")) and
                   os.path.exists(os.path.join(args.results_dir, d,
                       f"task_{d.replace('task_','')}.json"))])
    print(f"scoring {len(dirs)} tasks", flush=True)
    results = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(score_dir, d, args): d for d in dirs}
        for f in as_completed(futs):
            r = f.result()
            results.append(r)
            sc = r.get("score")
            print(f"  task {r['task_id']}: score={sc} {r.get('error') or ''}", flush=True)

    results.sort(key=lambda x: x["task_id"])
    # pass rate: null scores count as 0
    scored = [r for r in results]
    n = len(scored)
    npass = sum(1 for r in scored if r.get("score") == 1.0)
    nnull = sum(1 for r in scored if r.get("score") is None)
    # per-site
    by_id = {t["task_id"]: t for t in C.load_raw_tasks()}
    per_site = {}
    for r in scored:
        sites = tuple(by_id[r["task_id"]]["sites"]) if r["task_id"] in by_id else ("?",)
        key = "+".join(sites)
        per_site.setdefault(key, [0, 0])
        per_site[key][1] += 1
        if r.get("score") == 1.0:
            per_site[key][0] += 1
    summary = {"results_dir": args.results_dir, "n": n, "n_pass": npass,
               "pass_rate": round(npass / n, 4) if n else None,
               "n_null": nnull,
               "per_site": {k: {"pass": v[0], "n": v[1], "rate": round(v[0]/v[1], 3)}
                            for k, v in sorted(per_site.items())},
               "tasks": results}
    out = args.out or os.path.join(args.results_dir, "SCORES.json")
    json.dump(summary, open(out, "w"), indent=2)
    print(f"\n=== PASS RATE {npass}/{n} = {summary['pass_rate']} (null={nnull}) ===")
    for k, v in summary["per_site"].items():
        print(f"  {k}: {v['pass']}/{v['n']} = {v['rate']}")
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
