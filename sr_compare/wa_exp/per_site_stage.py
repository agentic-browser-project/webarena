"""Per-site LENIENT pass-rate comparison on a fixed task range (default 0-199),
across dense + the sparse configs that have run. Apples-to-apples (same task subset).
"""
import json, os, glob, sys

RES = os.environ.get("WA_RESULTS", "/home/cc/temp/webarena/sr_compare/results")
UPTO = int(sys.argv[1]) if len(sys.argv) > 1 else 200
RUNS = [("dense Qwen3-VL-32B", "dense"),
        ("TSA top_k=128", "tsa_tk128"),
        ("TSA top_k=96", "tsa_tk96"),
        ("TSA top_k=64", "tsa_tk64"),
        ("TSA top_k=48", "tsa_tk48"),
        ("TSA top_k=32", "tsa_tk32"),
        ("vortex-block topk30", "vortex"),
        ("vortex-quest topk61", "quest_rec")]
SITES = ["shopping", "shopping_admin", "reddit", "gitlab", "map", "multi"]
CN = {"shopping": "购物", "shopping_admin": "购物后台", "reddit": "论坛Reddit",
      "gitlab": "GitLab", "map": "地图Map", "multi": "多站点", "TOTAL": "总计"}

def sites_of(d, tid):
    for p in (f"{RES}/{d}/task_{tid}/task_{tid}.json", f"{RES}/{d}/task_{tid}/input.json"):
        try:
            s = json.load(open(p)).get("sites") or []
            if s: return s
        except Exception: pass
    return []

def stats(d):
    sc = json.load(open(f"{RES}/{d}/SCORES.json"))
    try: fixed = set(json.load(open(f"{RES}/{d}/SCORES_adjusted.json")).get("format_fixed_ids", []))
    except Exception: fixed = set()
    agg = {s: [0, 0] for s in SITES}; tot = [0, 0]
    for t in sc["tasks"]:
        tid = t["task_id"]
        if tid >= UPTO: continue
        lp = (t.get("score") or 0) >= 1 or tid in fixed
        ss = sites_of(d, tid)
        cat = ss[0] if len(ss) == 1 and ss[0] in agg else ("multi" if ss else "multi")
        agg[cat][0] += int(lp); agg[cat][1] += 1
        tot[0] += int(lp); tot[1] += 1
    return agg, tot

def c(pn):
    p, n = pn
    return f"{100*p/n:.0f}% ({p}/{n})" if n else "—"

cols = SITES + ["TOTAL"]
hdr = f"{'配置':<20}" + "".join(f"{CN[s]:>14}" for s in cols)
print(f"== 分站点 宽松打分 pass rate(任务 0-{UPTO-1},同一批任务) ==")
print(hdr); print("-" * (20 + 14 * len(cols)))
for label, d in RUNS:
    if not os.path.exists(f"{RES}/{d}/SCORES.json"):
        print(f"{label:<20} (未打分)"); continue
    agg, tot = stats(d)
    print(f"{label:<20}" + "".join(f"{c(agg[s]):>14}" for s in SITES) + f"{c(tot):>14}")
