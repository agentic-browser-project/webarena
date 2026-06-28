"""Print a cumulative 3-way pass-rate comparison (+ dense baseline) through task <upto>.
Reads each config's SCORES.json / SCORES_adjusted.json / sparse_metrics_summary.json.
Usage: python stage_summary.py <upto_task_id>
"""
import sys, json, os, glob

RES = os.environ.get("WA_RESULTS", "/home/cc/temp/webarena/sr_compare/results")
UPTO = int(sys.argv[1]) if len(sys.argv) > 1 else 812
CONFIGS = [("dense (812, ref)", "dense"),
           ("TSA top_k=128", "tsa_tk128"),
           ("TSA top_k=96", "tsa_tk96"),
           ("TSA top_k=64", "tsa_tk64"),
           ("TSA top_k=48", "tsa_tk48"),
           ("TSA top_k=32", "tsa_tk32"),
           ("vortex-block (topk30,rec)", "vortex"),
           ("vortex-quest (topk61,rec)", "quest_rec")]

def load(p):
    try: return json.load(open(p))
    except Exception: return {}

def count_done(d):
    return len(glob.glob(os.path.join(RES, d, "task_*", "task_*.json")))

print(f"{'config':<26} {'done':>6} {'official pass':>16} {'lenient pass':>16} {'sparse(chunk/page)':>20}")
print("-" * 88)
for name, d in CONFIGS:
    sc = load(os.path.join(RES, d, "SCORES.json"))
    adj = load(os.path.join(RES, d, "SCORES_adjusted.json"))
    sm = load(os.path.join(RES, d, "sparse_metrics_summary.json"))
    done = count_done(d)
    n = sc.get("n", 0)
    npass = sc.get("n_pass")
    off = f"{npass}/{n} ({100*sc.get('pass_rate',0):.1f}%)" if n else "-- not scored --"
    apass = adj.get("adjusted_pass")
    al = f"{apass}/{adj.get('n',n)} ({100*adj.get('adjusted_rate',0):.1f}%)" if apass is not None else "--"
    cr = sm.get("chunk_ratio_mean"); pr = sm.get("page_ratio_mean")
    sp = (f"{cr}/{pr}" if cr is not None else (f"-/{pr}" if pr is not None else "--"))
    print(f"{name:<26} {done:>6} {off:>16} {al:>16} {sp:>20}")
print("-" * 88)
print(f"(cumulative through task {UPTO}; dense is the full-812 reference. "
      f"sparse: mean selected/total chunk ratio (TSA) and page ratio.)")
