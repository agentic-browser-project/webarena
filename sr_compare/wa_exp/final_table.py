"""One unified results table: 实测保留KV | 保留chunks(TSA) | success rate (LENIENT).
Covers all tested methods; blanks where data is missing.
"""
import json, os

RES = os.environ.get("WA_RESULTS", "/home/cc/temp/webarena/sr_compare/results")
# (label, dir, setting, is_dense)
ROWS = [
    ("Qwen3-VL-32B dense",      "dense",   "full attn",      True),
    ("Qwen3-32B dense",         "qwen332_full",  "full attn",      True),
    ("Llama-3.3-70B dense",     "llama_dense",   "full attn",      True),
    ("Llama-3.3-70B dense det", "llama_dense_det","full attn",     True),
    ("Gemma dense",             "gemma_dense",   "full attn",      True),
    ("Qwen3.6 dense",           "qwen36_dense",  "full attn",      True),
    ("Qwen3-VL AWM(workflow)",  "qwenvl_awm",    "full attn",      True),
    ("Qwen3-VL ctrl",           "qwenvl_ctrl",   "full attn",      True),
    ("TSA top_k=128",           "tsa_tk128",     "top_k=128",      False),
    ("TSA top_k=96",            "tsa_tk96",      "top_k=96",       False),
    ("TSA top_k=64",            "tsa_tk64",      "top_k=64",       False),
    ("TSA top_k=48",            "tsa_tk48",      "top_k=48",       False),
    ("TSA top_k=32",            "tsa_tk32",      "top_k=32",       False),
    ("vortex-block (rec)",      "vortex",        "topk=30",        False),
    ("vortex-quest (rec)",      "quest_rec",     "topk=61",        False),
]

def load(p):
    try: return json.load(open(p))
    except Exception: return None

def fmt_pct(x):
    return f"{100*x:.1f}%" if x is not None else "—"

print(f"{'方法':<26}{'设定':<17}{'实测保留KV':>11}{'保留chunks':>11}{'success(宽松)':>15}{'任务数':>8}")
print("-" * 90)
for label, d, setting, dense in ROWS:
    adj = load(f"{RES}/{d}/SCORES_adjusted.json")
    sm = load(f"{RES}/{d}/sparse_metrics_summary.json")
    # success (lenient)
    if adj:
        succ = f"{adj['adjusted_pass']}/{adj['n']} ({100*adj['adjusted_rate']:.1f}%)"
    else:
        succ = "—(未打分)"
    # retained KV (page_ratio); dense = 100%
    kv = "100%" if dense else (fmt_pct(sm["page_ratio_mean"]) if sm and sm.get("page_ratio_mean") is not None else "—")
    # retained chunks (TSA only)
    ch = fmt_pct(sm["chunk_ratio_mean"]) if (sm and sm.get("chunk_ratio_mean") is not None) else "—"
    n = adj["n"] if adj else "—"
    print(f"{label:<26}{setting:<17}{kv:>11}{ch:>11}{succ:>15}{str(n):>8}")
print("-" * 90)
print("success = 宽松打分通过率; 实测保留KV=平均selected/total pages; 保留chunks=TSA的selected/total chunks。")
print("注: 不同方法的任务覆盖不同(812 / 0-199 / 0-99),见任务数列;100-task跑的是task 0-99(map偏多)。")
