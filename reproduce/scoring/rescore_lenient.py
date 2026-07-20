"""Re-score failed tasks leniently: use the judge to credit answers that CONTAIN the correct
reference value but failed strict exact_match due to verbose phrasing / formatting (false negatives),
while keeping genuinely wrong answers as fails. Only re-judges tasks the official scorer failed.
"""
import json, glob, re, sys, os, requests

JUDGE_URL = "http://127.0.0.1:18000/v1/chat/completions"
JUDGE_MODEL = "llama-judge"
RESDIR = sys.argv[1] if len(sys.argv) > 1 else "results/llama_dense"

def judge(question, reference, pred):
    msg = ("You are grading a web-agent's free-form answer against a reference answer. "
           "The agent often wraps the answer in a full sentence or adds extra words — that is FINE. "
           "Mark CORRECT if the agent's answer clearly STATES the reference answer value "
           "(ignoring surrounding words, capitalization, punctuation, and symbols like the trademark sign). "
           "Mark INCORRECT if the agent gives a wrong value, misses required items, is vague/uncertain, "
           "says it could not find it, or lists multiple guesses without committing to the right one.\n\n"
           f"Question: {question}\nReference answer: {reference}\nAgent answer: {pred}\n\n"
           "Reply with exactly one word: CORRECT or INCORRECT.")
    try:
        r = requests.post(JUDGE_URL, json={"model": JUDGE_MODEL,
              "messages": [{"role": "system", "content": "You are a strict but fair grader."},
                           {"role": "user", "content": msg}],
              "temperature": 0, "max_tokens": 5}, timeout=60)
        out = r.json()["choices"][0]["message"]["content"].strip().upper()
        return out.startswith("CORRECT") or ("CORRECT" in out and "INCORRECT" not in out)
    except Exception as e:
        return None

scores = json.load(open(os.path.join(RESDIR,"SCORES.json")))
official_pass = scores["n_pass"]
adj_pass_ids, still_fail, skipped = [], [], []
for t in scores["tasks"]:
    tid = t["task_id"]
    if t["score"] == 1.0:
        continue  # already pass
    inp = json.load(open(os.path.join(RESDIR,f"task_{tid}/input.json")))
    ev = inp.get("eval", {}); ref = ev.get("reference_answers") or {}
    intent = inp.get("intent", inp.get("confirmed_task", ""))
    try:
        ans = json.load(open(os.path.join(RESDIR,f"task_{tid}/task_{tid}.json"))).get("answer") or ""
    except Exception:
        ans = ""
    # build a reference string for string_match tasks
    refval = ref.get("exact_match") or ref.get("must_include") or ref.get("fuzzy_match")
    if refval is None or "url_match" in t["eval_types"] or "program_html" in t["eval_types"]:
        skipped.append(tid); continue
    refstr = " AND ".join(refval) if isinstance(refval, list) else str(refval)
    verdict = judge(intent, refstr, ans)
    if verdict:
        adj_pass_ids.append(tid)
    else:
        still_fail.append(tid)

n = scores["n"]
adj_total = official_pass + len(adj_pass_ids)
print(f"official pass: {official_pass}/{n} = {official_pass/n:.2%}")
print(f"lenient recovered (formatting false-negatives): {len(adj_pass_ids)} -> task ids {sorted(adj_pass_ids)}")
print(f"still failed (genuinely wrong): {len(still_fail)}")
print(f"skipped (url_match/program_html need navigation, not re-judged): {len(skipped)} -> {sorted(skipped)}")
print(f"==> adjusted (lenient) pass: {adj_total}/{n} = {adj_total/n:.2%}")
# per-site adjusted
sites = {}
for t in scores["tasks"]:
    tid = t["task_id"]
    inp = json.load(open(os.path.join(RESDIR,f"task_{tid}/input.json")))
    site = "+".join(inp.get("sites", ["?"]))
    p = 1 if (t["score"] == 1.0 or tid in adj_pass_ids) else 0
    s = sites.setdefault(site, [0, 0]); s[0] += p; s[1] += 1
print("per-site (lenient):")
for site, (p, tot) in sorted(sites.items()):
    print(f"  {site}: {p}/{tot} = {p/tot:.1%}")
json.dump({"official_pass": official_pass, "adjusted_pass": adj_total, "n": n,
           "adjusted_rate": adj_total/n, "format_fixed_ids": sorted(adj_pass_ids),
           "still_fail_ids": sorted(still_fail), "skipped_ids": sorted(skipped)},
          open(os.path.join(RESDIR,"SCORES_adjusted.json"), "w"), indent=1)
