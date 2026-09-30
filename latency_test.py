"""
latency_test.py (v3)

Automated latency and reliability testing for the rule explainer,
operationalising the system-level metrics in Section 6, Table 2 of the
research proposal:

    - Response latency: under 3 seconds for 95% of queries (P95)
    - Stability: zero critical failures over 100 test interactions

What changed from v2 (why the numbers from v2 are out of date)
--------------------------------------------------------------
v2 gated with the old binary is_in_scope() check, so it never exercised the
general-background tier; it chose questions with random.choice(), which
left the visualisation path with a single trial in 100 (so its latency was
unmeasured); and part of its pool referred to a rule set that no longer
exists. v3:

  * mirrors backend.py exactly: classify_scope() -> refusal / general
    background / grounded answer (with forced-or-model visualisation);
  * STRATIFIES the sample, so every question (and therefore every path)
    is measured about equally often instead of by luck of the draw;
  * records the consistency-check outcome (flagged / repaired) per trial,
    since a repair costs an extra model call;
  * has an HTTP mode (--url) that measures the DEPLOYED API end to end,
    which is what a real user experiences (network + cold start included).

Each trial is tagged with the PATH it took, because paths have very
different latency profiles and a single blended figure hides that:

    out_of_scope  refused with no model call
    general       general-background tier (one model call)
    no_tool_call  grounded answer, plain text (one model call)
    tool_call     grounded answer with a map (forced in code: one model
                  call; model-chosen: two)
    fallback      the model was unavailable; deterministic rule matching

Usage
-----
    python latency_test.py --n 100                      # in-process (needs OLLAMA_API_KEY)
    python latency_test.py --n 100 --url https://YOUR-BACKEND.azurewebsites.net

Output: a per-path summary printed to the terminal and latency_results.csv
(one row per trial) for the dissertation appendix / results.
"""

import argparse
import csv
import json
import math
import random
import statistics
import time
import traceback
import urllib.error
import urllib.request
from collections import defaultdict

LATENCY_TARGET_SEC = 3.0

# ---------------------------------------------------------
# Question pool. Each branch of the real request path is represented, and
# the pool is sampled in rotation (see build_schedule), not at random.
# ---------------------------------------------------------
GROUNDED_TEXT = [
    "What does channel AF7 predict?",
    "What does channel AFp8 tell us?",
    "What's the dominance score and accuracy of the AFp8 low rule?",
    "Summarize the model's overall accuracy.",
    "What is the model's cross-validated test MCC?",
    "Which rules predict 0 back?",
    "Which rules predict 2/3 back?",
    "How many rules does the model have in total?",
    "What is the highest-accuracy rule in the model?",
    "Is the AF7 rule more reliable than the AFp8 rule?",
    "Does the model use channel PPOz anywhere?",
    "Explain rule 4 in plain language.",
    "List every rule you have, exactly as given.",
]
VISUAL = [
    "Where is AF7 located on the head?",
    "where's af7 at",
    "Which channels are near AF7 on the scalp?",
    "What channels surround AFp8?",
    "How are AF7 and C6h related?",
    "How are AFp8, C5h and CCP3 connected in the rules?",
    "Show me the whole channel map",
    "Where is C6h located?",
]
GENERAL = [
    "What is the hemodynamic response function?",
    "What is EEG?",
    "What is functional connectivity in neuroscience research?",
    "explain cognitive load",
    "Explain the n-back task",
    "What is the prefrontal cortex?",
]
OUT_OF_SCOPE = [
    "What's the weather like today?",
    "Can you write me a poem about the ocean?",
    "What is the capital of France?",
]
QUERY_POOL = GROUNDED_TEXT + VISUAL + GENERAL + OUT_OF_SCOPE


def build_schedule(n: int, seed: int) -> list:
    """n questions in which every pool question appears floor(n/len) or
    ceil(n/len) times, shuffled - so no path is starved by chance."""
    rng = random.Random(seed)
    schedule = []
    while len(schedule) < n:
        block = QUERY_POOL[:]
        rng.shuffle(block)
        schedule.extend(block)
    return schedule[:n]


# ---------------------------------------------------------
# One trial, two ways: in-process (calls rag_core directly, exactly as
# backend.py does) or over HTTP (calls the deployed /ask endpoint).
# ---------------------------------------------------------
def make_inprocess_runner():
    from rag_core import (
        ask_general_neuro, ask_with_visualization, build_fallback_answer,
        classify_scope, get_client, load_brain_mapping, load_domain_background,
        load_literature, load_rules, retrieve_rules, RULES_PATH,
    )
    from viz_tools import load_channel_atlas, plan_viz_call

    data = load_rules(RULES_PATH)
    background, mapping = load_domain_background(), load_brain_mapping()
    atlas, literature = load_channel_atlas(), load_literature()
    client = get_client()

    def run(question: str) -> dict:
        tier = classify_scope(question, data["rules"], data["feature_names"])
        out = {"scope_tier": tier, "path": "", "viz": "none", "routing": "none",
               "answer": "", "flagged": 0, "repaired": False, "citations": 0}
        if tier == "out_of_scope":
            out["path"] = "out_of_scope"
            out["answer"] = "refused"
            return out
        if tier == "general":
            result = ask_general_neuro(question, client, domain_background=background, literature=literature)
            out.update(path="general", answer=result["answer"] or "",
                       citations=len(result.get("citations") or []))
            return out
        try:
            retrieved = retrieve_rules(question, data["rules"], data["feature_names"])
            forced = plan_viz_call(question, atlas, retrieved) is not None
            result = ask_with_visualization(question, data, client, domain_background=background,
                                            brain_mapping=mapping, channel_atlas=atlas)
            viz = (result.get("visualization") or {}).get("type", "none")
            cons = result.get("consistency") or {}
            out.update(path="tool_call" if viz != "none" else "no_tool_call", viz=viz,
                       routing=("forced" if forced else "model") if viz != "none" else "none",
                       answer=result.get("answer") or "",
                       flagged=len(cons.get("issues") or []) + (1 if cons.get("repaired") else 0),
                       repaired=bool(cons.get("repaired")))
        except Exception:
            retrieved = retrieve_rules(question, data["rules"], data["feature_names"])
            out.update(path="fallback", answer=build_fallback_answer(retrieved, data["target_name"]))
        return out

    return run


def make_http_runner(base_url: str):
    endpoint = base_url.rstrip("/")
    if not endpoint.endswith("/ask"):
        endpoint += "/ask"

    def run(question: str) -> dict:
        req = urllib.request.Request(
            endpoint, data=json.dumps({"question": question}).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        with urllib.request.urlopen(req, timeout=180) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        tier = body.get("scope_tier", "grounded")
        viz = (body.get("visualization") or {}).get("type", "none")
        if tier == "out_of_scope":
            path = "out_of_scope"
        elif tier == "general":
            path = "general"
        elif body.get("used_fallback"):
            path = "fallback"
        else:
            path = "tool_call" if viz != "none" else "no_tool_call"
        repaired = bool(body.get("consistency_repaired"))
        flagged = (1 if body.get("consistency_warning") else 0) + (1 if repaired else 0)
        return {"scope_tier": tier, "path": path, "viz": viz, "routing": "n/a",
                "answer": body.get("answer") or "", "flagged": flagged, "repaired": repaired,
                "citations": len(body.get("citations") or [])}

    return run


def run_trials(runner, schedule: list, warmup: bool) -> list:
    if warmup:
        # The first request after idle can be very slow (App Service cold
        # start); it is not representative, so it is made and discarded.
        try:
            print("warm-up request (not counted)...")
            runner("What is the model's cross-validated test MCC?")
        except Exception as e:
            print(f"warm-up failed ({type(e).__name__}: {e}); continuing")

    results = []
    n = len(schedule)
    for i, question in enumerate(schedule, 1):
        start = time.perf_counter()
        error, info = None, {"scope_tier": "", "path": "error", "viz": "none", "routing": "none",
                             "answer": "", "flagged": 0, "repaired": False, "citations": 0}
        try:
            info = runner(question)
        except (urllib.error.URLError, TimeoutError, OSError, Exception) as e:  # noqa: B014
            error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        elapsed = time.perf_counter() - start
        results.append({
            "trial": i, "question": question, "path": info["path"], "scope_tier": info["scope_tier"],
            "latency_sec": round(elapsed, 3), "success": error is None, "error": error or "",
            "answer_length_chars": len(info["answer"]), "visualization_type": info["viz"],
            "viz_routing": info["routing"], "consistency_flagged": info["flagged"],
            "consistency_repaired": info["repaired"], "citations": info["citations"],
        })
        print(f"[{i}/{n}] {elapsed:6.2f}s  {'OK  ' if error is None else 'FAIL'}  [{info['path']}]  {question[:52]}")
    return results


# ---------------------------------------------------------
# Reporting
# ---------------------------------------------------------
def p95(values):
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def print_stats(latencies):
    if not latencies:
        print("  (no successful trials in this group)")
        return
    under = 100 * sum(1 for x in latencies if x < LATENCY_TARGET_SEC) / len(latencies)
    print(f"  mean {statistics.mean(latencies):.2f}s | median {statistics.median(latencies):.2f}s | "
          f"P95 {p95(latencies):.2f}s | max {max(latencies):.2f}s | under {LATENCY_TARGET_SEC:.0f}s: {under:.0f}%")


def summarise(results):
    ok = [r for r in results if r["success"]]
    print("\n================ SUMMARY (all paths combined) ================")
    print(f"Trials: {len(results)} | failures: {len(results) - len(ok)} "
          f"(target: 0 over 100) | success rate: {100 * len(ok) / max(1, len(results)):.1f}%")
    print_stats([r["latency_sec"] for r in ok])
    print(f"Proposal target: P95 < {LATENCY_TARGET_SEC:.0f}s  ->  "
          f"{'MET' if ok and p95([r['latency_sec'] for r in ok]) < LATENCY_TARGET_SEC else 'NOT MET'} (blended)")

    print("\n================ BY PATH ================")
    by_path = defaultdict(list)
    for r in ok:
        by_path[r["path"]].append(r["latency_sec"])
    for path, lats in sorted(by_path.items()):
        print(f"{path} (n={len(lats)})")
        print_stats(lats)

    forced = [r["latency_sec"] for r in ok if r["viz_routing"] == "forced"]
    model = [r["latency_sec"] for r in ok if r["viz_routing"] == "model"]
    if forced or model:
        print("\n================ VISUALISATION ROUTING ================")
        for label, lats in (("forced in code", forced), ("chosen by the model", model)):
            print(f"{label} (n={len(lats)})")
            print_stats(lats)

    flagged = [r for r in results if r["consistency_flagged"]]
    repaired = sum(1 for r in results if r["consistency_repaired"])
    grounded = sum(1 for r in results if r["path"] in ("tool_call", "no_tool_call"))
    print("\n================ CONSISTENCY CHECK ================")
    print(f"Grounded answers: {grounded} | flagged by the check: {len(flagged)} | auto-repaired: {repaired}")
    print("\nReport latency BY PATH in the write-up: a refusal costs no model call and a "
          "model-chosen map costs two, so one blended number misleads in both directions.")


def write_csv(results, path):
    fields = ["trial", "question", "path", "scope_tier", "latency_sec", "success", "error",
              "answer_length_chars", "visualization_type", "viz_routing",
              "consistency_flagged", "consistency_repaired", "citations"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {len(results)} rows to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=100, help="number of trials (default 100)")
    parser.add_argument("--seed", type=int, default=42, help="seed for the question order")
    parser.add_argument("--url", type=str, default="", help="test the DEPLOYED API at this base URL instead of in-process")
    parser.add_argument("--no-warmup", action="store_true", help="skip the discarded warm-up request")
    parser.add_argument("--out", type=str, default="latency_results.csv")
    args = parser.parse_args()

    runner = make_http_runner(args.url) if args.url else make_inprocess_runner()
    schedule = build_schedule(args.n, args.seed)
    results = run_trials(runner, schedule, warmup=not args.no_warmup and bool(args.url))
    summarise(results)
    write_csv(results, args.out)