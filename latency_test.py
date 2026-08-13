"""
latency_test.py

Automated latency and stability testing for the Ollama-backed rule
explainer, operationalising the system-level metrics in Section 6,
Table 2 of the research proposal:

    - Backend Latency: API response time < 500 ms
    - Response Latency: response time < 3 seconds for 95% of queries
    - Stability: zero critical failures over 100 test interactions

This does NOT require human participants - it's an automated script you
run against your own rag_core.py, so it's something you can do entirely
on your own right now.

Usage:
    python latency_test.py --n 100

Output:
    - Prints a summary (mean/median/p95 latency, failure count)
    - Writes latency_results.csv with one row per query for your
      dissertation appendix / results chapter
"""

import argparse
import csv
import random
import statistics
import time
import traceback
from pathlib import Path

from rag_core import load_rules, get_client, ask, RULES_PATH

# ---------------------------------------------------------
# Query pool - mix of question types so the test isn't measuring
# latency for only one kind of query. Extend this list freely; more
# variety here gives a more representative stability/latency estimate.
# ---------------------------------------------------------
QUERY_POOL = [
    "What does channel AF8 predict?",
    "What does channel AF7 predict?",
    "What's the dominance score and accuracy for the AFF5/AFpz rule?",
    "Summarize the model's overall accuracy.",
    "List every rule you have, exactly as given.",
    "What does channel AFp8 tell us?",
    "Which hemisphere shows more activity overall?",
    "What is the model's cross-validated test MCC?",
    "Which rules predict 0 back?",
    "Which rules predict 2/3 back?",
    "How many rules does the model have in total?",
    "What is the highest-accuracy rule in the model?",
    "Is the AF7 rule more reliable than the AF8 rule?",
    "Does the model use channel PPOz anywhere?",
    "Explain rule 4 in plain language.",
]

LATENCY_TARGET_SEC = 3.0
BACKEND_TARGET_SEC = 0.5  # measured separately if you have a backend hop; see note below


def run_trials(n: int, seed: int = 42):
    data = load_rules(RULES_PATH)
    client = get_client()

    rng = random.Random(seed)
    results = []

    for i in range(1, n + 1):
        question = rng.choice(QUERY_POOL)
        start = time.perf_counter()
        error = None
        answer_len = 0
        try:
            answer, retrieved = ask(question, data, client)
            answer_len = len(answer)
        except Exception as e:
            error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        elapsed = time.perf_counter() - start

        results.append({
            "trial": i,
            "question": question,
            "latency_sec": round(elapsed, 3),
            "success": error is None,
            "error": error or "",
            "answer_length_chars": answer_len,
        })

        status = "OK" if error is None else "FAIL"
        print(f"[{i}/{n}] {elapsed:.2f}s  {status}  {question[:50]}")

    return results


def summarise(results):
    latencies = [r["latency_sec"] for r in results if r["success"]]
    n_total = len(results)
    n_fail = sum(1 for r in results if not r["success"])

    print("\n================ SUMMARY ================")
    print(f"Total queries: {n_total}")
    print(f"Failures: {n_fail}  (target: 0 over 100, per Section 6, Table 2)")

    if latencies:
        latencies_sorted = sorted(latencies)
        p95_idx = int(0.95 * len(latencies_sorted)) - 1
        p95_idx = max(0, min(p95_idx, len(latencies_sorted) - 1))
        p95 = latencies_sorted[p95_idx]
        under_target = sum(1 for l in latencies if l < LATENCY_TARGET_SEC)
        pct_under_target = 100 * under_target / len(latencies)

        print(f"Mean latency: {statistics.mean(latencies):.2f}s")
        print(f"Median latency: {statistics.median(latencies):.2f}s")
        print(f"P95 latency: {p95:.2f}s")
        print(f"Min / Max: {min(latencies):.2f}s / {max(latencies):.2f}s")
        print(f"% of queries under {LATENCY_TARGET_SEC:.0f}s target: {pct_under_target:.1f}% "
              f"(target: >= 95%, per Section 6, Table 2)")
    else:
        print("No successful queries - cannot compute latency statistics.")

    print(
        "\nNote: the proposal's 'Backend Latency < 500ms' criterion (Section 6, Table 2) "
        "refers to a RESTful API hop that this prototype does not have (Streamlit calls "
        "rag_core.ask() directly). Report the end-to-end latency above instead, and note "
        "in your limitations section that the architecture was simplified relative to the "
        "proposed three-tier design (see your write-up on the simplified NLP layer)."
    )


def write_csv(results, path="latency_results.csv"):
    fieldnames = ["trial", "question", "latency_sec", "success", "error", "answer_length_chars"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWrote {len(results)} rows to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=100,
                         help="Number of test queries to run (default 100, matching the "
                              "proposal's 'zero critical failures over 100 test interactions').")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for query selection.")
    parser.add_argument("--out", type=str, default="latency_results.csv")
    args = parser.parse_args()

    results = run_trials(args.n, seed=args.seed)
    summarise(results)
    write_csv(results, args.out)
