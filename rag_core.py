"""
rag_core.py

Pure retrieval-augmented-generation logic for the ExFuzzy rule explainer.
No Streamlit, no input() loop, no client construction at import time -
this module is imported by streamlit_app.py, which reruns on every user
interaction, so anything expensive (loading JSON, building the Ollama
client) is left to the caller to cache (see streamlit_app.py's
@st.cache_resource).

This is a direct refactor of ollamaRuleExplainer.py - the retrieval,
prompt-construction, and tokenisation logic are unchanged (and already
verified per RAG_Verification_and_Streamlit_Guide.md), just reorganised
into importable functions.
"""

import json
import os
import re
from pathlib import Path

from ollama import Client

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
RULES_PATH = "extracted_rules_final.json"
CLOUD_MODEL = "gpt-oss:20b-cloud"
TOP_K = 5

STOPWORDS = {
    "is", "the", "a", "an", "and", "or", "does", "do", "what", "which",
    "who", "how", "for", "of", "in", "on", "to", "tell", "us", "about",
    "that", "this", "with", "are", "was", "were", "be", "it", "its",
}


# ---------------------------------------------------------
# 1. Load rule base
# ---------------------------------------------------------
def load_rules(path: str = RULES_PATH) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    if not data.get("rules") or "raw_rules_text" in data["rules"][0]:
        raise ValueError(
            f"No structured rules found in {path} - "
            "re-run MainFold.py to regenerate it."
        )
    return data


# ---------------------------------------------------------
# 2. Load API key (with fallback to local files)
# ---------------------------------------------------------
def load_api_key_from_local_file() -> str:
    """Try to load OLLAMA_API_KEY from common local file locations."""
    candidates = [
        Path(__file__).resolve().parent / ".streamlit" / "secrets.toml",
        Path(__file__).resolve().parent / "secrets.toml",
        Path(__file__).resolve().parent / "streamlitlol" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ]

    for path in candidates:
        if not path.exists():
            continue
        try:
            import tomllib
        except ImportError:
            import tomli as tomllib

        try:
            with path.open("rb") as fh:
                data = tomllib.load(fh)
        except Exception:
            continue

        value = data.get("OLLAMA_API_KEY", "")
        if value:
            return str(value)

    return ""


def get_client() -> Client:
    api_key = os.environ.get("OLLAMA_API_KEY", "")
    if not api_key:
        api_key = load_api_key_from_local_file()
    if not api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY is not set. Set it via environment variable "
            "or Streamlit secrets (see step 4 below)."
        )
    return Client(
        host="https://ollama.com",
        headers={"Authorization": "Bearer " + api_key},
    )


# ---------------------------------------------------------
# 3. Retrieval (keyword overlap + substring fallback)
# ---------------------------------------------------------
def tokenize(text: str) -> set:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def retrieve_rules(query: str, rules: list, feature_names: list, top_k: int = TOP_K) -> list:
    query_tokens = tokenize(query) - STOPWORDS
    feature_tokens = {f.lower() for f in feature_names if f.lower() in query_tokens}

    scored = []
    for rule in rules:
        rule_text = f"{rule.get('antecedent', '')} {rule.get('consequent', '')}"
        rule_tokens = tokenize(rule_text)
        overlap = len(query_tokens & rule_tokens) + 2 * len(feature_tokens & rule_tokens)

        # Substring fallback: "AF8" in a query never exactly equals the
        # stored token "chaf8" (from tmb_s2_chAF8), so without this,
        # channel-specific queries silently fall back to the default
        # accuracy-ranked list. See failure mode #3 in the verification guide.
        for qt in query_tokens:
            if len(qt) >= 3:
                for rt in rule_tokens:
                    if qt in rt or rt in qt:
                        overlap += 1
                        break

        scored.append((overlap, rule))

    scored.sort(key=lambda x: x[0], reverse=True)

    # If nothing matched at all, fall back to the highest-accuracy rules
    # rather than returning nothing.
    if scored[0][0] == 0:
        scored.sort(key=lambda x: x[1].get("accuracy", 0), reverse=True)

    return [rule for _, rule in scored[:top_k]]


# ---------------------------------------------------------
# 4. Prompt construction
# ---------------------------------------------------------
def build_prompt(question: str, retrieved: list, target_name: str,
                  total_rules: int, cv_stats: dict = None) -> list:
    rules_block = "\n".join(
        f"- Rule {r['rule_id']}: IF {r['antecedent']} THEN {target_name} = {r['consequent']} "
        f"(dominance_score={r.get('dominance_score', r.get('confidence', 'n/a'))}, "
        f"accuracy={r.get('accuracy', 'n/a')})"
        for r in retrieved
    )

    cv_block = ""
    if cv_stats:
        cv_block = (
            "\n\nModel-level cross-validation performance (applies to the whole "
            f"model, not any single rule): test accuracy = "
            f"{cv_stats.get('test_accuracy_mean', 'n/a'):.3f} +/- "
            f"{cv_stats.get('test_accuracy_std', 'n/a'):.3f}, test MCC = "
            f"{cv_stats.get('test_mcc_mean', 'n/a'):.3f} +/- "
            f"{cv_stats.get('test_mcc_std', 'n/a'):.3f}, across "
            f"{cv_stats.get('n_folds', 'n/a')} subject-grouped folds. Use this "
            "when asked about overall/general model reliability or accuracy, "
            "as distinct from any single rule's own accuracy figure."
        )

    system_prompt = (
        "You are an assistant that explains fuzzy rule-based classifier output. "
        "You must ONLY use the rules and statistics provided below. Do not invent "
        "rules, statistics, or feature relationships that are not explicitly listed. "
        "Refer to the target classes using EXACTLY the labels given (e.g. '0 back', "
        "'2/3 back') - do not invent alternate names or descriptions for them. "
        "If the provided rules don't answer the question, say so plainly instead "
        "of guessing.\n\n"
        f"You have been given {len(retrieved)} of {total_rules} total rules in "
        "this model, selected as most relevant to the question below. If the "
        "user asks you to list ALL rules or seems to expect the full rule set, "
        f"you MUST explicitly state that you are showing {len(retrieved)} of "
        f"{total_rules} total rules, not the complete set.\n\n"
        f"Rules:\n{rules_block}"
        f"{cv_block}"
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


# ---------------------------------------------------------
# 5. End-to-end ask() - returns (answer_text, retrieved_rules)
#    so the UI can display "sources used" alongside the answer.
# ---------------------------------------------------------
def ask(question: str, data: dict, client: Client, model: str = CLOUD_MODEL) -> tuple:
    retrieved = retrieve_rules(question, data["rules"], data["feature_names"])
    messages = build_prompt(
        question, retrieved, data["target_name"],
        total_rules=len(data["rules"]),
        cv_stats=data.get("cv_performance_estimate"),
    )
    response = client.chat(model=model, messages=messages)
    return response["message"]["content"], retrieved