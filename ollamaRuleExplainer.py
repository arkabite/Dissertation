# # """
# # Stage 4: Ollama Cloud RAG over ExFuzzy rules.

# # Loads extracted_rules.json (produced by main.py), retrieves the rules
# # most relevant to a user's question, and asks a cloud-hosted Ollama model
# # to explain them WITHOUT inventing anything not present in the rule base.

# # Setup:
# #     pip install ollama
# #     export OLLAMA_API_KEY="your-key-from-ollama.com/settings/keys"

# # Run:
# #     python ollama_rule_explainer.py
# # """

# # import json
# # import os
# # import re
# # from ollama import Client 
# # from dotenv import load_dotenv
# # load_dotenv()

# # # ---------------------------------------------------------
# # # Config
# # # ---------------------------------------------------------
# # RULES_PATH = "extracted_rules.json"
# # CLOUD_MODEL = "gpt-oss:20b-cloud"   # level 1/2 model -> light on free-tier quota
# # TOP_K = 5                           # how many rules to inject per question

# # client = Client(
# #     host="https://ollama.com",
# #     headers={"Authorization": "Bearer " + os.environ.get("OLLAMA_API_KEY", "")},
# # )


# # # ---------------------------------------------------------
# # # 1. Load rule base
# # # ---------------------------------------------------------
# # def load_rules(path: str) -> dict:
# #     with open(path, "r") as f:
# #         data = json.load(f)
# #     if not data.get("rules") or "raw_rules_text" in data["rules"][0]:
# #         raise ValueError(
# #             "No structured rules found in extracted_rules.json — "
# #             "re-run main.py to regenerate it."
# #         )
# #     return data


# # # ---------------------------------------------------------
# # # 2. Retrieval (keyword overlap — no vector DB needed for
# # #    a rule base this small; swap for embeddings later if
# # #    nRules grows into the hundreds)
# # # ---------------------------------------------------------
# # def tokenize(text: str) -> set:
# #     return set(re.findall(r"[a-zA-Z]+", text.lower()))


# # def retrieve_rules(query: str, rules: list, feature_names: list, top_k: int = TOP_K) -> list:
# #     query_tokens = tokenize(query)
# #     # let feature names count double so "sex" or "fare" strongly pulls in
# #     # rules that mention them, even in short queries
# #     feature_tokens = {f.lower() for f in feature_names if f.lower() in query_tokens}

# #     scored = []
# #     for rule in rules:
# #         rule_text = f"{rule.get('antecedent', '')} {rule.get('consequent', '')}"
# #         rule_tokens = tokenize(rule_text)
# #         overlap = len(query_tokens & rule_tokens) + 2 * len(feature_tokens & rule_tokens)
# #         scored.append((overlap, rule))

# #     scored.sort(key=lambda x: x[0], reverse=True)

# #     # if nothing matched at all, fall back to the highest-accuracy rules
# #     # rather than returning nothing
# #     if scored[0][0] == 0:
# #         scored.sort(key=lambda x: x[1].get("accuracy", 0), reverse=True)

# #     return [rule for _, rule in scored[:top_k]]


# # # ---------------------------------------------------------
# # # 3. Prompt construction (grounding is the whole point here)
# # # ---------------------------------------------------------
# # def build_prompt(question: str, retrieved: list, target_name: str) -> list:
# #     rules_block = "\n".join(
# #         f"- Rule {r['rule_id']}: IF {r['antecedent']} THEN {target_name} = {r['consequent']} "
# #         f"(dominance_score={r.get('dominance_score', r.get('confidence', 'n/a'))}, "
# #         f"accuracy={r.get('accuracy', 'n/a')})"
# #         for r in retrieved
# #     )

# #     system_prompt = (
# #         "You are an assistant that explains fuzzy rule-based classifier output. "
# #         "You must ONLY use the rules provided below. Do not invent rules, "
# #         "statistics, or feature relationships that are not explicitly listed. "
# #         "If the provided rules don't answer the question, say so plainly instead "
# #         "of guessing.\n\n"
# #         f"Rules:\n{rules_block}"
# #     )

# #     return [
# #         {"role": "system", "content": system_prompt},
# #         {"role": "user", "content": question},
# #     ]


# # # ---------------------------------------------------------
# # # 4. Query the cloud model
# # # ---------------------------------------------------------
# # def ask(question: str, data: dict) -> str:
# #     retrieved = retrieve_rules(question, data["rules"], data["feature_names"])
# #     messages = build_prompt(question, retrieved, data["target_name"])

# #     response = client.chat(model=CLOUD_MODEL, messages=messages)
# #     return response["message"]["content"]


# # # ---------------------------------------------------------
# # # 5. Simple chat loop
# # # ---------------------------------------------------------
# # def main():
# #     if not os.environ.get("OLLAMA_API_KEY"):
# #         raise SystemExit(
# #             "Set OLLAMA_API_KEY first: export OLLAMA_API_KEY=your_key\n"
# #             "(get one at https://ollama.com/settings/keys)"
# #         )

# #     data = load_rules(RULES_PATH)
# #     print(f"Loaded {len(data['rules'])} rules over target '{data['target_name']}'.")
# #     print("Ask a question about the model's rules (Ctrl+C to quit).\n")

# #     while True:
# #         try:
# #             question = input("You: ").strip()
# #         except (KeyboardInterrupt, EOFError):
# #             print("\nExiting.")
# #             break

# #         if not question:
# #             continue

# #         answer = ask(question, data)
# #         print(f"\nAssistant: {answer}\n")


# # if __name__ == "__main__":
# #     main()

# """
# Stage 4: Ollama Cloud RAG over ExFuzzy rules.

# Loads extracted_rules.json (produced by main.py), retrieves the rules
# most relevant to a user's question, and asks a cloud-hosted Ollama model
# to explain them WITHOUT inventing anything not present in the rule base.

# Setup:
#     pip install ollama
#     export OLLAMA_API_KEY="your-key-from-ollama.com/settings/keys"

# Run:
#     python ollama_rule_explainer.py
# """

# import json
# import os
# import re
# from ollama import Client
# from dotenv import load_dotenv
# load_dotenv()

# # ---------------------------------------------------------
# # Config
# # ---------------------------------------------------------
# RULES_PATH = "extracted_rules.json"
# CLOUD_MODEL = "gpt-oss:20b-cloud"   # level 1/2 model -> light on free-tier quota
# TOP_K = 5                           # how many rules to inject per question

# client = Client(
#     host="https://ollama.com",
#     headers={"Authorization": "Bearer " + os.environ.get("OLLAMA_API_KEY", "")},
# )


# # ---------------------------------------------------------
# # 1. Load rule base
# # ---------------------------------------------------------
# def load_rules(path: str) -> dict:
#     with open(path, "r") as f:
#         data = json.load(f)
#     if not data.get("rules") or "raw_rules_text" in data["rules"][0]:
#         raise ValueError(
#             "No structured rules found in extracted_rules.json — "
#             "re-run main.py to regenerate it."
#         )
#     return data


# # ---------------------------------------------------------
# # 2. Retrieval (keyword overlap — no vector DB needed for
# #    a rule base this small; swap for embeddings later if
# #    nRules grows into the hundreds)
# # ---------------------------------------------------------
# def tokenize(text: str) -> set:
#     return set(re.findall(r"[a-zA-Z]+", text.lower()))


# def retrieve_rules(query: str, rules: list, feature_names: list, top_k: int = TOP_K) -> list:
#     query_tokens = tokenize(query)
#     # let feature names count double so "sex" or "fare" strongly pulls in
#     # rules that mention them, even in short queries
#     feature_tokens = {f.lower() for f in feature_names if f.lower() in query_tokens}

#     scored = []
#     for rule in rules:
#         rule_text = f"{rule.get('antecedent', '')} {rule.get('consequent', '')}"
#         rule_tokens = tokenize(rule_text)
#         overlap = len(query_tokens & rule_tokens) + 2 * len(feature_tokens & rule_tokens)
#         scored.append((overlap, rule))

#     scored.sort(key=lambda x: x[0], reverse=True)

#     # if nothing matched at all, fall back to the highest-accuracy rules
#     # rather than returning nothing
#     if scored[0][0] == 0:
#         scored.sort(key=lambda x: x[1].get("accuracy", 0), reverse=True)

#     return [rule for _, rule in scored[:top_k]]


# # ---------------------------------------------------------
# # 3. Prompt construction (grounding is the whole point here)
# # ---------------------------------------------------------
# def build_prompt(question: str, retrieved: list, target_name: str) -> list:
#     rules_block = "\n".join(
#         f"- Rule {r['rule_id']}: IF {r['antecedent']} THEN {target_name} = {r['consequent']} "
#         f"(dominance_score={r.get('dominance_score', r.get('confidence', 'n/a'))}, "
#         f"accuracy={r.get('accuracy', 'n/a')})"
#         for r in retrieved
#     )

#     system_prompt = (
#         "You are an assistant that explains fuzzy rule-based classifier output. "
#         "You must ONLY use the rules provided below. Do not invent rules, "
#         "statistics, or feature relationships that are not explicitly listed. "
#         "If the provided rules don't answer the question, say so plainly instead "
#         "of guessing.\n\n"
#         f"Rules:\n{rules_block}"
#     )

#     return [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": question},
#     ]


# # ---------------------------------------------------------
# # 4. Query the cloud model
# # ---------------------------------------------------------
# def ask(question: str, data: dict) -> str:
#     retrieved = retrieve_rules(question, data["rules"], data["feature_names"])
#     print(f"[debug] retrieved {len(retrieved)} rule(s): "
#           f"{[r['rule_id'] for r in retrieved]}")
#     messages = build_prompt(question, retrieved, data["target_name"])

#     response = client.chat(model=CLOUD_MODEL, messages=messages)
#     return response["message"]["content"]


# # ---------------------------------------------------------
# # 5. Simple chat loop
# # ---------------------------------------------------------
# def main():
#     if not os.environ.get("OLLAMA_API_KEY"):
#         raise SystemExit(
#             "Set OLLAMA_API_KEY first: export OLLAMA_API_KEY=your_key\n"
#             "(get one at https://ollama.com/settings/keys)"
#         )

#     data = load_rules(RULES_PATH)
#     print(f"Loaded {len(data['rules'])} rules over target '{data['target_name']}'.")
#     print("Ask a question about the model's rules (Ctrl+C to quit).\n")

#     while True:
#         try:
#             question = input("You: ").strip()
#         except (KeyboardInterrupt, EOFError):
#             print("\nExiting.")
#             break

#         if not question:
#             continue

#         answer = ask(question, data)
#         print(f"\nAssistant: {answer}\n")


# if __name__ == "__main__":
#     main()

"""
Stage 4: Ollama Cloud RAG over ExFuzzy rules.

Loads extracted_rules.json (produced by main.py), retrieves the rules
most relevant to a user's question, and asks a cloud-hosted Ollama model
to explain them WITHOUT inventing anything not present in the rule base.

Setup:
    pip install ollama
    export OLLAMA_API_KEY="your-key-from-ollama.com/settings/keys"

Run:
    python ollama_rule_explainer.py
"""

import json
import os
import re
from ollama import Client
from dotenv import load_dotenv
load_dotenv()

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
RULES_PATH = "extracted_rules_final.json"
CLOUD_MODEL = "gpt-oss:20b-cloud"   # level 1/2 model -> light on free-tier quota
TOP_K = 5                           # how many rules to inject per question

client = Client(
    host="https://ollama.com",
    headers={"Authorization": "Bearer " + os.environ.get("OLLAMA_API_KEY", "")},
)


# ---------------------------------------------------------
# 1. Load rule base
# ---------------------------------------------------------
def load_rules(path: str) -> dict:
    with open(path, "r") as f:
        data = json.load(f)
    if not data.get("rules") or "raw_rules_text" in data["rules"][0]:
        raise ValueError(
            "No structured rules found in extracted_rules.json — "
            "re-run main.py to regenerate it."
        )
    return data


# ---------------------------------------------------------
# 2. Retrieval (keyword overlap — no vector DB needed for
#    a rule base this small; swap for embeddings later if
#    nRules grows into the hundreds)
# ---------------------------------------------------------
STOPWORDS = {
    "is", "the", "a", "an", "and", "or", "does", "do", "what", "which",
    "who", "how", "for", "of", "in", "on", "to", "tell", "us", "about",
    "that", "this", "with", "are", "was", "were", "be", "it", "its",
}


def tokenize(text: str) -> set:
    return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


def retrieve_rules(query: str, rules: list, feature_names: list, top_k: int = TOP_K) -> list:
    query_tokens = tokenize(query) - STOPWORDS
    # let feature names count double so "sex" or "fare" strongly pulls in
    # rules that mention them, even in short queries
    feature_tokens = {f.lower() for f in feature_names if f.lower() in query_tokens}

    scored = []
    for rule in rules:
        rule_text = f"{rule.get('antecedent', '')} {rule.get('consequent', '')}"
        rule_tokens = tokenize(rule_text)
        overlap = len(query_tokens & rule_tokens) + 2 * len(feature_tokens & rule_tokens)

        # Substring fallback: users naturally say "AF8" or "af8", not the
        # underlying token "chaf8" (channel names are stored as tmb_s2_chAF8,
        # which tokenizes to tmb/s2/chaf8 once underscores are split off).
        # Without this, a query token like "af8" never equals "chaf8" via set
        # intersection, so channel-specific questions silently fall back to
        # "top rules by accuracy" instead of actually retrieving the right rule.
        for qt in query_tokens:
            if len(qt) >= 3:
                for rt in rule_tokens:
                    if qt in rt or rt in qt:
                        overlap += 1
                        break

        scored.append((overlap, rule))

    scored.sort(key=lambda x: x[0], reverse=True)

    # if nothing matched at all, fall back to the highest-accuracy rules
    # rather than returning nothing
    if scored[0][0] == 0:
        scored.sort(key=lambda x: x[1].get("accuracy", 0), reverse=True)

    return [rule for _, rule in scored[:top_k]]


# ---------------------------------------------------------
# 3. Prompt construction (grounding is the whole point here)
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
        "'2/3 back') — do not invent alternate names or descriptions for them. "
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
# 4. Query the cloud model
# ---------------------------------------------------------
def ask(question: str, data: dict) -> str:
    retrieved = retrieve_rules(question, data["rules"], data["feature_names"])
    print(f"[debug] retrieved {len(retrieved)} rule(s): "
          f"{[r['rule_id'] for r in retrieved]}")
    messages = build_prompt(
        question, retrieved, data["target_name"],
        total_rules=len(data["rules"]),
        cv_stats=data.get("cv_performance_estimate"),
    )

    response = client.chat(model=CLOUD_MODEL, messages=messages)
    return response["message"]["content"]


# ---------------------------------------------------------
# 5. Simple chat loop
# ---------------------------------------------------------
def main():
    if not os.environ.get("OLLAMA_API_KEY"):
        raise SystemExit(
            "Set OLLAMA_API_KEY first: export OLLAMA_API_KEY=your_key\n"
            "(get one at https://ollama.com/settings/keys)"
        )

    data = load_rules(RULES_PATH)
    print(f"Loaded {len(data['rules'])} rules over target '{data['target_name']}'.")
    print("Ask a question about the model's rules (Ctrl+C to quit).\n")

    while True:
        try:
            question = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not question:
            continue

        answer = ask(question, data)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    main()