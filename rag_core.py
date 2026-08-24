# # """
# # rag_core.py

# # Pure retrieval-augmented-generation logic for the ExFuzzy rule explainer.
# # No Streamlit, no input() loop, no client construction at import time -
# # this module is imported by streamlit_app.py, which reruns on every user
# # interaction, so anything expensive (loading JSON, building the Ollama
# # client) is left to the caller to cache (see streamlit_app.py's
# # @st.cache_resource).

# # Retrieval and tokenisation logic match the originally-verified
# # ollamaRuleExplainer.py (see RAG_Verification_and_Streamlit_Guide.md).
# # Prompt construction has since been extended twice, each requiring its own
# # re-verification pass:
# #   - added cross-seed rule stability caveats (n_seeds / found-in-X/N-refits)
# #   - added a two-tier channel glossary (brain_mapping.json) for plain-English
# #     location/signal labels. Tier 1 (location + HbO/HbR direction) is sourced
# #     from this project's own MainFold.py code and standard 10-10 fNIRS
# #     naming; Tier 2 (cognitive-function associations) is general literature
# #     background and must always be presented hedged, never as something this
# #     specific model's rules demonstrated. See brain_mapping.json's "_notes"
# #     field for the sourcing rationale.
# # """

# # import json
# # import os
# # import re
# # from pathlib import Path

# # from ollama import Client

# # # ---------------------------------------------------------
# # # Config
# # # ---------------------------------------------------------
# # RULES_PATH = "extracted_rules_final_raw.json"
# # CLOUD_MODEL = "gpt-oss:20b-cloud"
# # TOP_K = 5

# # STOPWORDS = {
# #     "is", "the", "a", "an", "and", "or", "does", "do", "what", "which",
# #     "who", "how", "for", "of", "in", "on", "to", "tell", "us", "about",
# #     "that", "this", "with", "are", "was", "were", "be", "it", "its",
# # }


# # # ---------------------------------------------------------
# # # 1. Load rule base & Domain Knowledge
# # # ---------------------------------------------------------
# # def load_rules(path: str = RULES_PATH) -> dict:
# #     with open(path, "r") as f:
# #         data = json.load(f)
# #     if not data.get("rules") or "raw_rules_text" in data["rules"][0]:
# #         raise ValueError(
# #             f"No structured rules found in {path} - "
# #             "re-run MainFold.py to regenerate it."
# #         )
# #     return data

# # def load_domain_background(path: str = "domain_background.md") -> str:
# #     """Small, hand-curated background block extracted from the project's
# #     own literature review."""
# #     try:
# #         with open(path, "r") as f:
# #             return f.read().strip()
# #     except FileNotFoundError:
# #         return ""

# # def load_brain_mapping(path: str = "brain_mapping.json") -> dict:
# #     """Loads the neurological dictionary to translate opaque channel names."""
# #     try:
# #         with open(path, "r") as f:
# #             return json.load(f)
# #     except FileNotFoundError:
# #         return {"signals": {}, "regions": {}, "hemispheres": {}}


# # # ---------------------------------------------------------
# # # 2. Load API key (with fallback to local files)
# # # ---------------------------------------------------------
# # def load_api_key_from_local_file() -> str:
# #     """Try to load OLLAMA_API_KEY from common local file locations."""
# #     candidates = [
# #         Path(__file__).resolve().parent / ".streamlit" / "secrets.toml",
# #         Path(__file__).resolve().parent / "secrets.toml",
# #         Path(__file__).resolve().parent / "streamlitlol" / "secrets.toml",
# #         Path.home() / ".streamlit" / "secrets.toml",
# #     ]

# #     for path in candidates:
# #         if not path.exists():
# #             continue
# #         try:
# #             import tomllib
# #         except ImportError:
# #             import tomli as tomllib

# #         try:
# #             with path.open("rb") as fh:
# #                 data = tomllib.load(fh)
# #         except Exception:
# #             continue

# #         value = data.get("OLLAMA_API_KEY", "")
# #         if value:
# #             return str(value)

# #     return ""


# # def get_client() -> Client:
# #     api_key = os.environ.get("OLLAMA_API_KEY", "")
# #     if not api_key:
# #         api_key = load_api_key_from_local_file()
# #     if not api_key:
# #         raise RuntimeError(
# #             "OLLAMA_API_KEY is not set. Set it via environment variable "
# #             "or Streamlit secrets (see step 4 below)."
# #         )
# #     return Client(
# #         host="https://ollama.com",
# #         headers={"Authorization": "Bearer " + api_key},
# #     )


# # # ---------------------------------------------------------
# # # 3. Retrieval (keyword overlap + substring fallback)
# # # ---------------------------------------------------------
# # def tokenize(text: str) -> set:
# #     return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


# # def retrieve_rules(query: str, rules: list, feature_names: list, top_k: int = TOP_K) -> list:
# #     query_tokens = tokenize(query) - STOPWORDS
# #     feature_tokens = {f.lower() for f in feature_names if f.lower() in query_tokens}

# #     scored = []
# #     for rule in rules:
# #         rule_text = f"{rule.get('antecedent', '')} {rule.get('consequent', '')}"
# #         rule_tokens = tokenize(rule_text)
# #         overlap = len(query_tokens & rule_tokens) + 2 * len(feature_tokens & rule_tokens)

# #         # Substring fallback
# #         for qt in query_tokens:
# #             if len(qt) >= 3:
# #                 for rt in rule_tokens:
# #                     if qt in rt or rt in qt:
# #                         overlap += 1
# #                         break

# #         scored.append((overlap, rule))

# #     scored.sort(key=lambda x: x[0], reverse=True)

# #     if scored[0][0] == 0:
# #         scored.sort(key=lambda x: x[1].get("accuracy", 0), reverse=True)

# #     return [rule for _, rule in scored[:top_k]]


# # # ---------------------------------------------------------
# # # 4. Prompt construction
# # # ---------------------------------------------------------
# # def build_stability_lookup(data: dict) -> dict:
# #     return {
# #         (r["antecedent"], r["consequent"]): r["n_seeds_found"]
# #         for r in data.get("rule_stability_across_seeds", [])
# #     }

# # def pre_translate_rule(rule_text: str, brain_map: dict) -> str:
# #     """
# #     Deterministically (no LLM involved) translates opaque channel codes like
# #     'tmb_s1_chAF5h' into a location + signal label the LLM can use directly,
# #     e.g. 'Left AF [Anterior-frontal] (HbO)'. Only SOURCED fields (signal
# #     direction from MainFold.py, location from standard 10-10 naming) go into
# #     this inline label - cognitive-function associations are deliberately
# #     left out here and supplied separately, clearly hedged, in the prompt.
# #     """
# #     def replacer(match):
# #         signal = match.group(1)       # 's1' or 's2'
# #         region_code = match.group(2)  # 'AF', 'C', 'PO', etc.
# #         hemi_code = match.group(3)    # '5h', 'z', '8', etc.

# #         signal_info = brain_map.get("signals", {}).get(signal, {})
# #         signal_name = signal_info.get("label", signal).split(" ")[-1].strip("()")
# #         if not signal_name or signal_name == signal:
# #             signal_name = "HbO" if signal == "s1" else "HbR"

# #         hemi_conv = brain_map.get("hemisphere_convention", {})
# #         if 'z' in hemi_code.lower():
# #             hemi_name = hemi_conv.get("z", "Midline")
# #         elif any(c in hemi_code for c in '13579'):
# #             hemi_name = hemi_conv.get("odd", "Left")
# #         elif any(c in hemi_code for c in '02468'):
# #             hemi_name = hemi_conv.get("even", "Right")
# #         else:
# #             hemi_name = "Unknown-hemisphere"
# #         # Trim the parenthetical caveat for inline brevity; full caveat is
# #         # given once in the glossary block below, not repeated per-channel.
# #         hemi_name = hemi_name.split(" (")[0]

# #         region_info = brain_map.get("regions", {}).get(region_code, {})
# #         region_desc = region_info.get("short", region_code)

# #         return f"[{hemi_name} {region_desc} ({signal_name})]"

# #     pattern = r"tmb_(s1|s2)_ch([a-zA-Z]+)(\d+[a-zA-Z]*|z)"
# #     return re.sub(pattern, replacer, rule_text)


# # def build_prompt(question: str, retrieved: list, target_name: str,
# #                  total_rules: int, cv_stats: dict = None,
# #                  stability_lookup: dict = None, n_seeds_total: int = 1,
# #                  domain_background: str = "", brain_mapping: dict = None) -> list:
# #     stability_lookup = stability_lookup or {}
# #     brain_mapping = brain_mapping or {}

# #     def stability_tag(r):
# #         if n_seeds_total <= 1:
# #             return ""
# #         n_found = stability_lookup.get((r["antecedent"], r["consequent"]), 1)
# #         return f", found in {n_found}/{n_seeds_total} independent model refits"
    
# #     rules_block = "\n".join(
# #         f"- Rule {r['rule_id']}: IF {pre_translate_rule(r['antecedent'], brain_mapping)} "
# #         f"THEN {target_name} = {r['consequent']} "
# #         f"(dominance_score={r.get('dominance_score', r.get('confidence', 'n/a'))}, "
# #         f"accuracy={r.get('accuracy', 'n/a')}{stability_tag(r)})"
# #         for r in retrieved
# #     )

# #     cv_block = ""
# #     if cv_stats:
# #         cv_block = (
# #             "\n\nModel-level cross-validation performance (applies to the whole "
# #             f"model, not any single rule): test accuracy = "
# #             f"{cv_stats.get('test_accuracy_mean', 'n/a'):.3f} +/- "
# #             f"{cv_stats.get('test_accuracy_std', 'n/a'):.3f}, test MCC = "
# #             f"{cv_stats.get('test_mcc_mean', 'n/a'):.3f} +/- "
# #             f"{cv_stats.get('test_mcc_std', 'n/a'):.3f}, across "
# #             f"{cv_stats.get('n_folds', 'n/a')} subject-grouped folds. Use this "
# #             "when asked about overall/general model reliability or accuracy, "
# #             "as distinct from any single rule's own accuracy figure."
# #         )

# #     stability_block = ""
# #     if n_seeds_total > 1:
# #         stability_block = (
# #             f"\n\nIMPORTANT CAVEAT: This rule set is one representative fit "
# #             f"out of {n_seeds_total} independently-trained seeds (the GA "
# #             "fitting procedure is stochastic). The 'found in X/N refits' "
# #             "figure on each rule shows how many of those seeds re-discovered "
# #             "that EXACT rule. For dominance_score specifically: state the "
# #             "number itself and, if asked, that it is the rule's relative "
# #             "contribution to the model's decisions - do NOT characterize "
# #             "what a low or high score 'typically' or 'usually' implies "
# #             "about how much the rule matters beyond that definition; you "
# #             "have one number per rule, not a distribution to generalize "
# #             "from. Do not assert that a pattern 'remains present', 'is "
# #             "still there', or 'is real underneath the wording' when a rule "
# #             "was found in only 1 of N refits - a low recurrence count means "
# #             "you do NOT know whether the same underlying signal reappears "
# #             "with different wording, or a genuinely different signal was "
# #             "found by chance. Say that uncertainty explicitly rather than "
# #             "assuming stability you have no evidence for. This is separate "
# #             "from the model-level cross-validation stats above, which DO "
# #             "describe genuinely stable, ensembled predictive performance."
# #         )

# #     background_block = ""
# #     if domain_background:
# #         background_block = (
# #             "\n\n=== BACKGROUND CONTEXT (general literature, NOT this "
# #             "model's own findings) ===\n"
# #             f"{domain_background}\n"
# #             "=== END BACKGROUND CONTEXT ===\n"
# #         )

# #     mapping_block = ""
# #     if brain_mapping:
# #         # Only include glossary entries for region codes that actually
# #         # appear in the retrieved rules, not the full glossary every turn -
# #         # keeps the prompt tight and avoids diluting context with unused
# #         # channel definitions.
# #         used_region_codes = {
# #             m.group(2)
# #             for r in retrieved
# #             for m in re.finditer(
# #                 r"tmb_(s1|s2)_ch([a-zA-Z]+)(\d+[a-zA-Z]*|z)",
# #                 r.get("antecedent", ""),
# #             )
# #         }

# #         sig = brain_mapping.get("signals", {})
# #         sig_lines = "\n".join(
# #             f"- {k}: {v.get('label', k)} - {v.get('task_load_direction', '')}"
# #             for k, v in sig.items()
# #         )
# #         hemi = brain_mapping.get("hemisphere_convention", {})
# #         hemi_lines = "\n".join(
# #             f"- {k}: {v}" for k, v in hemi.items() if k != "confidence"
# #         )
# #         region_lines = "\n".join(
# #             f"- {k}: LOCATION (10-5 naming) = {v.get('location', k)}. "
# #             f"FUNCTIONAL REGION ({v.get('functional_region_basis', 'n/a')}) "
# #             f"= {v.get('functional_region', 'not determined')}. "
# #             f"GENERAL COGNITIVE ASSOCIATION (hedge required) = "
# #             f"{v.get('general_association', 'none supplied')}"
# #             for k, v in brain_mapping.get("regions", {}).items()
# #             if k in used_region_codes
# #         )
# #         mapping_block = (
# #             "\n\n=== CHANNEL GLOSSARY (three confidence tiers - read carefully) ===\n"
# #             "Each rule below has already been annotated inline with "
# #             "[Hemisphere Location (Signal)] using this glossary, so you don't "
# #             "need to re-derive it - use those inline annotations directly "
# #             "when explaining a rule to the user.\n\n"
# #             "TIER 1 - SOURCED, state confidently:\n"
# #             "Signal direction (from this project's own feature-engineering "
# #             f"code):\n{sig_lines}\n"
# #             f"Hemisphere convention (confirmed for this dataset - its "
# #             f"methods description states the montage follows the "
# #             f"international 10-5 system):\n{hemi_lines}\n"
# #             f"Scalp location and FUNCTIONAL REGION (frontal/motor/parietal/"
# #             f"occipital) for the channels appearing in the rules above, "
# #             f"sourced from this dataset's own methods description "
# #             f"(NOTE: 'PPO' is a genuine exception - flagged ambiguous "
# #             f"below, do not assign it a single region):\n{region_lines}\n\n"
# #             "TIER 2 - GENERAL COGNITIVE ASSOCIATION, always hedge explicitly:\n"
# #             "The 'general cognitive association' text above (e.g. "
# #             "'prefrontal regions are broadly discussed in connection with "
# #             "executive function') is background literature about the "
# #             "SCALP/FUNCTIONAL REGION IN GENERAL, not a finding from this "
# #             "project's rules or data. When you use it, you MUST phrase it "
# #             "as general background - e.g. 'this sits in the frontal region "
# #             "(sourced from the dataset montage), which the wider literature "
# #             "often associates with attention (general background, not "
# #             "something this specific rule demonstrated)' - never as 'this "
# #             "rule shows the brain is doing X'. This hedge requirement also "
# #             "covers anatomy-flavored shorthand for a functional region, not "
# #             "just obviously interpretive claims: e.g. calling the occipital "
# #             "region 'the visual cortex' or the motor region 'the movement "
# #             "control center' is ALSO a Tier 2 claim requiring the same "
# #             "hedge - only the bare functional_region label itself (frontal/"
# #             "motor/parietal/occipital) is Tier 1 and needs no hedge.\n"
# #             "TIER 3 - AMBIGUOUS, do not resolve by guessing:\n"
# #             "If a channel's functional_region is marked AMBIGUOUS (this "
# #             "applies to 'PPO' codes), say plainly that this channel sits on "
# #             "a boundary between two regions per the dataset montage "
# #             "description and you cannot state a single region for it "
# #             "confidently - do not pick one to sound more decisive.\n"
# #             "=== END CHANNEL GLOSSARY ===\n"
# #         )

# #     system_prompt = (
# #         "You are an assistant that explains a fuzzy rule-based classifier's "
# #         "output to a non-expert audience in plain English, while staying "
# #         "strictly grounded in the material provided below. You must ONLY "
# #         "use the rules and statistics provided below. Do not invent rules, "
# #         "statistics, or feature relationships that are not explicitly "
# #         "listed. Refer to the target classes using EXACTLY the labels given "
# #         "(e.g. '0 back', '2/3 back') - do not invent alternate names or "
# #         "descriptions for them. If the provided material doesn't answer the "
# #         "question, say so plainly instead of guessing.\n\n"
# #         f"You have been given {len(retrieved)} of {total_rules} total rules "
# #         "in this model, selected as most relevant to the question below. If "
# #         "the user asks you to list ALL rules or seems to expect the full "
# #         f"rule set, you MUST explicitly state that you are showing "
# #         f"{len(retrieved)} of {total_rules} total rules, not the complete "
# #         "set.\n\n"
# #         f"Rules:\n{rules_block}"
# #         f"{cv_block}"
# #         f"{stability_block}"
# #         f"{background_block}"
# #         f"{mapping_block}"
# #         "\n\nWhen explaining a rule to a non-expert:\n"
# #         "1. Use the inline [Hemisphere Location (Signal)] annotations "
# #         "already applied to each rule above - these are TIER 1 sourced "
# #         "facts (project code + standard naming convention) and you may "
# #         "state them plainly.\n"
# #         "2. If it aids understanding, you MAY add TIER 2 general "
# #         "literature associations from the channel glossary, but ONLY with "
# #         "the explicit hedge language described in the glossary section - "
# #         "never present a general association as something this specific "
# #         "rule demonstrated.\n"
# #         "3. Do not add cognitive-function claims, brain-state narratives, "
# #         "or interpretations beyond what TIER 1 facts and explicitly-hedged "
# #         "TIER 2 background support. If asked something the glossary doesn't "
# #         "cover (e.g. exact scalp coordinates, clinical meaning), say that "
# #         "plainly rather than inferring it."
# #     )

# #     return [
# #         {"role": "system", "content": system_prompt},
# #         {"role": "user", "content": question},
# #     ]


# # # ---------------------------------------------------------
# # # 5. End-to-end ask()
# # # ---------------------------------------------------------
# # def ask(question: str, data: dict, client: Client, model: str = CLOUD_MODEL,
# #         domain_background: str = None, brain_mapping: dict = None) -> tuple:
# #     if domain_background is None:
# #         domain_background = load_domain_background()
# #     if brain_mapping is None:
# #         brain_mapping = load_brain_mapping()
        
# #     retrieved = retrieve_rules(question, data["rules"], data["feature_names"])
# #     messages = build_prompt(
# #         question, retrieved, data["target_name"],
# #         total_rules=len(data["rules"]),
# #         cv_stats=data.get("cv_performance_estimate"),
# #         stability_lookup=build_stability_lookup(data),
# #         n_seeds_total=data.get("n_seeds", 1),
# #         domain_background=domain_background,
# #         brain_mapping=brain_mapping,
# #     )
# #     response = client.chat(model=model, messages=messages)
# #     return response["message"]["content"], retrieved

# """
# rag_core.py

# Pure retrieval-augmented-generation logic for the ExFuzzy rule explainer.
# No Streamlit, no input() loop, no client construction at import time -
# this module is imported by streamlit_app.py, which reruns on every user
# interaction, so anything expensive (loading JSON, building the Ollama
# client) is left to the caller to cache (see streamlit_app.py's
# @st.cache_resource).

# Retrieval and tokenisation logic match the originally-verified
# ollamaRuleExplainer.py (see RAG_Verification_and_Streamlit_Guide.md).
# Prompt construction has since been extended twice, each requiring its own
# re-verification pass:
#   - added cross-seed rule stability caveats (n_seeds / found-in-X/N-refits)
#   - added a two-tier channel glossary (brain_mapping.json) for plain-English
#     location/signal labels. Tier 1 (location + HbO/HbR direction) is sourced
#     from this project's own MainFold.py code and standard 10-10 fNIRS
#     naming; Tier 2 (cognitive-function associations) is general literature
#     background and must always be presented hedged, never as something this
#     specific model's rules demonstrated. See brain_mapping.json's "_notes"
#     field for the sourcing rationale.
# """

# import json
# import os
# import re
# from pathlib import Path

# from ollama import Client

# # ---------------------------------------------------------
# # Config
# # ---------------------------------------------------------
# RULES_PATH = "extracted_rules_final_raw.json"
# CLOUD_MODEL = "gpt-oss:20b-cloud"
# TOP_K = 5

# STOPWORDS = {
#     "is", "the", "a", "an", "and", "or", "does", "do", "what", "which",
#     "who", "how", "for", "of", "in", "on", "to", "tell", "us", "about",
#     "that", "this", "with", "are", "was", "were", "be", "it", "its",
# }


# # ---------------------------------------------------------
# # 1. Load rule base & Domain Knowledge
# # ---------------------------------------------------------
# def load_rules(path: str = RULES_PATH) -> dict:
#     with open(path, "r") as f:
#         data = json.load(f)
#     if not data.get("rules") or "raw_rules_text" in data["rules"][0]:
#         raise ValueError(
#             f"No structured rules found in {path} - "
#             "re-run MainFold.py to regenerate it."
#         )
#     return data

# def load_domain_background(path: str = "domain_background.md") -> str:
#     """Small, hand-curated background block extracted from the project's
#     own literature review."""
#     try:
#         with open(path, "r") as f:
#             return f.read().strip()
#     except FileNotFoundError:
#         return ""

# def load_brain_mapping(path: str = "brain_mapping.json") -> dict:
#     """Loads the neurological dictionary to translate opaque channel names."""
#     try:
#         with open(path, "r") as f:
#             return json.load(f)
#     except FileNotFoundError:
#         return {"signals": {}, "regions": {}, "hemispheres": {}}


# # ---------------------------------------------------------
# # 2. Load API key (with fallback to local files)
# # ---------------------------------------------------------
# def load_api_key_from_local_file() -> str:
#     """Try to load OLLAMA_API_KEY from common local file locations."""
#     candidates = [
#         Path(__file__).resolve().parent / ".streamlit" / "secrets.toml",
#         Path(__file__).resolve().parent / "secrets.toml",
#         Path(__file__).resolve().parent / "streamlitlol" / "secrets.toml",
#         Path.home() / ".streamlit" / "secrets.toml",
#     ]

#     for path in candidates:
#         if not path.exists():
#             continue
#         try:
#             import tomllib
#         except ImportError:
#             import tomli as tomllib

#         try:
#             with path.open("rb") as fh:
#                 data = tomllib.load(fh)
#         except Exception:
#             continue

#         value = data.get("OLLAMA_API_KEY", "")
#         if value:
#             return str(value)

#     return ""


# def get_client() -> Client:
#     api_key = os.environ.get("OLLAMA_API_KEY", "")
#     if not api_key:
#         api_key = load_api_key_from_local_file()
#     if not api_key:
#         raise RuntimeError(
#             "OLLAMA_API_KEY is not set. Set it via environment variable "
#             "or Streamlit secrets (see step 4 below)."
#         )
#     return Client(
#         host="https://ollama.com",
#         headers={"Authorization": "Bearer " + api_key},
#     )


# # ---------------------------------------------------------
# # 3. Retrieval (keyword overlap + substring fallback)
# # ---------------------------------------------------------
# def tokenize(text: str) -> set:
#     return set(re.findall(r"[a-zA-Z0-9]+", text.lower()))


# def retrieve_rules(query: str, rules: list, feature_names: list, top_k: int = TOP_K) -> list:
#     query_tokens = tokenize(query) - STOPWORDS
#     feature_tokens = {f.lower() for f in feature_names if f.lower() in query_tokens}

#     scored = []
#     for rule in rules:
#         rule_text = f"{rule.get('antecedent', '')} {rule.get('consequent', '')}"
#         rule_tokens = tokenize(rule_text)
#         overlap = len(query_tokens & rule_tokens) + 2 * len(feature_tokens & rule_tokens)

#         # Substring fallback
#         for qt in query_tokens:
#             if len(qt) >= 3:
#                 for rt in rule_tokens:
#                     if qt in rt or rt in qt:
#                         overlap += 1
#                         break

#         scored.append((overlap, rule))

#     scored.sort(key=lambda x: x[0], reverse=True)

#     if scored[0][0] == 0:
#         scored.sort(key=lambda x: x[1].get("accuracy", 0), reverse=True)

#     return [rule for _, rule in scored[:top_k]]


# # Generic vocabulary for questions that are clearly *about the model* but
# # wouldn't share any tokens with a specific rule/feature name (e.g. "how
# # accurate is this?", "which channels matter most?"). Kept separate from
# # STOPWORDS since these words are meaningful for scoping, just not for
# # rule-level retrieval relevance.
# META_DOMAIN_WORDS = {
#     "rule", "rules", "model", "predict", "prediction", "predicts",
#     "accuracy", "accurate", "mcc", "channel", "channels", "brain",
#     "memory", "load", "back", "fuzzy", "classifier", "dominance",
#     "explain", "explanation", "confidence", "stable", "stability",
#     "seed", "seeds", "hemisphere", "hemispheres", "region", "regions",
#     "signal", "signals", "hbo", "hbr", "subject", "subjects", "fold",
#     "folds", "reliable", "reliability", "performance", "antecedent",
#     "consequent", "score", "scores", "trust", "hallucinate",
# }


# def is_in_scope(query: str, rules: list, feature_names: list) -> bool:
#     """Cheap, deterministic pre-check for whether a question is even
#     plausibly about this model, run before spending an LLM call on it.

#     This exists because the agent is scoped to interpreting this model's
#     own outputs, not open-ended conversation - see the proposal's scope
#     note. It intentionally errs toward permissive (any shared token with
#     the rule text, feature names, or the meta-domain vocabulary above
#     counts as in-scope) since a false "in scope" just costs one LLM call,
#     while a false "out of scope" silently blocks a legitimate question.
#     """
#     query_tokens = tokenize(query) - STOPWORDS
#     if not query_tokens:
#         return False

#     domain_tokens = set(META_DOMAIN_WORDS)
#     for f in feature_names:
#         domain_tokens |= tokenize(f)
#     for rule in rules:
#         domain_tokens |= tokenize(f"{rule.get('antecedent', '')} {rule.get('consequent', '')}")

#     return bool(query_tokens & domain_tokens)


# # ---------------------------------------------------------
# # 4. Prompt construction
# # ---------------------------------------------------------
# def build_stability_lookup(data: dict) -> dict:
#     return {
#         (r["antecedent"], r["consequent"]): r["n_seeds_found"]
#         for r in data.get("rule_stability_across_seeds", [])
#     }

# def pre_translate_rule(rule_text: str, brain_map: dict) -> str:
#     """
#     Deterministically (no LLM involved) translates opaque channel codes like
#     'tmb_s1_chAF5h' into a location + signal label the LLM can use directly,
#     e.g. 'Left AF [Anterior-frontal] (HbO)'. Only SOURCED fields (signal
#     direction from MainFold.py, location from standard 10-10 naming) go into
#     this inline label - cognitive-function associations are deliberately
#     left out here and supplied separately, clearly hedged, in the prompt.
#     """
#     def replacer(match):
#         signal = match.group(1)       # 's1' or 's2'
#         region_code = match.group(2)  # 'AF', 'C', 'PO', etc.
#         hemi_code = match.group(3)    # '5h', 'z', '8', etc.

#         signal_info = brain_map.get("signals", {}).get(signal, {})
#         signal_name = signal_info.get("label", signal).split(" ")[-1].strip("()")
#         if not signal_name or signal_name == signal:
#             signal_name = "HbO" if signal == "s1" else "HbR"

#         hemi_conv = brain_map.get("hemisphere_convention", {})
#         if 'z' in hemi_code.lower():
#             hemi_name = hemi_conv.get("z", "Midline")
#         elif any(c in hemi_code for c in '13579'):
#             hemi_name = hemi_conv.get("odd", "Left")
#         elif any(c in hemi_code for c in '02468'):
#             hemi_name = hemi_conv.get("even", "Right")
#         else:
#             hemi_name = "Unknown-hemisphere"
#         # Trim the parenthetical caveat for inline brevity; full caveat is
#         # given once in the glossary block below, not repeated per-channel.
#         hemi_name = hemi_name.split(" (")[0]

#         region_info = brain_map.get("regions", {}).get(region_code, {})
#         region_desc = region_info.get("short", region_code)

#         return f"[{hemi_name} {region_desc} ({signal_name})]"

#     pattern = r"tmb_(s1|s2)_ch([a-zA-Z]+)(\d+[a-zA-Z]*|z)"
#     return re.sub(pattern, replacer, rule_text)


# def build_prompt(question: str, retrieved: list, target_name: str,
#                  total_rules: int, cv_stats: dict = None,
#                  stability_lookup: dict = None, n_seeds_total: int = 1,
#                  domain_background: str = "", brain_mapping: dict = None) -> list:
#     stability_lookup = stability_lookup or {}
#     brain_mapping = brain_mapping or {}

#     def stability_tag(r):
#         if n_seeds_total <= 1:
#             return ""
#         n_found = stability_lookup.get((r["antecedent"], r["consequent"]), 1)
#         return f", found in {n_found}/{n_seeds_total} independent model refits"
    
#     rules_block = "\n".join(
#         f"- Rule {r['rule_id']}: IF {pre_translate_rule(r['antecedent'], brain_mapping)} "
#         f"THEN {target_name} = {r['consequent']} "
#         f"(dominance_score={r.get('dominance_score', r.get('confidence', 'n/a'))}, "
#         f"accuracy={r.get('accuracy', 'n/a')}{stability_tag(r)})"
#         for r in retrieved
#     )

#     cv_block = ""
#     if cv_stats:
#         cv_block = (
#             "\n\nModel-level cross-validation performance (applies to the whole "
#             f"model, not any single rule): test accuracy = "
#             f"{cv_stats.get('test_accuracy_mean', 'n/a'):.3f} +/- "
#             f"{cv_stats.get('test_accuracy_std', 'n/a'):.3f}, test MCC = "
#             f"{cv_stats.get('test_mcc_mean', 'n/a'):.3f} +/- "
#             f"{cv_stats.get('test_mcc_std', 'n/a'):.3f}, across "
#             f"{cv_stats.get('n_folds', 'n/a')} subject-grouped folds. Use this "
#             "when asked about overall/general model reliability or accuracy, "
#             "as distinct from any single rule's own accuracy figure."
#         )

#     stability_block = ""
#     if n_seeds_total > 1:
#         stability_block = (
#             f"\n\nIMPORTANT CAVEAT: This rule set is one representative fit "
#             f"out of {n_seeds_total} independently-trained seeds (the GA "
#             "fitting procedure is stochastic). The 'found in X/N refits' "
#             "figure on each rule shows how many of those seeds re-discovered "
#             "that EXACT rule. For dominance_score specifically: state the "
#             "number itself and, if asked, that it is the rule's relative "
#             "contribution to the model's decisions - do NOT characterize "
#             "what a low or high score 'typically' or 'usually' implies "
#             "about how much the rule matters beyond that definition; you "
#             "have one number per rule, not a distribution to generalize "
#             "from. Do not assert that a pattern 'remains present', 'is "
#             "still there', or 'is real underneath the wording' when a rule "
#             "was found in only 1 of N refits - a low recurrence count means "
#             "you do NOT know whether the same underlying signal reappears "
#             "with different wording, or a genuinely different signal was "
#             "found by chance. Say that uncertainty explicitly rather than "
#             "assuming stability you have no evidence for. This is separate "
#             "from the model-level cross-validation stats above, which DO "
#             "describe genuinely stable, ensembled predictive performance."
#         )

#     background_block = ""
#     if domain_background:
#         background_block = (
#             "\n\n=== BACKGROUND CONTEXT (general literature, NOT this "
#             "model's own findings) ===\n"
#             f"{domain_background}\n"
#             "=== END BACKGROUND CONTEXT ===\n"
#         )

#     mapping_block = ""
#     if brain_mapping:
#         # Only include glossary entries for region codes that actually
#         # appear in the retrieved rules, not the full glossary every turn -
#         # keeps the prompt tight and avoids diluting context with unused
#         # channel definitions.
#         used_region_codes = {
#             m.group(2)
#             for r in retrieved
#             for m in re.finditer(
#                 r"tmb_(s1|s2)_ch([a-zA-Z]+)(\d+[a-zA-Z]*|z)",
#                 r.get("antecedent", ""),
#             )
#         }

#         sig = brain_mapping.get("signals", {})
#         sig_lines = "\n".join(
#             f"- {k}: {v.get('label', k)} - {v.get('task_load_direction', '')}"
#             for k, v in sig.items()
#         )
#         hemi = brain_mapping.get("hemisphere_convention", {})
#         hemi_lines = "\n".join(
#             f"- {k}: {v}" for k, v in hemi.items() if k != "confidence"
#         )
#         region_lines = "\n".join(
#             f"- {k}: LOCATION (10-5 naming) = {v.get('location', k)}. "
#             f"FUNCTIONAL REGION ({v.get('functional_region_basis', 'n/a')}) "
#             f"= {v.get('functional_region', 'not determined')}. "
#             f"GENERAL COGNITIVE ASSOCIATION (hedge required) = "
#             f"{v.get('general_association', 'none supplied')}"
#             for k, v in brain_mapping.get("regions", {}).items()
#             if k in used_region_codes
#         )
#         mapping_block = (
#             "\n\n=== CHANNEL GLOSSARY (three confidence tiers - read carefully) ===\n"
#             "Each rule below has already been annotated inline with "
#             "[Hemisphere Location (Signal)] using this glossary, so you don't "
#             "need to re-derive it - use those inline annotations directly "
#             "when explaining a rule to the user.\n\n"
#             "TIER 1 - SOURCED, state confidently:\n"
#             "Signal direction (from this project's own feature-engineering "
#             f"code):\n{sig_lines}\n"
#             f"Hemisphere convention (confirmed for this dataset - its "
#             f"methods description states the montage follows the "
#             f"international 10-5 system):\n{hemi_lines}\n"
#             f"Scalp location and FUNCTIONAL REGION (frontal/motor/parietal/"
#             f"occipital) for the channels appearing in the rules above, "
#             f"sourced from this dataset's own methods description "
#             f"(NOTE: 'PPO' is a genuine exception - flagged ambiguous "
#             f"below, do not assign it a single region):\n{region_lines}\n\n"
#             "TIER 2 - GENERAL COGNITIVE ASSOCIATION, always hedge explicitly:\n"
#             "The 'general cognitive association' text above (e.g. "
#             "'prefrontal regions are broadly discussed in connection with "
#             "executive function') is background literature about the "
#             "SCALP/FUNCTIONAL REGION IN GENERAL, not a finding from this "
#             "project's rules or data. When you use it, you MUST phrase it "
#             "as general background - e.g. 'this sits in the frontal region "
#             "(sourced from the dataset montage), which the wider literature "
#             "often associates with attention (general background, not "
#             "something this specific rule demonstrated)' - never as 'this "
#             "rule shows the brain is doing X'. This hedge requirement also "
#             "covers anatomy-flavored shorthand for a functional region, not "
#             "just obviously interpretive claims: e.g. calling the occipital "
#             "region 'the visual cortex' or the motor region 'the movement "
#             "control center' is ALSO a Tier 2 claim requiring the same "
#             "hedge - only the bare functional_region label itself (frontal/"
#             "motor/parietal/occipital) is Tier 1 and needs no hedge.\n"
#             "TIER 3 - AMBIGUOUS, do not resolve by guessing:\n"
#             "If a channel's functional_region is marked AMBIGUOUS (this "
#             "applies to 'PPO' codes), say plainly that this channel sits on "
#             "a boundary between two regions per the dataset montage "
#             "description and you cannot state a single region for it "
#             "confidently - do not pick one to sound more decisive.\n"
#             "=== END CHANNEL GLOSSARY ===\n"
#         )

#     system_prompt = (
#         "You are an assistant that explains a fuzzy rule-based classifier's "
#         "output to a non-expert audience in plain English, while staying "
#         "strictly grounded in the material provided below. You must ONLY "
#         "use the rules and statistics provided below. Do not invent rules, "
#         "statistics, or feature relationships that are not explicitly "
#         "listed. Refer to the target classes using EXACTLY the labels given "
#         "(e.g. '0 back', '2/3 back') - do not invent alternate names or "
#         "descriptions for them. If the provided material doesn't answer the "
#         "question, say so plainly instead of guessing.\n\n"
#         f"You have been given {len(retrieved)} of {total_rules} total rules "
#         "in this model, selected as most relevant to the question below. If "
#         "the user asks you to list ALL rules or seems to expect the full "
#         f"rule set, you MUST explicitly state that you are showing "
#         f"{len(retrieved)} of {total_rules} total rules, not the complete "
#         "set.\n\n"
#         f"Rules:\n{rules_block}"
#         f"{cv_block}"
#         f"{stability_block}"
#         f"{background_block}"
#         f"{mapping_block}"
#         "\n\nWhen explaining a rule to a non-expert:\n"
#         "1. Use the inline [Hemisphere Location (Signal)] annotations "
#         "already applied to each rule above - these are TIER 1 sourced "
#         "facts (project code + standard naming convention) and you may "
#         "state them plainly.\n"
#         "2. If it aids understanding, you MAY add TIER 2 general "
#         "literature associations from the channel glossary, but ONLY with "
#         "the explicit hedge language described in the glossary section - "
#         "never present a general association as something this specific "
#         "rule demonstrated.\n"
#         "3. Do not add cognitive-function claims, brain-state narratives, "
#         "or interpretations beyond what TIER 1 facts and explicitly-hedged "
#         "TIER 2 background support. If asked something the glossary doesn't "
#         "cover (e.g. exact scalp coordinates, clinical meaning), say that "
#         "plainly rather than inferring it."
#     )

#     return [
#         {"role": "system", "content": system_prompt},
#         {"role": "user", "content": question},
#     ]


# # ---------------------------------------------------------
# # 5. End-to-end ask()
# # ---------------------------------------------------------
# def ask(question: str, data: dict, client: Client, model: str = CLOUD_MODEL,
#         domain_background: str = None, brain_mapping: dict = None) -> tuple:
#     if domain_background is None:
#         domain_background = load_domain_background()
#     if brain_mapping is None:
#         brain_mapping = load_brain_mapping()
        
#     retrieved = retrieve_rules(question, data["rules"], data["feature_names"])
#     messages = build_prompt(
#         question, retrieved, data["target_name"],
#         total_rules=len(data["rules"]),
#         cv_stats=data.get("cv_performance_estimate"),
#         stability_lookup=build_stability_lookup(data),
#         n_seeds_total=data.get("n_seeds", 1),
#         domain_background=domain_background,
#         brain_mapping=brain_mapping,
#     )
#     response = client.chat(model=model, messages=messages)
#     return response["message"]["content"], retrieved


# # ---------------------------------------------------------
# # 6. No-LLM fallback
# # ---------------------------------------------------------
# def build_fallback_answer(retrieved: list, target_name: str) -> str:
#     """Deterministic answer built directly from the retrieved rules, with
#     no LLM call involved. Used when the LLM API is unreachable, times
#     out, or errors, so the system degrades to something still useful
#     instead of returning a bare error - see the proposal's risk table
#     entry for LLM inconsistency/hallucination mitigation."""
#     if not retrieved:
#         return (
#             "The explanation model is temporarily unavailable, and no "
#             "matching rules were found for this question either. Please "
#             "try again shortly, or rephrase the question."
#         )
#     lines = [
#         "The natural-language explanation model is temporarily "
#         "unavailable, so here are the raw matching rules without further "
#         "explanation:",
#         "",
#     ]
#     for r in retrieved:
#         lines.append(
#             f"- Rule {r['rule_id']}: IF {r['antecedent']} THEN "
#             f"{target_name} = {r['consequent']} "
#             f"(accuracy={r.get('accuracy', 'n/a')}, "
#             f"dominance={r.get('dominance_score', r.get('confidence', 'n/a'))})"
#         )
#     return "\n".join(lines)

"""
rag_core.py

Pure retrieval-augmented-generation logic for the ExFuzzy rule explainer.
No Streamlit, no input() loop, no client construction at import time -
this module is imported by streamlit_app.py, which reruns on every user
interaction, so anything expensive (loading JSON, building the Ollama
client) is left to the caller to cache (see streamlit_app.py's
@st.cache_resource).

Retrieval and tokenisation logic match the originally-verified
ollamaRuleExplainer.py (see RAG_Verification_and_Streamlit_Guide.md).
Prompt construction has since been extended several times, each requiring
its own re-verification pass:
  - added cross-seed rule stability caveats (n_seeds / found-in-X/N-refits)
  - added a two-tier channel glossary (brain_mapping.json) for plain-English
    location/signal labels. Tier 1 (location + HbO/HbR direction) is sourced
    from this project's own MainFold.py code and standard 10-10 fNIRS
    naming; Tier 2 (cognitive-function associations) is general literature
    background and must always be presented hedged, never as something this
    specific model's rules demonstrated. See brain_mapping.json's "_notes"
    field for the sourcing rationale.
  - added scope-guarding (is_in_scope) and a no-LLM fallback
    (build_fallback_answer) for backend.py.
  - added visualization tool-calling (ask_with_visualization): a two-call
    flow validated against gpt-oss:20b-cloud in tool_call_test.py. Call 1
    lets the model decide (via native tool-calling) whether a topomap or
    chord diagram would help and which channels are involved; the backend
    then builds that visualization's data DETERMINISTICALLY from the
    already-retrieved rules (never from the LLM's raw tool arguments -
    see viz_tools.py's module docstring for why), and call 2 asks the
    model to narrate around exactly what was rendered. When no tool is
    called, this collapses back to a single call, identical to ask().
"""

import json
import os
import re
from pathlib import Path

from ollama import Client

from viz_tools import (
    VIZ_TOOLS,
    load_channel_atlas,
    resolve_requested_channels,
    build_topomap_payload,
    build_chord_payload,
    summarize_topomap_for_llm,
    summarize_chord_for_llm,
)

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
RULES_PATH = "extracted_rules_final_raw.json"
CLOUD_MODEL = "gpt-oss:20b-cloud"
TOP_K = 5

STOPWORDS = {
    "is", "the", "a", "an", "and", "or", "does", "do", "what", "which",
    "who", "how", "for", "of", "in", "on", "to", "tell", "us", "about",
    "that", "this", "with", "are", "was", "were", "be", "it", "its",
}

# Generic vocabulary for questions that are clearly *about the model* but
# wouldn't share any tokens with a specific rule/feature name (e.g. "how
# accurate is this?", "which channels matter most?"). Kept separate from
# STOPWORDS since these words are meaningful for scoping, just not for
# rule-level retrieval relevance.
META_DOMAIN_WORDS = {
    "rule", "rules", "model", "predict", "prediction", "predicts",
    "accuracy", "accurate", "mcc", "channel", "channels", "brain",
    "memory", "load", "back", "fuzzy", "classifier", "dominance",
    "explain", "explanation", "confidence", "stable", "stability",
    "seed", "seeds", "hemisphere", "hemispheres", "region", "regions",
    "signal", "signals", "hbo", "hbr", "subject", "subjects", "fold",
    "folds", "reliable", "reliability", "performance", "antecedent",
    "consequent", "score", "scores", "trust", "hallucinate",
}


# ---------------------------------------------------------
# 1. Load rule base & Domain Knowledge
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


def load_domain_background(path: str = "domain_background.md") -> str:
    """Small, hand-curated background block extracted from the project's
    own literature review."""
    try:
        with open(path, "r") as f:
            return f.read().strip()
    except FileNotFoundError:
        return ""


def load_brain_mapping(path: str = "brain_mapping.json") -> dict:
    """Loads the neurological dictionary to translate opaque channel names."""
    try:
        with open(path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {"signals": {}, "regions": {}, "hemispheres": {}}


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


def get_client(api_key: str = None) -> Client:
    if not api_key:
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

        # Substring fallback
        for qt in query_tokens:
            if len(qt) >= 3:
                for rt in rule_tokens:
                    if qt in rt or rt in qt:
                        overlap += 1
                        break

        scored.append((overlap, rule))

    scored.sort(key=lambda x: x[0], reverse=True)

    if scored[0][0] == 0:
        scored.sort(key=lambda x: x[1].get("accuracy", 0), reverse=True)

    return [rule for _, rule in scored[:top_k]]


def is_in_scope(query: str, rules: list, feature_names: list) -> bool:
    """Cheap, deterministic pre-check for whether a question is even
    plausibly about this model, run before spending an LLM call on it.

    This exists because the agent is scoped to interpreting this model's
    own outputs, not open-ended conversation - see the proposal's scope
    note. It intentionally errs toward permissive (any shared token with
    the rule text, feature names, or the meta-domain vocabulary above
    counts as in-scope) since a false "in scope" just costs one LLM call,
    while a false "out of scope" silently blocks a legitimate question.
    """
    query_tokens = tokenize(query) - STOPWORDS
    if not query_tokens:
        return False

    domain_tokens = set(META_DOMAIN_WORDS)
    for f in feature_names:
        domain_tokens |= tokenize(f)
    for rule in rules:
        domain_tokens |= tokenize(f"{rule.get('antecedent', '')} {rule.get('consequent', '')}")

    return bool(query_tokens & domain_tokens)


# ---------------------------------------------------------
# 4. Prompt construction
# ---------------------------------------------------------
def build_stability_lookup(data: dict) -> dict:
    return {
        (r["antecedent"], r["consequent"]): r["n_seeds_found"]
        for r in data.get("rule_stability_across_seeds", [])
    }


def pre_translate_rule(rule_text: str, brain_map: dict) -> str:
    """
    Deterministically (no LLM involved) translates opaque channel codes like
    'tmb_s1_chAF5h' into a location + signal label the LLM can use directly,
    e.g. 'Left AF [Anterior-frontal] (HbO)'. Only SOURCED fields (signal
    direction from MainFold.py, location from standard 10-10 naming) go into
    this inline label - cognitive-function associations are deliberately
    left out here and supplied separately, clearly hedged, in the prompt.
    """
    def replacer(match):
        signal = match.group(1)       # 's1' or 's2'
        region_code = match.group(2)  # 'AF', 'C', 'PO', etc.
        hemi_code = match.group(3)    # '5h', 'z', '8', etc.

        signal_info = brain_map.get("signals", {}).get(signal, {})
        signal_name = signal_info.get("label", signal).split(" ")[-1].strip("()")
        if not signal_name or signal_name == signal:
            signal_name = "HbO" if signal == "s1" else "HbR"

        hemi_conv = brain_map.get("hemisphere_convention", {})
        if 'z' in hemi_code.lower():
            hemi_name = hemi_conv.get("z", "Midline")
        elif any(c in hemi_code for c in '13579'):
            hemi_name = hemi_conv.get("odd", "Left")
        elif any(c in hemi_code for c in '02468'):
            hemi_name = hemi_conv.get("even", "Right")
        else:
            hemi_name = "Unknown-hemisphere"
        hemi_name = hemi_name.split(" (")[0]

        region_info = brain_map.get("regions", {}).get(region_code, {})
        region_desc = region_info.get("short", region_code)

        return f"[{hemi_name} {region_desc} ({signal_name})]"

    pattern = r"tmb_(s1|s2)_ch([a-zA-Z]+)(\d+[a-zA-Z]*|z)"
    return re.sub(pattern, replacer, rule_text)


def build_prompt(question: str, retrieved: list, target_name: str,
                 total_rules: int, cv_stats: dict = None,
                 stability_lookup: dict = None, n_seeds_total: int = 1,
                 domain_background: str = "", brain_mapping: dict = None,
                 enable_viz_tools: bool = False) -> list:
    stability_lookup = stability_lookup or {}
    brain_mapping = brain_mapping or {}

    def stability_tag(r):
        if n_seeds_total <= 1:
            return ""
        n_found = stability_lookup.get((r["antecedent"], r["consequent"]), 1)
        return f", found in {n_found}/{n_seeds_total} independent model refits"

    rules_block = "\n".join(
        f"- Rule {r['rule_id']}: IF {pre_translate_rule(r['antecedent'], brain_mapping)} "
        f"THEN {target_name} = {r['consequent']} "
        f"(dominance_score={r.get('dominance_score', r.get('confidence', 'n/a'))}, "
        f"accuracy={r.get('accuracy', 'n/a')}{stability_tag(r)})"
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

    stability_block = ""
    if n_seeds_total > 1:
        stability_block = (
            f"\n\nIMPORTANT CAVEAT: This rule set is one representative fit "
            f"out of {n_seeds_total} independently-trained seeds (the GA "
            "fitting procedure is stochastic). The 'found in X/N refits' "
            "figure on each rule shows how many of those seeds re-discovered "
            "that EXACT rule. For dominance_score specifically: state the "
            "number itself and, if asked, that it is the rule's relative "
            "contribution to the model's decisions - do NOT characterize "
            "what a low or high score 'typically' or 'usually' implies "
            "about how much the rule matters beyond that definition; you "
            "have one number per rule, not a distribution to generalize "
            "from. Do not assert that a pattern 'remains present', 'is "
            "still there', or 'is real underneath the wording' when a rule "
            "was found in only 1 of N refits - a low recurrence count means "
            "you do NOT know whether the same underlying signal reappears "
            "with different wording, or a genuinely different signal was "
            "found by chance. Say that uncertainty explicitly rather than "
            "assuming stability you have no evidence for. This is separate "
            "from the model-level cross-validation stats above, which DO "
            "describe genuinely stable, ensembled predictive performance."
        )

    background_block = ""
    if domain_background:
        background_block = (
            "\n\n=== BACKGROUND CONTEXT (general literature, NOT this "
            "model's own findings) ===\n"
            f"{domain_background}\n"
            "=== END BACKGROUND CONTEXT ===\n"
        )

    mapping_block = ""
    if brain_mapping:
        used_region_codes = {
            m.group(2)
            for r in retrieved
            for m in re.finditer(
                r"tmb_(s1|s2)_ch([a-zA-Z]+)(\d+[a-zA-Z]*|z)",
                r.get("antecedent", ""),
            )
        }

        sig = brain_mapping.get("signals", {})
        sig_lines = "\n".join(
            f"- {k}: {v.get('label', k)} - {v.get('task_load_direction', '')}"
            for k, v in sig.items()
        )
        hemi = brain_mapping.get("hemisphere_convention", {})
        hemi_lines = "\n".join(
            f"- {k}: {v}" for k, v in hemi.items() if k != "confidence"
        )
        region_lines = "\n".join(
            f"- {k}: LOCATION (10-5 naming) = {v.get('location', k)}. "
            f"FUNCTIONAL REGION ({v.get('functional_region_basis', 'n/a')}) "
            f"= {v.get('functional_region', 'not determined')}. "
            f"GENERAL COGNITIVE ASSOCIATION (hedge required) = "
            f"{v.get('general_association', 'none supplied')}"
            for k, v in brain_mapping.get("regions", {}).items()
            if k in used_region_codes
        )
        mapping_block = (
            "\n\n=== CHANNEL GLOSSARY (three confidence tiers - read carefully) ===\n"
            "Each rule below has already been annotated inline with "
            "[Hemisphere Location (Signal)] using this glossary, so you don't "
            "need to re-derive it - use those inline annotations directly "
            "when explaining a rule to the user.\n\n"
            "TIER 1 - SOURCED, state confidently:\n"
            "Signal direction (from this project's own feature-engineering "
            f"code):\n{sig_lines}\n"
            f"Hemisphere convention (confirmed for this dataset - its "
            f"methods description states the montage follows the "
            f"international 10-5 system):\n{hemi_lines}\n"
            f"Scalp location and FUNCTIONAL REGION (frontal/motor/parietal/"
            f"occipital) for the channels appearing in the rules above, "
            f"sourced from this dataset's own methods description "
            f"(NOTE: 'PPO' is a genuine exception - flagged ambiguous "
            f"below, do not assign it a single region):\n{region_lines}\n\n"
            "TIER 2 - GENERAL COGNITIVE ASSOCIATION, always hedge explicitly:\n"
            "The 'general cognitive association' text above (e.g. "
            "'prefrontal regions are broadly discussed in connection with "
            "executive function') is background literature about the "
            "SCALP/FUNCTIONAL REGION IN GENERAL, not a finding from this "
            "project's rules or data. When you use it, you MUST phrase it "
            "as general background - e.g. 'this sits in the frontal region "
            "(sourced from the dataset montage), which the wider literature "
            "often associates with attention (general background, not "
            "something this specific rule demonstrated)' - never as 'this "
            "rule shows the brain is doing X'. This hedge requirement also "
            "covers anatomy-flavored shorthand for a functional region, not "
            "just obviously interpretive claims: e.g. calling the occipital "
            "region 'the visual cortex' or the motor region 'the movement "
            "control center' is ALSO a Tier 2 claim requiring the same "
            "hedge - only the bare functional_region label itself (frontal/"
            "motor/parietal/occipital) is Tier 1 and needs no hedge.\n"
            "TIER 3 - AMBIGUOUS, do not resolve by guessing:\n"
            "If a channel's functional_region is marked AMBIGUOUS (this "
            "applies to 'PPO' codes), say plainly that this channel sits on "
            "a boundary between two regions per the dataset montage "
            "description and you cannot state a single region for it "
            "confidently - do not pick one to sound more decisive.\n"
            "=== END CHANNEL GLOSSARY ===\n"
        )

    viz_block = ""
    if enable_viz_tools:
        viz_block = (
            "\n\nYou also have access to show_topomap and show_chord_diagram "
            "tools. Call one ONLY if the user is genuinely asking to see "
            "channel locations (show_topomap) or relationships/connections "
            "between channels (show_chord_diagram) - plain informational "
            "questions (e.g. about accuracy, rule counts, stability) should "
            "get a text answer with no tool call."
        )

    system_prompt = (
        "You are an assistant that explains a fuzzy rule-based classifier's "
        "output to a non-expert audience in plain English, while staying "
        "strictly grounded in the material provided below. You must ONLY "
        "use the rules and statistics provided below. Do not invent rules, "
        "statistics, or feature relationships that are not explicitly "
        "listed. Refer to the target classes using EXACTLY the labels given "
        "(e.g. '0 back', '2/3 back') - do not invent alternate names or "
        "descriptions for them. If the provided material doesn't answer the "
        "question, say so plainly instead of guessing.\n\n"
        f"You have been given {len(retrieved)} of {total_rules} total rules "
        "in this model, selected as most relevant to the question below. If "
        "the user asks you to list ALL rules or seems to expect the full "
        f"rule set, you MUST explicitly state that you are showing "
        f"{len(retrieved)} of {total_rules} total rules, not the complete "
        "set.\n\n"
        f"Rules:\n{rules_block}"
        f"{cv_block}"
        f"{stability_block}"
        f"{background_block}"
        f"{mapping_block}"
        f"{viz_block}"
        "\n\nWhen explaining a rule to a non-expert:\n"
        "1. Use the inline [Hemisphere Location (Signal)] annotations "
        "already applied to each rule above - these are TIER 1 sourced "
        "facts (project code + standard naming convention) and you may "
        "state them plainly.\n"
        "2. If it aids understanding, you MAY add TIER 2 general "
        "literature associations from the channel glossary, but ONLY with "
        "the explicit hedge language described in the glossary section - "
        "never present a general association as something this specific "
        "rule demonstrated.\n"
        "3. Do not add cognitive-function claims, brain-state narratives, "
        "or interpretations beyond what TIER 1 facts and explicitly-hedged "
        "TIER 2 background support. If asked something the glossary doesn't "
        "cover (e.g. exact scalp coordinates, clinical meaning), say that "
        "plainly rather than inferring it."
    )

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]


# ---------------------------------------------------------
# 5. End-to-end ask() - no visualization tools, single LLM call
# ---------------------------------------------------------
def ask(question: str, data: dict, client: Client, model: str = CLOUD_MODEL,
        domain_background: str = None, brain_mapping: dict = None) -> tuple:
    if domain_background is None:
        domain_background = load_domain_background()
    if brain_mapping is None:
        brain_mapping = load_brain_mapping()

    retrieved = retrieve_rules(question, data["rules"], data["feature_names"])
    messages = build_prompt(
        question, retrieved, data["target_name"],
        total_rules=len(data["rules"]),
        cv_stats=data.get("cv_performance_estimate"),
        stability_lookup=build_stability_lookup(data),
        n_seeds_total=data.get("n_seeds", 1),
        domain_background=domain_background,
        brain_mapping=brain_mapping,
    )
    response = client.chat(model=model, messages=messages)
    return response["message"]["content"], retrieved


# ---------------------------------------------------------
# 6. ask_with_visualization() - the two-call tool-calling flow
# ---------------------------------------------------------
def ask_with_visualization(question: str, data: dict, client: Client,
                            model: str = CLOUD_MODEL,
                            domain_background: str = None,
                            brain_mapping: dict = None,
                            channel_atlas: dict = None) -> dict:
    """
    Call 1: model sees the tool schemas and decides whether to call
    show_topomap / show_chord_diagram. If it calls one, content comes
    back EMPTY (confirmed in tool_call_test.py against gpt-oss:20b-cloud)
    - that's expected model behaviour, not an error.

    If a tool was called: the backend builds the actual visualization
    payload deterministically from `retrieved` (see viz_tools.py), then
    Call 2 hands the model a compact summary of what was rendered and
    asks it to narrate around it.

    If no tool was called: Call 1's content IS the final answer, and this
    collapses to the same cost/shape as ask().

    Returns a dict rather than a tuple (unlike ask()) since there's a
    third thing to hand back now - keeps the growing return shape
    self-documenting at call sites instead of positional and easy to
    mix up.
    """
    if domain_background is None:
        domain_background = load_domain_background()
    if brain_mapping is None:
        brain_mapping = load_brain_mapping()
    if channel_atlas is None:
        channel_atlas = load_channel_atlas()

    retrieved = retrieve_rules(question, data["rules"], data["feature_names"])
    messages = build_prompt(
        question, retrieved, data["target_name"],
        total_rules=len(data["rules"]),
        cv_stats=data.get("cv_performance_estimate"),
        stability_lookup=build_stability_lookup(data),
        n_seeds_total=data.get("n_seeds", 1),
        domain_background=domain_background,
        brain_mapping=brain_mapping,
        enable_viz_tools=True,
    )

    first = client.chat(model=model, messages=messages, tools=VIZ_TOOLS)
    tool_calls = first["message"].get("tool_calls") or []

    if not tool_calls:
        return {
            "answer": first["message"]["content"],
            "retrieved": retrieved,
            "visualization": None,
        }

    # Only act on the first tool call - MVP scope, one visualization per
    # turn. Revisit if a question genuinely warrants both diagrams at once.
    call = tool_calls[0]
    tool_name = call["function"]["name"]
    raw_args = call["function"]["arguments"] or {}
    llm_channels = raw_args.get("channels", [])

    requested = resolve_requested_channels(llm_channels, retrieved)

    if tool_name == "show_topomap":
        points = build_topomap_payload(retrieved, requested, channel_atlas)
        visualization = {"type": "topomap", "topomap": points, "chord": None}
        tool_summary = summarize_topomap_for_llm(points)
    elif tool_name == "show_chord_diagram":
        chord = build_chord_payload(retrieved, requested, channel_atlas)
        visualization = {"type": "chord", "topomap": None, "chord": chord}
        tool_summary = summarize_chord_for_llm(chord)
    else:
        # Unknown tool name - degrade gracefully rather than crash the turn.
        return {
            "answer": first["message"]["content"] or (
                "I tried to show a visualization but something went wrong "
                "selecting it - here's what I can tell you from the rules "
                "directly instead."
            ),
            "retrieved": retrieved,
            "visualization": None,
        }

    second_messages = messages + [
        {"role": "assistant", "content": "", "tool_calls": tool_calls},
        {"role": "tool", "content": tool_summary},
        {"role": "user", "content": (
            "Now explain the answer to my original question in plain "
            "English, referring to the visualization above where it "
            "helps. Don't describe it as an image you can see - just "
            "narrate what it shows using the data given."
        )},
    ]
    second = client.chat(model=model, messages=second_messages)

    return {
        "answer": second["message"]["content"],
        "retrieved": retrieved,
        "visualization": visualization,
    }


# ---------------------------------------------------------
# 7. No-LLM fallback
# ---------------------------------------------------------
def build_fallback_answer(retrieved: list, target_name: str) -> str:
    """Deterministic answer built directly from the retrieved rules, with
    no LLM call involved. Used when the LLM API is unreachable, times
    out, or errors, so the system degrades to something still useful
    instead of returning a bare error - see the proposal's risk table
    entry for LLM inconsistency/hallucination mitigation."""
    if not retrieved:
        return (
            "The explanation model is temporarily unavailable, and no "
            "matching rules were found for this question either. Please "
            "try again shortly, or rephrase the question."
        )
    lines = [
        "The natural-language explanation model is temporarily "
        "unavailable, so here are the raw matching rules without further "
        "explanation:",
        "",
    ]
    for r in retrieved:
        lines.append(
            f"- Rule {r['rule_id']}: IF {r['antecedent']} THEN "
            f"{target_name} = {r['consequent']} "
            f"(accuracy={r.get('accuracy', 'n/a')}, "
            f"dominance={r.get('dominance_score', r.get('confidence', 'n/a'))})"
        )
    return "\n".join(lines)