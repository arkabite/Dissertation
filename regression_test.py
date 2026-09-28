"""
regression_test.py

Offline, deterministic regression suite for rag_core.py / viz_tools.py /
backend.py. Unlike tool_call_test.py (which needs a live OLLAMA_API_KEY
and tests the LLM's own tool-calling judgment), everything here runs
instantly with no network access and no API key, because every bug found
during this project's testing turned out to be deterministic code, not
LLM variance:

  - is_in_scope() blocking natural viz phrasing (bare channel codes /
    viz vocabulary not in the domain token set)
  - retrieve_rules()'s zero-overlap tie-break silently hiding the
    lowest-accuracy rule on every generic question
  - show_channel_neighbors fabricating non-existent channel names when
    asked a spatial-adjacency question
  - classify_scope() misrouting "explain X" questions to the grounded
    tier regardless of topic, due to overly generic META_DOMAIN_WORDS
  - retrieve_literature() needing to return nothing rather than an
    irrelevant citation when nothing matches

Run locally, any time you change rag_core.py / viz_tools.py / backend.py,
before trusting a live-model transcript to tell you something still
works:

    python regression_test.py

Exits 0 if every check passes, 1 otherwise - safe to wire into a CI step
or a pre-deploy check later.
"""

import sys

import rag_core as rc
import viz_tools as vt
import backend

from fastapi.testclient import TestClient

FAILURES = []


def check(label: str, condition: bool, detail: str = ""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}" + (f" - {detail}" if detail and not condition else ""))
    if not condition:
        FAILURES.append(label)


# ---------------------------------------------------------
# Shared fixtures - the real project data, not synthetic stand-ins.
# ---------------------------------------------------------
DATA = rc.load_rules()
RULES = DATA["rules"]
FEATURES = DATA["feature_names"]
ATLAS = vt.load_channel_atlas()
LITERATURE = rc.load_literature()


class FakeClient:
    """Deterministic stand-in for ollama.Client - never makes a real
    call, so this whole suite runs offline. Branches on system-prompt
    content to return the right shape of fake response for whichever
    code path is under test."""

    def chat(self, model, messages, tools=None):
        sysmsg = messages[0]["content"]

        if "general neuroscience/fNIRS background" in sysmsg:
            return {"message": {"content": (
                "**General neuroscience background** - not derived from "
                "this model's own findings.\n\nFake general answer."
            )}}

        # Narration turn (second call in ask_with_visualization) - has a
        # tool role message right before it.
        if any(m.get("role") == "tool" for m in messages):
            return {"message": {"content": "Fake narration referencing the visualization."}}

        # First call of ask_with_visualization - decide a tool call based
        # on keywords in the last user message, mimicking realistic model
        # behaviour for each of the three viz tools.
        question = messages[-1]["content"].lower()
        if "neighbor" in question or "near" in question:
            return {"message": {"content": "", "tool_calls": [
                {"function": {"name": "show_channel_neighbors", "arguments": {"channel": "AF7"}}}
            ]}}
        if "chord" in question or "connect" in question or "relate" in question:
            return {"message": {"content": "", "tool_calls": [
                {"function": {"name": "show_chord_diagram", "arguments": {"channels": ["AF7"]}}}
            ]}}
        if "where" in question or "locat" in question:
            return {"message": {"content": "", "tool_calls": [
                {"function": {"name": "show_topomap", "arguments": {"channels": ["AF7"]}}}
            ]}}
        return {"message": {"content": "Fake grounded text answer.", "tool_calls": []}}


FAKE_CLIENT = FakeClient()


# ===========================================================
# 1. classify_scope() - the three-way router
# ===========================================================
print("\n--- 1. classify_scope() ---")
SCOPE_CASES = [
    ("Where is AF7 located on the head?", "grounded"),
    ("How are AF7 and C6h related?", "grounded"),
    ("Which channels are near AF7 on the scalp?", "grounded"),
    ("How reliable is this model overall?", "grounded"),
    ("What is working memory?", "grounded"),          # "memory" is this model's own vocabulary
    ("Explain the n-back task", "grounded"),           # "back" is this model's own target-class vocabulary (0-back/2/3-back) - same reasoning as "memory" above
    ("What is the hemodynamic response function?", "general"),
    ("explain cognitive load", "general"),             # regression: "explain"/"load" false-positive
    ("explain the water cycle", "out_of_scope"),        # smoking gun for the same bug
    ("what is the prefrontal cortex", "general"),
    ("What is the capital of France?", "out_of_scope"),
    ("Write me a poem about the weather", "out_of_scope"),
]
for question, expected in SCOPE_CASES:
    got = rc.classify_scope(question, RULES, FEATURES)
    check(f"classify_scope({question!r})", got == expected, f"expected {expected}, got {got}")


# ===========================================================
# 2. retrieve_rules() - zero-overlap tie-break must not hide the
#    lowest-accuracy rule, and normal keyword matches still work.
# ===========================================================
print("\n--- 2. retrieve_rules() ---")
generic = rc.retrieve_rules("Show me the whole channel map", RULES, FEATURES)
check(
    "generic overview query returns ALL rules (not accuracy-sorted top-k)",
    len(generic) == len(RULES),
    f"got {len(generic)} of {len(RULES)} rules",
)
lowest_accuracy_id = min(RULES, key=lambda r: r["accuracy"])["rule_id"]
check(
    "lowest-accuracy rule is NOT silently excluded from a generic query",
    lowest_accuracy_id in [r["rule_id"] for r in generic],
    f"rule {lowest_accuracy_id} missing",
)
specific = rc.retrieve_rules("Where is AF7 located?", RULES, FEATURES)
check(
    "specific channel query still ranks the matching rule first",
    specific[0]["rule_id"] == 1,
    f"got rule {specific[0]['rule_id']} first",
)


# ===========================================================
# 3. show_channel_neighbors - must never fabricate a channel
# ===========================================================
print("\n--- 3. Channel-neighbor tool (viz_tools.py) ---")
resolved = vt.resolve_single_channel("AF7", ATLAS)
payload = vt.build_neighbors_payload(resolved, ATLAS)
real_neighbors = set(n["channel"] for n in payload["neighbors"])
check(
    "AF7's neighbors match the atlas exactly",
    real_neighbors == {"AF5h", "AFF5", "AFp7"},
    f"got {real_neighbors}",
)
fake_channel = vt.resolve_single_channel("AF99", ATLAS)
fake_payload = vt.build_neighbors_payload(fake_channel, ATLAS)
check(
    "nonexistent channel resolves to found=False, never fabricated",
    fake_payload["found"] is False and fake_payload["neighbors"] == [],
)
check(
    "lowercase/colloquial input still resolves correctly",
    vt.resolve_single_channel("af7", ATLAS) == "AF7",
)


# ===========================================================
# 4. retrieve_literature() - must return nothing rather than a forced,
#    irrelevant citation
# ===========================================================
print("\n--- 4. retrieve_literature() ---")
check(
    "irrelevant/unrelated query returns no citation",
    rc.retrieve_literature("What is the capital of France?", LITERATURE) == [],
)
matched = rc.retrieve_literature("What is EEG?", LITERATURE)
check(
    "EEG question matches the Shin et al. dataset paper",
    any(e["id"] == "shin2018" for e in matched),
    f"got {[e['id'] for e in matched]}",
)


# ===========================================================
# 5. ask_with_visualization() - narration-safety note always present,
#    every visualization type produces the right payload shape
# ===========================================================
print("\n--- 5. ask_with_visualization() (offline, via FakeClient) ---")
for question, expected_type in [
    ("Where is AF7 located?", "topomap"),
    ("How are AF7 and C6h related?", "chord"),
    ("Which channels are near AF7?", "neighbors"),
]:
    result = rc.ask_with_visualization(
        question, DATA, FAKE_CLIENT,
        domain_background=rc.load_domain_background(),
        brain_mapping=rc.load_brain_mapping(),
        channel_atlas=ATLAS,
    )
    check(
        f"'{question}' -> visualization.type == {expected_type!r}",
        result["visualization"] is not None and result["visualization"]["type"] == expected_type,
        f"got {result['visualization']}",
    )
    check(
        f"'{question}' -> scope_tier tagged 'grounded'",
        result["scope_tier"] == "grounded",
    )


# ===========================================================
# 6. Full HTTP surface - backend.py's /ask, all three tiers, over a
#    real FastAPI TestClient (exercises startup, Pydantic validation,
#    and the actual endpoint code, not just the underlying functions).
# ===========================================================
print("\n--- 6. backend.py /ask endpoint (FastAPI TestClient) ---")
with TestClient(backend.app) as http_client:
    backend.state["client"] = FAKE_CLIENT
    backend.state["client_error"] = None

    endpoint_cases = [
        ("Where is AF7 located on the head?", "grounded"),
        ("What is the hemodynamic response function?", "general"),
        ("What is the capital of France?", "out_of_scope"),
    ]
    for question, expected_tier in endpoint_cases:
        resp = http_client.post("/ask", json={"question": question})
        body = resp.json()
        check(
            f"POST /ask {question!r} -> 200",
            resp.status_code == 200,
            f"got {resp.status_code}",
        )
        check(
            f"POST /ask {question!r} -> scope_tier == {expected_tier!r}",
            body.get("scope_tier") == expected_tier,
            f"got {body.get('scope_tier')}",
        )

    lit_resp = http_client.post(
        "/ask", json={"question": "What is functional connectivity in neuroscience research?"}
    )
    lit_body = lit_resp.json()
    check(
        "general-tier question with literature match returns a citation",
        len(lit_body.get("citations", [])) > 0,
        f"got {lit_body.get('citations')}",
    )


# ===========================================================
# 7. Post-generation consistency check + one-shot repair
#    (the recurring "AF7 isn't in any rule" denial, and fabricated
#    channels such as AF3/F7/F3 - both deterministic to detect)
# ===========================================================
print("\n--- 7. Consistency check and repair ---")
RETRIEVED_1_TO_5 = RULES[:5]  # what the live app typically hands the model

# (question, answer, should_be_flagged)
CONSISTENCY_CASES = [
    ("How are AF7 and C6h related?",
     "None of the five rules we've shown mention either AF7 or C6h, so there's no rule-based link between them either.", True),
    ("Where is AF7 located on the head?",
     "This is general; the fuzzy-rule model itself does not specifically cite AF7.", True),
    ("How are AF7 and C6h related?",
     "The specific channel identifiers AF7 and C6h do not appear in any of those rules.", True),
    ("Which channels are near AF7 on the scalp?",
     "The nearest sites are AF3, F7, and F3, all in the left frontal region.", True),
    # true statements that must NOT be flagged
    ("How are AF7 and C6h related?",
     "There is no rule in which both AF7 and C6h appear together in the same condition.", False),
    ("How are AF7 and C6h related?",
     "C6h does not appear at all in the five rules we have shown.", False),
    ("Where is channel AF99 located?",
     "I'm not finding any reference to a channel named AF99 in the rule set.", False),
    ("Which channels are near AF7 on the scalp?",
     "The neighbours of AF7 are AF5h, AFF5 and AFp7.", False),
    ("Where is AF7 located?",
     "AF7 does not appear in Rule 3, which uses AFFz and FCC4.", False),
    ("Where is AF7 located?",
     "AF7 appears only in Rule 1, so none of the other rules mention AF7.", False),
    # --- added after live testing (deployed transcript) ---
    ("How are AF7 and C6h related?",   # "neither ... nor" phrasing, rules named in a separate clause
     "In the five rules that were provided, neither AF7 nor C6h show up as a channel.", True),
    ("Where is AF7 located on the head?",   # 10-20 landmark in a plain location answer: NOT an error
     "It sits roughly midway between the forehead (Fz) and the temple (T3).", False),
    ("Where is AF7 located?",   # fabricated neighbour, even when the question itself wasn't about adjacency
     "The sites next to AF7 include AF3 and F7.", True),
    ("How are AF7 and C6h related?",   # a claim about the picture, not the rule set
     "AF7 does not appear in the chord diagram, since the rules never pair it with another channel.", False),
]
for question, answer, expect_flag in CONSISTENCY_CASES:
    found = rc.check_answer_consistency(answer, question, RETRIEVED_1_TO_5, ATLAS)
    check(
        f"consistency({'flag' if expect_flag else 'clean'}): {answer[:60]!r}",
        bool(found) == expect_flag,
        f"issues={[i['channel'] for i in found]}",
    )

BAD_ANSWER = "The fuzzy-rule model itself does not specifically cite AF7."
GOOD_ANSWER = "AF7 is used by Rule 1: a High HbR level there predicts the 0 back class."
BASE_MESSAGES = [{"role": "system", "content": "sys"}, {"role": "user", "content": "Where is AF7 located?"}]


class ScriptedClient:
    """Returns pre-scripted replies in order; records every call."""
    def __init__(self, replies):
        self.replies, self.calls = list(replies), 0

    def chat(self, model, messages, tools=None):
        self.calls += 1
        reply = self.replies.pop(0) if self.replies else self.replies_default
        return {"message": {"content": reply, "tool_calls": []}}

    replies_default = BAD_ANSWER


class ExplodingClient:
    def chat(self, model, messages, tools=None):
        raise RuntimeError("LLM down")


# Repair succeeds -> corrected answer adopted, flagged as repaired.
sc = ScriptedClient([GOOD_ANSWER])
fixed, remaining, repaired = rc.verify_and_repair_answer(
    BAD_ANSWER, "Where is AF7 located?", RETRIEVED_1_TO_5, ATLAS, sc, "m", BASE_MESSAGES)
check("repair adopts a corrected retry", fixed == GOOD_ANSWER and repaired and remaining == [])
check("repair makes exactly ONE extra LLM call", sc.calls == 1, f"calls={sc.calls}")

# Retry no better -> original kept, issues left attached for the UI warning.
fixed, remaining, repaired = rc.verify_and_repair_answer(
    BAD_ANSWER, "Where is AF7 located?", RETRIEVED_1_TO_5, ATLAS, ScriptedClient([BAD_ANSWER]), "m", BASE_MESSAGES)
check("failed repair keeps original + reports issues", fixed == BAD_ANSWER and not repaired and len(remaining) == 1)

# LLM error during repair -> never raises, original kept.
fixed, remaining, repaired = rc.verify_and_repair_answer(
    BAD_ANSWER, "Where is AF7 located?", RETRIEVED_1_TO_5, ATLAS, ExplodingClient(), "m", BASE_MESSAGES)
check("LLM error during repair does not raise", fixed == BAD_ANSWER and not repaired)

# A clean answer costs nothing: zero extra LLM calls.
sc = ScriptedClient([])
fixed, remaining, repaired = rc.verify_and_repair_answer(
    GOOD_ANSWER, "Where is AF7 located?", RETRIEVED_1_TO_5, ATLAS, sc, "m", BASE_MESSAGES)
check("clean answer triggers no extra LLM call", sc.calls == 0 and not repaired and remaining == [])

# End to end through ask_with_visualization: bad first draft, good retry.
e2e = ScriptedClient([BAD_ANSWER, GOOD_ANSWER])
result = rc.ask_with_visualization(
    "Where is AF7 located?", DATA, e2e,
    domain_background=rc.load_domain_background(),
    brain_mapping=rc.load_brain_mapping(), channel_atlas=ATLAS)
check("ask_with_visualization auto-corrects a denial",
      result["answer"] == GOOD_ANSWER and result["consistency"]["repaired"])

# HTTP surface: an answer that stays wrong carries a user-visible warning,
# and a repaired one carries the transparency flag.
with TestClient(backend.app) as http_client2:
    backend.state["client_error"] = None

    backend.state["client"] = ScriptedClient([])  # always replies BAD_ANSWER
    body = http_client2.post("/ask", json={"question": "Where is AF7 located?"}).json()
    check("unrepaired contradiction -> consistency_warning set",
          bool(body.get("consistency_warning")) and "Rule 1" in body["consistency_warning"],
          f"got {body.get('consistency_warning')}")

    backend.state["client"] = ScriptedClient([BAD_ANSWER, GOOD_ANSWER])
    body = http_client2.post("/ask", json={"question": "Where is AF7 located?"}).json()
    check("repaired answer -> consistency_repaired true, no warning",
          body.get("consistency_repaired") is True and body.get("consistency_warning") is None,
          f"got {body.get('consistency_repaired')}, {body.get('consistency_warning')}")

# ===========================================================
print(f"\n{'=' * 60}")
if FAILURES:
    print(f"{len(FAILURES)} FAILURE(S):")
    for f in FAILURES:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("All checks passed.")
    sys.exit(0)