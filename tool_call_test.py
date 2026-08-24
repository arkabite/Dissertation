"""
tool_call_test.py

Standalone de-risking test for gpt-oss:20b-cloud's native tool-calling,
BEFORE building the real visualization router around it.

Run locally (needs OLLAMA_API_KEY set, same as rag_core.py):
    python tool_call_test.py

What this checks:
  1. Does the model call the RIGHT tool when the prompt obviously wants a
     topomap? (channel-location style question)
  2. Does the model call the RIGHT tool when the prompt obviously wants a
     chord/connectivity diagram? (relationship-between-channels question)
  3. Does the model correctly call NO tool for a plain question that isn't
     visual at all? (false-positive check - this matters as much as the
     other two)
  4. What does the raw response actually look like? This is the part the
     online reports flagged as inconsistent across client libraries - we
     want to see it with our own eyes against Ollama's own python client
     specifically, not assume the docs are right for our stack.

Nothing here talks to rag_core.py on purpose - keep this test isolated so
a bug in retrieval/prompting can't muddy whether tool-calling itself works.
"""

import json
import os
from pathlib import Path

from ollama import Client

# ---------------------------------------------------------
# Reuse the same key-loading pattern as rag_core.py so this test reflects
# the real deployment, not a hand-set env var that won't exist later.
# ---------------------------------------------------------
def load_api_key() -> str:
    key = os.environ.get("OLLAMA_API_KEY", "")
    if key:
        return key
    candidates = [
        Path(__file__).resolve().parent / ".streamlit" / "secrets.toml",
        Path(__file__).resolve().parent / "secrets.toml",
    ]
    for path in candidates:
        if not path.exists():
            continue
        import tomllib
        with path.open("rb") as fh:
            data = tomllib.load(fh)
        if data.get("OLLAMA_API_KEY"):
            return str(data["OLLAMA_API_KEY"])
    raise RuntimeError("OLLAMA_API_KEY not set - export it or add it to secrets.toml")


client = Client(host="https://ollama.com",
                 headers={"Authorization": "Bearer " + load_api_key()})

MODEL = "gpt-oss:20b-cloud"

# ---------------------------------------------------------
# Tool schemas shaped like the REAL tools this project needs - not
# generic examples. If tool-calling works here, it'll work in the actual
# router; if it doesn't, better to find out on something this small.
# ---------------------------------------------------------
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "show_topomap",
            "description": (
                "Display a scalp topomap showing the location and "
                "activation level (Low/Medium/High) of specific fNIRS "
                "channels. Use this when the user asks WHERE on the head "
                "a channel or set of channels is, or wants to see "
                "channel locations/activation visually."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Channel codes to display, e.g. ['tmb_s2_chAF7', 'tmb_s1_chAF8']",
                    }
                },
                "required": ["channels"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "show_chord_diagram",
            "description": (
                "Display a chord/connectivity diagram showing how two or "
                "more channels relate to each other within a rule. Use "
                "this when the user asks about a RELATIONSHIP or "
                "CONNECTION between channels, not a single channel's "
                "location."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel_pairs": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "string"},
                        },
                        "description": "List of [channel_a, channel_b] pairs to connect",
                    }
                },
                "required": ["channel_pairs"],
            },
        },
    },
]

# ---------------------------------------------------------
# Test prompts - one per expected outcome
# ---------------------------------------------------------
TEST_CASES = [
    {
        "label": "Should call show_topomap",
        "prompt": "Where on the head is channel AF7 located?",
        "expected_tool": "show_topomap",
    },
    {
        "label": "Should call show_chord_diagram",
        "prompt": "How are channels AF7 and C6h connected in the rules?",
        "expected_tool": "show_chord_diagram",
    },
    {
        "label": "Should call NO tool (false-positive check)",
        "prompt": "How accurate is this model overall?",
        "expected_tool": None,
    },
]

SYSTEM_PROMPT = (
    "You explain a fuzzy rule-based classifier's outputs. You have access "
    "to visualization tools. Only call a tool if the user's question is "
    "genuinely asking to see channel locations or relationships - plain "
    "informational questions should get a text answer with no tool call."
)


def run_case(case: dict) -> None:
    print("=" * 70)
    print(case["label"])
    print(f"Prompt: {case['prompt']!r}")
    print(f"Expected tool: {case['expected_tool']}")
    print("-" * 70)

    response = client.chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": case["prompt"]},
        ],
        tools=TOOLS,
    )

    # Print the RAW message object - this is the part worth eyeballing
    # closely, since this is exactly what a parser has to handle.
    print("Raw response.message:")
    print(json.dumps(dict(response["message"]), indent=2, default=str))

    tool_calls = response["message"].get("tool_calls") or []
    called_names = [tc["function"]["name"] for tc in tool_calls]

    print()
    if case["expected_tool"] is None:
        ok = len(called_names) == 0
        print(f"PASS: {ok}  (called: {called_names or 'none'})")
    else:
        ok = case["expected_tool"] in called_names
        print(f"PASS: {ok}  (called: {called_names or 'none'})")
    print()


if __name__ == "__main__":
    for case in TEST_CASES:
        run_case(case)

    print("=" * 70)
    print("Done. Look for:")
    print("  1. Did each case PASS as expected above?")
    print("  2. Is response.message['tool_calls'] a clean structured list")
    print("     (good - Ollama's client parsed it), or did the call show up")
    print("     as raw text inside response.message['content'] instead")
    print("     (bad - means you'd need to hand-parse the model's own")
    print("     '<|channel|>commentary to=functions.X' format, which some")
    print("     online reports flagged as inconsistent).")
    print("  3. Did arguments (channels / channel_pairs) come back as valid")
    print("     JSON matching the schema, or malformed/partial?")
