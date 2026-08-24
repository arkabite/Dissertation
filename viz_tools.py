# """
# viz_tools.py

# Tool-calling schemas + deterministic visualization payload builders for
# the fuzzy rule explainer.

# DESIGN PRINCIPLE - the LLM decides WHETHER and WHICH, never WHAT:
# The gpt-oss tool call tells us the model *wants* a topomap or a chord
# diagram, and gives a rough hint of which channels the user meant (e.g.
# "AF7"). It does NOT tell us the actual levels, positions, regions, or
# which rules those channels came from - all of that is derived here,
# deterministically, from the SAME `retrieved` rule list already used to
# ground the text answer. This matters for two reasons:
#   1. The LLM's channel arguments come back in whatever shorthand it
#      picked (e.g. "AF7"), not the project's actual tmb_s1_ch*/tmb_s2_ch*
#      feature names - they need resolving against real feature names
#      before they mean anything.
#   2. Trusting LLM-generated arguments as the visualization's data would
#      violate the same "only ground truth from the rules" principle that
#      rag_core.py's system prompt already enforces for text answers - a
#      hallucinated channel in a picture is just as ungrounded as one in a
#      sentence.
# If the LLM's requested channels don't match anything in the retrieved
# rules, we fall back to visualizing every channel that IS in the
# retrieved rules, rather than rendering nothing or trusting the guess.
# """

# import json
# import re
# from pathlib import Path

# CHANNEL_ATLAS_PATH = "channel_atlas.json"
# FEATURE_PATTERN = re.compile(r"tmb_(s1|s2)_ch([a-zA-Z]+\d*[a-zA-Z]*|[a-zA-Z]+z)")
# RULE_TOKEN_PATTERN = re.compile(
#     r"tmb_(s1|s2)_ch([a-zA-Z0-9]+)\s+IS\s+(Low|Medium|High)", re.IGNORECASE
# )

# SIGNAL_LABELS = {"s1": "HbO", "s2": "HbR"}
# # Small visual offset so HbO/HbR points at the same physical electrode
# # don't sit exactly on top of each other - same idea as the notebook's
# # own offset_vector for s2 signals in section 6.
# SIGNAL_OFFSET = {"s1": -0.02, "s2": 0.02}


# def load_channel_atlas(path: str = CHANNEL_ATLAS_PATH) -> dict:
#     with open(path, "r") as f:
#         return json.load(f)


# # ---------------------------------------------------------
# # 1. Tool schemas for gpt-oss function calling
# # ---------------------------------------------------------
# VIZ_TOOLS = [
#     {
#         "type": "function",
#         "function": {
#             "name": "show_topomap",
#             "description": (
#                 "Display a scalp topomap: the physical location of one or "
#                 "more fNIRS channels, coloured by their Low/Medium/High "
#                 "activation level in the relevant rule(s). Call this when "
#                 "the user asks WHERE a channel is, wants to see channel "
#                 "locations, or asks about a single channel's activation "
#                 "pattern spatially."
#             ),
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "channels": {
#                         "type": "array",
#                         "items": {"type": "string"},
#                         "description": (
#                             "Electrode names mentioned or implied by the "
#                             "question, e.g. ['AF7', 'C6h']. Best guess is "
#                             "fine - the backend resolves these against the "
#                             "actual rules."
#                         ),
#                     }
#                 },
#                 "required": ["channels"],
#             },
#         },
#     },
#     {
#         "type": "function",
#         "function": {
#             "name": "show_chord_diagram",
#             "description": (
#                 "Display a chord/connectivity diagram showing which "
#                 "channels co-occur together within the same rule(s). Call "
#                 "this when the user asks how channels RELATE or CONNECT, "
#                 "not when they ask about a single channel's location."
#             ),
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "channels": {
#                         "type": "array",
#                         "items": {"type": "string"},
#                         "description": (
#                             "Electrode names involved in the relationship "
#                             "the user is asking about, e.g. ['AF7', 'C6h']."
#                         ),
#                     }
#                 },
#                 "required": ["channels"],
#             },
#         },
#     },
# ]


# # ---------------------------------------------------------
# # 2. Parsing rule antecedents into structured per-channel facts
# # ---------------------------------------------------------
# def parse_rule_channels(antecedent: str) -> list:
#     """Turns 'tmb_s2_chAF7 IS High AND tmb_s1_chC6h IS Medium' into
#     [{"feature": "tmb_s2_chAF7", "bare_channel": "AF7", "signal": "s2",
#       "level": "High"}, ...]."""
#     out = []
#     for signal, region_hemi, level in RULE_TOKEN_PATTERN.findall(antecedent):
#         out.append({
#             "feature": f"tmb_{signal}_ch{region_hemi}",
#             "bare_channel": region_hemi,
#             "signal": signal,
#             "level": level.title(),
#         })
#     return out


# def bare_channel_from_feature(feature: str) -> str:
#     m = FEATURE_PATTERN.match(feature)
#     return m.group(2) if m else feature


# # ---------------------------------------------------------
# # 3. Resolving the LLM's tool-call arguments against real rule data
# # ---------------------------------------------------------
# def resolve_requested_channels(llm_channels: list, retrieved: list) -> set:
#     """Matches whatever the LLM said (e.g. 'AF7', 'af7', 'tmb_s2_chAF7')
#     against the bare channel names actually present in the retrieved
#     rules. Falls back to ALL channels in the retrieved rules if nothing
#     matches - never trusts an unresolvable guess, never renders empty."""
#     available = set()
#     for rule in retrieved:
#         for tok in parse_rule_channels(rule.get("antecedent", "")):
#             available.add(tok["bare_channel"])

#     if not llm_channels:
#         return available

#     requested_norm = set()
#     for raw in llm_channels:
#         bare = bare_channel_from_feature(raw) if raw.lower().startswith("tmb_") else raw
#         requested_norm.add(bare.lower())

#     matched = {ch for ch in available if ch.lower() in requested_norm}
#     return matched if matched else available


# # ---------------------------------------------------------
# # 4. Deterministic payload builders
# # ---------------------------------------------------------
# def build_topomap_payload(retrieved: list, requested_channels: set, atlas: dict) -> list:
#     """One point per (channel, signal, rule) that actually appears in the
#     retrieved rules and was requested - positions come only from the
#     static atlas, never from the LLM."""
#     points = []
#     seen = set()
#     for rule in retrieved:
#         for tok in parse_rule_channels(rule.get("antecedent", "")):
#             if tok["bare_channel"] not in requested_channels:
#                 continue
#             key = (tok["feature"], rule["rule_id"])
#             if key in seen:
#                 continue
#             seen.add(key)

#             atlas_entry = atlas.get(tok["bare_channel"])
#             if atlas_entry is None:
#                 continue  # unmapped channel - skip rather than guess a position

#             points.append({
#                 "feature": tok["feature"],
#                 "bare_channel": tok["bare_channel"],
#                 "signal": SIGNAL_LABELS[tok["signal"]],
#                 "level": tok["level"],
#                 "x": atlas_entry["x"] + SIGNAL_OFFSET[tok["signal"]],
#                 "y": atlas_entry["y"],
#                 "functional_region": atlas_entry["functional_region"],
#                 "hemisphere": atlas_entry["hemisphere"],
#                 "rule_id": rule["rule_id"],
#                 "consequent": rule["consequent"],
#                 "accuracy": rule.get("accuracy"),
#             })
#     return points


# def build_chord_payload(retrieved: list, requested_channels: set, atlas: dict) -> dict:
#     """Nodes = unique bare channels involved; edges = channel pairs that
#     co-occur within the SAME rule's antecedent (this is what "connected"
#     means here - literal AND-co-occurrence in a fired rule, not a
#     measured physiological connectivity claim)."""
#     nodes = {}
#     edges = []

#     for rule in retrieved:
#         toks = [
#             t for t in parse_rule_channels(rule.get("antecedent", ""))
#             if t["bare_channel"] in requested_channels
#         ]
#         bare_in_rule = sorted({t["bare_channel"] for t in toks})

#         for t in toks:
#             if t["bare_channel"] not in nodes:
#                 atlas_entry = atlas.get(t["bare_channel"], {})
#                 nodes[t["bare_channel"]] = {
#                     "channel": t["bare_channel"],
#                     "functional_region": atlas_entry.get("functional_region", "unknown"),
#                     "hemisphere": atlas_entry.get("hemisphere", "unknown"),
#                 }

#         for i in range(len(bare_in_rule)):
#             for j in range(i + 1, len(bare_in_rule)):
#                 edges.append({
#                     "source": bare_in_rule[i],
#                     "target": bare_in_rule[j],
#                     "rule_id": rule["rule_id"],
#                     "consequent": rule["consequent"],
#                     "dominance_score": rule.get("dominance_score"),
#                     "accuracy": rule.get("accuracy"),
#                 })

#     return {"nodes": list(nodes.values()), "edges": edges}


# # ---------------------------------------------------------
# # 5. Compact text summary handed BACK to the LLM for the narration turn
# #    (not the raw payload - the LLM doesn't need 30 fields per point,
# #    just enough to talk about what's on screen accurately)
# # ---------------------------------------------------------
# def summarize_topomap_for_llm(points: list) -> str:
#     if not points:
#         return "No matching channels were found to plot."
#     lines = [
#         f"- {p['bare_channel']} ({p['signal']}): {p['level']}, "
#         f"from rule {p['rule_id']} (predicts {p['consequent']})"
#         for p in points
#     ]
#     return "Topomap rendered with these points:\n" + "\n".join(lines)


# def summarize_chord_for_llm(chord: dict) -> str:
#     if not chord["edges"]:
#         return "No co-occurring channel pairs were found to connect."
#     lines = [
#         f"- {e['source']} <-> {e['target']} (rule {e['rule_id']}, "
#         f"predicts {e['consequent']}, accuracy={e.get('accuracy', 'n/a')})"
#         for e in chord["edges"]
#     ]
#     return "Chord diagram rendered with these connections:\n" + "\n".join(lines)

"""
viz_tools.py

Tool-calling schemas + deterministic visualization payload builders for
the fuzzy rule explainer.

DESIGN PRINCIPLE - the LLM decides WHETHER and WHICH, never WHAT:
The gpt-oss tool call tells us the model *wants* a topomap or a chord
diagram, and gives a rough hint of which channels the user meant (e.g.
"AF7"). It does NOT tell us the actual levels, positions, regions, or
which rules those channels came from - all of that is derived here,
deterministically, from the SAME `retrieved` rule list already used to
ground the text answer. This matters for two reasons:
  1. The LLM's channel arguments come back in whatever shorthand it
     picked (e.g. "AF7"), not the project's actual tmb_s1_ch*/tmb_s2_ch*
     feature names - they need resolving against real feature names
     before they mean anything.
  2. Trusting LLM-generated arguments as the visualization's data would
     violate the same "only ground truth from the rules" principle that
     rag_core.py's system prompt already enforces for text answers - a
     hallucinated channel in a picture is just as ungrounded as one in a
     sentence.
If the LLM's requested channels don't match anything in the retrieved
rules, we fall back to visualizing every channel that IS in the
retrieved rules, rather than rendering nothing or trusting the guess.
"""

import json
import re
from pathlib import Path

CHANNEL_ATLAS_PATH = "channel_atlas.json"
FEATURE_PATTERN = re.compile(r"tmb_(s1|s2)_ch([a-zA-Z]+\d*[a-zA-Z]*|[a-zA-Z]+z)")
RULE_TOKEN_PATTERN = re.compile(
    r"tmb_(s1|s2)_ch([a-zA-Z0-9]+)\s+IS\s+(Low|Medium|High)", re.IGNORECASE
)

SIGNAL_LABELS = {"s1": "HbO", "s2": "HbR"}
# Small visual offset so HbO/HbR points at the same physical electrode
# don't sit exactly on top of each other - same idea as the notebook's
# own offset_vector for s2 signals in section 6.
SIGNAL_OFFSET = {"s1": -0.02, "s2": 0.02}


def load_channel_atlas(path: str = CHANNEL_ATLAS_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------
# 1. Tool schemas for gpt-oss function calling
# ---------------------------------------------------------
VIZ_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "show_topomap",
            "description": (
                "Display a scalp topomap: the physical location of one or "
                "more fNIRS channels, coloured by their Low/Medium/High "
                "activation level in the relevant rule(s). Call this when "
                "the user asks WHERE a channel is, wants to see channel "
                "locations, or asks about a single channel's activation "
                "pattern spatially."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Electrode names mentioned or implied by the "
                            "question, e.g. ['AF7', 'C6h']. Best guess is "
                            "fine - the backend resolves these against the "
                            "actual rules."
                        ),
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
                "Display a chord/connectivity diagram showing which "
                "channels co-occur together within the same rule(s). Call "
                "this when the user asks how channels RELATE or CONNECT, "
                "not when they ask about a single channel's location."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Electrode names involved in the relationship "
                            "the user is asking about, e.g. ['AF7', 'C6h']."
                        ),
                    }
                },
                "required": ["channels"],
            },
        },
    },
]


# ---------------------------------------------------------
# 2. Parsing rule antecedents into structured per-channel facts
# ---------------------------------------------------------
def parse_rule_channels(antecedent: str) -> list:
    """Turns 'tmb_s2_chAF7 IS High AND tmb_s1_chC6h IS Medium' into
    [{"feature": "tmb_s2_chAF7", "bare_channel": "AF7", "signal": "s2",
      "level": "High"}, ...]."""
    out = []
    for signal, region_hemi, level in RULE_TOKEN_PATTERN.findall(antecedent):
        out.append({
            "feature": f"tmb_{signal}_ch{region_hemi}",
            "bare_channel": region_hemi,
            "signal": signal,
            "level": level.title(),
        })
    return out


def bare_channel_from_feature(feature: str) -> str:
    m = FEATURE_PATTERN.match(feature)
    return m.group(2) if m else feature


# ---------------------------------------------------------
# 3. Resolving the LLM's tool-call arguments against real rule data
# ---------------------------------------------------------
def resolve_requested_channels(llm_channels: list, retrieved: list) -> set:
    """Matches whatever the LLM said (e.g. 'AF7', 'af7', 'tmb_s2_chAF7')
    against the bare channel names actually present in the retrieved
    rules. Falls back to ALL channels in the retrieved rules if nothing
    matches - never trusts an unresolvable guess, never renders empty."""
    available = set()
    for rule in retrieved:
        for tok in parse_rule_channels(rule.get("antecedent", "")):
            available.add(tok["bare_channel"])

    if not llm_channels:
        return available

    requested_norm = set()
    for raw in llm_channels:
        bare = bare_channel_from_feature(raw) if raw.lower().startswith("tmb_") else raw
        requested_norm.add(bare.lower())

    matched = {ch for ch in available if ch.lower() in requested_norm}
    return matched if matched else available


# ---------------------------------------------------------
# 4. Deterministic payload builders
# ---------------------------------------------------------
def build_topomap_payload(retrieved: list, requested_channels: set, atlas: dict) -> list:
    """One point per (channel, signal, rule) that actually appears in the
    retrieved rules and was requested - positions come only from the
    static atlas, never from the LLM."""
    points = []
    seen = set()
    for rule in retrieved:
        for tok in parse_rule_channels(rule.get("antecedent", "")):
            if tok["bare_channel"] not in requested_channels:
                continue
            key = (tok["feature"], rule["rule_id"])
            if key in seen:
                continue
            seen.add(key)

            atlas_entry = atlas["channels"].get(tok["bare_channel"])
            if atlas_entry is None:
                continue  # unmapped channel - skip rather than guess a position

            points.append({
                "feature": tok["feature"],
                "bare_channel": tok["bare_channel"],
                "signal": SIGNAL_LABELS[tok["signal"]],
                "level": tok["level"],
                "x": atlas_entry["x"] + SIGNAL_OFFSET[tok["signal"]],
                "y": atlas_entry["y"],
                "x3d": atlas_entry.get("x3d"),
                "y3d": atlas_entry.get("y3d"),
                "z3d": atlas_entry.get("z3d"),
                "functional_region": atlas_entry["functional_region"],
                "hemisphere": atlas_entry["hemisphere"],
                "rule_id": rule["rule_id"],
                "consequent": rule["consequent"],
                "accuracy": rule.get("accuracy"),
            })
    return points


def build_chord_payload(retrieved: list, requested_channels: set, atlas: dict) -> dict:
    """Nodes = unique bare channels involved; edges = channel pairs that
    co-occur within the SAME rule's antecedent (this is what "connected"
    means here - literal AND-co-occurrence in a fired rule, not a
    measured physiological connectivity claim)."""
    nodes = {}
    edges = []

    for rule in retrieved:
        toks = [
            t for t in parse_rule_channels(rule.get("antecedent", ""))
            if t["bare_channel"] in requested_channels
        ]
        bare_in_rule = sorted({t["bare_channel"] for t in toks})

        for t in toks:
            if t["bare_channel"] not in nodes:
                atlas_entry = atlas["channels"].get(t["bare_channel"], {})
                nodes[t["bare_channel"]] = {
                    "channel": t["bare_channel"],
                    "functional_region": atlas_entry.get("functional_region", "unknown"),
                    "hemisphere": atlas_entry.get("hemisphere", "unknown"),
                }

        for i in range(len(bare_in_rule)):
            for j in range(i + 1, len(bare_in_rule)):
                edges.append({
                    "source": bare_in_rule[i],
                    "target": bare_in_rule[j],
                    "rule_id": rule["rule_id"],
                    "consequent": rule["consequent"],
                    "dominance_score": rule.get("dominance_score"),
                    "accuracy": rule.get("accuracy"),
                })

    return {"nodes": list(nodes.values()), "edges": edges}


# ---------------------------------------------------------
# 5. Compact text summary handed BACK to the LLM for the narration turn
#    (not the raw payload - the LLM doesn't need 30 fields per point,
#    just enough to talk about what's on screen accurately)
# ---------------------------------------------------------
def summarize_topomap_for_llm(points: list) -> str:
    if not points:
        return "No matching channels were found to plot."
    lines = [
        f"- {p['bare_channel']} ({p['signal']}): {p['level']}, "
        f"from rule {p['rule_id']} (predicts {p['consequent']})"
        for p in points
    ]
    return "Topomap rendered with these points:\n" + "\n".join(lines)


def summarize_chord_for_llm(chord: dict) -> str:
    if not chord["edges"]:
        return "No co-occurring channel pairs were found to connect."
    lines = [
        f"- {e['source']} <-> {e['target']} (rule {e['rule_id']}, "
        f"predicts {e['consequent']}, accuracy={e.get('accuracy', 'n/a')})"
        for e in chord["edges"]
    ]
    return "Chord diagram rendered with these connections:\n" + "\n".join(lines)