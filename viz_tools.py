

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
#                 "the user asks WHERE a specific channel that appears in a "
#                 "rule is, or asks about that channel's activation pattern "
#                 "spatially. This does NOT answer 'which channels are near "
#                 "X' - use show_channel_neighbors for physical adjacency "
#                 "questions instead."
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
#                 "channels co-occur together WITHIN THE SAME RULE's "
#                 "antecedent (a statistical/logical co-occurrence, from the "
#                 "fitted rule set). Call this when the user asks how "
#                 "channels appearing in the rules RELATE or CONNECT in the "
#                 "model's decision logic. This does NOT model physical/"
#                 "spatial proximity on the scalp - a question like 'which "
#                 "channels are near AF7' is about physical adjacency, not "
#                 "rule co-occurrence, so use show_channel_neighbors for "
#                 "that instead, even though both use the word 'near' or "
#                 "'connect' loosely in everyday English."
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
#     {
#         "type": "function",
#         "function": {
#             "name": "show_channel_neighbors",
#             "description": (
#                 "Display which channels sit physically NEAR/ADJACENT TO a "
#                 "given channel on the actual scalp montage used by this "
#                 "study - a purely spatial fact from the study's own "
#                 "electrode layout, independent of any rule. Call this when "
#                 "the user asks which channels are near, next to, "
#                 "surrounding, or adjacent to a specific channel, or "
#                 "otherwise asks a physical/anatomical-proximity question "
#                 "about one channel's neighbors. Do NOT guess neighbors "
#                 "from general 10-10/10-5 EEG knowledge or invent channel "
#                 "names - this tool returns the ACTUAL neighbor list from "
#                 "this study's own montage graph, which is the only "
#                 "grounded source for this question."
#             ),
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "channel": {
#                         "type": "string",
#                         "description": (
#                             "The single electrode name the user is asking "
#                             "about the physical neighbors of, e.g. 'AF7'."
#                         ),
#                     }
#                 },
#                 "required": ["channel"],
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


# def resolve_single_channel(llm_channel: str, atlas: dict) -> str:
#     """Matches a single raw LLM-provided channel string (e.g. 'AF7', 'af7',
#     'tmb_s2_chAF7') against the ACTUAL channel codes in this study's atlas -
#     not against the retrieved rules, since a neighbor question is about the
#     physical montage, independent of which rules happened to be retrieved.
#     Returns the canonical atlas key, or "" if there's no real match (never
#     guesses a channel that isn't actually in this montage)."""
#     if not llm_channel:
#         return ""
#     bare = bare_channel_from_feature(llm_channel) if llm_channel.lower().startswith("tmb_") else llm_channel
#     bare_lower = bare.strip().lower()
#     for code in atlas.get("channels", {}):
#         if code.lower() == bare_lower:
#             return code
#     return ""


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

#             atlas_entry = atlas["channels"].get(tok["bare_channel"])
#             if atlas_entry is None:
#                 continue  # unmapped channel - skip rather than guess a position

#             points.append({
#                 "feature": tok["feature"],
#                 "bare_channel": tok["bare_channel"],
#                 "signal": SIGNAL_LABELS[tok["signal"]],
#                 "level": tok["level"],
#                 "x": atlas_entry["x"] + SIGNAL_OFFSET[tok["signal"]],
#                 "y": atlas_entry["y"],
#                 "x3d": atlas_entry.get("x3d"),
#                 "y3d": atlas_entry.get("y3d"),
#                 "z3d": atlas_entry.get("z3d"),
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
#                 atlas_entry = atlas["channels"].get(t["bare_channel"], {})
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


# def build_neighbors_payload(channel: str, atlas: dict) -> dict:
#     """Physical-adjacency payload for one channel, built ENTIRELY from the
#     atlas's own mesh_edges - no rule data, no LLM guessing. This is the
#     grounded answer to "which channels are near X", as distinct from
#     build_chord_payload's rule-co-occurrence answer to "which channels are
#     connected to X". `channel` should already be resolved via
#     resolve_single_channel(); an unresolved ("") channel returns found=False
#     rather than guessing."""
#     channels = atlas.get("channels", {})
#     if not channel or channel not in channels:
#         return {"channel": channel or None, "found": False, "position": None, "neighbors": []}

#     entry = channels[channel]
#     neighbor_codes = sorted({
#         e["b"] for e in atlas.get("mesh_edges", []) if e["a"] == channel
#     } | {
#         e["a"] for e in atlas.get("mesh_edges", []) if e["b"] == channel
#     })

#     neighbors = []
#     for code in neighbor_codes:
#         n_entry = channels.get(code)
#         if n_entry is None:
#             continue  # atlas mesh_edges referencing an unmapped code - skip rather than guess
#         neighbors.append({
#             "channel": code,
#             "x": n_entry["x"],
#             "y": n_entry["y"],
#             "x3d": n_entry.get("x3d"),
#             "y3d": n_entry.get("y3d"),
#             "z3d": n_entry.get("z3d"),
#             "functional_region": n_entry.get("functional_region", "unknown"),
#             "hemisphere": n_entry.get("hemisphere", "unknown"),
#         })

#     return {
#         "channel": channel,
#         "found": True,
#         "position": {
#             "x": entry["x"], "y": entry["y"],
#             "x3d": entry.get("x3d"), "y3d": entry.get("y3d"), "z3d": entry.get("z3d"),
#         },
#         "neighbors": neighbors,
#     }


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
#         if chord.get("nodes"):
#             # The channel(s) DO appear in the retrieved rules - they just
#             # never co-occur with another channel in the same rule's
#             # antecedent. Without this distinction spelled out, the LLM
#             # has been observed over-extrapolating "no edges" into the
#             # stronger, false claim "this channel isn't in any rule" -
#             # give it the actual node list so it can't make that leap.
#             names = ", ".join(sorted(n["channel"] for n in chord["nodes"]))
#             return (
#                 "No co-occurring channel pairs were found - none of these "
#                 "channels appear together with another channel in the "
#                 f"same rule's antecedent. However, these channel(s) DO "
#                 f"appear individually in the retrieved rules (each alone, "
#                 f"or only with channels outside the requested set): {names}. "
#                 "State plainly that no connection exists between them, "
#                 "not that the channel is absent from the rule set."
#             )
#         return "No matching channels were found in the retrieved rules to connect."
#     lines = [
#         f"- {e['source']} <-> {e['target']} (rule {e['rule_id']}, "
#         f"predicts {e['consequent']}, accuracy={e.get('accuracy', 'n/a')})"
#         for e in chord["edges"]
#     ]
#     return "Chord diagram rendered with these connections:\n" + "\n".join(lines)


# def summarize_neighbors_for_llm(payload: dict) -> str:
#     if not payload.get("found"):
#         requested = payload.get("channel") or "the requested name"
#         return (
#             f"'{requested}' does not match any channel in this study's "
#             "montage - do not invent or guess neighbor channels; state "
#             "plainly that this channel isn't part of the montage."
#         )
#     if not payload["neighbors"]:
#         return (
#             f"{payload['channel']} has no recorded neighbors in this "
#             "study's montage graph - say plainly that no adjacency data "
#             "is available for it, do not guess neighbors from general "
#             "10-10/10-5 EEG knowledge."
#         )
#     names = ", ".join(n["channel"] for n in payload["neighbors"])
#     return (
#         f"Channel-neighbor map rendered for {payload['channel']}. Its "
#         f"physically adjacent channels in this study's own montage graph "
#         f"(NOT a rule-derived relationship) are: {names}. These are the "
#         "ONLY channels you may state as neighbors - do not add any "
#         "channel name not in this list, even if it seems plausible from "
#         "general EEG/fNIRS layout knowledge."
#     )

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
    with open(path, "r", encoding="utf-8") as f:
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
                "the user asks WHERE a channel that appears in a rule is, "
                "asks about a channel's activation pattern spatially, OR "
                "asks to see the WHOLE/FULL/ENTIRE/ALL-channel map across "
                "every rule (e.g. 'show me the whole channel map', 'show "
                "every channel') - for that case, pass every channel "
                "mentioned across the currently retrieved rules, not just "
                "one. Do not refuse a 'whole map' request; this tool "
                "supports any number of channels at once, including all of "
                "them. This does NOT answer 'which channels are near X' - "
                "use show_channel_neighbors for physical adjacency "
                "questions instead."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channels": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Electrode names mentioned or implied by the "
                            "question, e.g. ['AF7', 'C6h']. For a 'whole "
                            "channel map' request, list every channel "
                            "mentioned across all currently retrieved "
                            "rules. Best guess is fine either way - the "
                            "backend resolves these against the actual "
                            "rules."
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
                "channels co-occur together WITHIN THE SAME RULE's "
                "antecedent (a statistical/logical co-occurrence, from the "
                "fitted rule set). Call this when the user asks how "
                "channels appearing in the rules RELATE or CONNECT in the "
                "model's decision logic. This does NOT model physical/"
                "spatial proximity on the scalp - a question like 'which "
                "channels are near AF7' is about physical adjacency, not "
                "rule co-occurrence, so use show_channel_neighbors for "
                "that instead, even though both use the word 'near' or "
                "'connect' loosely in everyday English."
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
    {
        "type": "function",
        "function": {
            "name": "show_channel_neighbors",
            "description": (
                "Display which channels sit physically NEAR/ADJACENT TO a "
                "given channel on the actual scalp montage used by this "
                "study - a purely spatial fact from the study's own "
                "electrode layout, independent of any rule. Call this when "
                "the user asks which channels are near, next to, "
                "surrounding, or adjacent to a specific channel, or "
                "otherwise asks a physical/anatomical-proximity question "
                "about one channel's neighbors. Do NOT guess neighbors "
                "from general 10-10/10-5 EEG knowledge or invent channel "
                "names - this tool returns the ACTUAL neighbor list from "
                "this study's own montage graph, which is the only "
                "grounded source for this question."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "channel": {
                        "type": "string",
                        "description": (
                            "The single electrode name the user is asking "
                            "about the physical neighbors of, e.g. 'AF7'."
                        ),
                    }
                },
                "required": ["channel"],
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


def resolve_single_channel(llm_channel: str, atlas: dict) -> str:
    """Matches a single raw LLM-provided channel string (e.g. 'AF7', 'af7',
    'tmb_s2_chAF7') against the ACTUAL channel codes in this study's atlas -
    not against the retrieved rules, since a neighbor question is about the
    physical montage, independent of which rules happened to be retrieved.
    Returns the canonical atlas key, or "" if there's no real match (never
    guesses a channel that isn't actually in this montage)."""
    if not llm_channel:
        return ""
    bare = bare_channel_from_feature(llm_channel) if llm_channel.lower().startswith("tmb_") else llm_channel
    bare_lower = bare.strip().lower()
    for code in atlas.get("channels", {}):
        if code.lower() == bare_lower:
            return code
    return ""


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


def build_neighbors_payload(channel: str, atlas: dict) -> dict:
    """Physical-adjacency payload for one channel, built ENTIRELY from the
    atlas's own mesh_edges - no rule data, no LLM guessing. This is the
    grounded answer to "which channels are near X", as distinct from
    build_chord_payload's rule-co-occurrence answer to "which channels are
    connected to X". `channel` should already be resolved via
    resolve_single_channel(); an unresolved ("") channel returns found=False
    rather than guessing."""
    channels = atlas.get("channels", {})
    if not channel or channel not in channels:
        return {"channel": channel or None, "found": False, "position": None, "neighbors": []}

    entry = channels[channel]
    neighbor_codes = sorted({
        e["b"] for e in atlas.get("mesh_edges", []) if e["a"] == channel
    } | {
        e["a"] for e in atlas.get("mesh_edges", []) if e["b"] == channel
    })

    neighbors = []
    for code in neighbor_codes:
        n_entry = channels.get(code)
        if n_entry is None:
            continue  # atlas mesh_edges referencing an unmapped code - skip rather than guess
        neighbors.append({
            "channel": code,
            "x": n_entry["x"],
            "y": n_entry["y"],
            "x3d": n_entry.get("x3d"),
            "y3d": n_entry.get("y3d"),
            "z3d": n_entry.get("z3d"),
            "functional_region": n_entry.get("functional_region", "unknown"),
            "hemisphere": n_entry.get("hemisphere", "unknown"),
        })

    return {
        "channel": channel,
        "found": True,
        "position": {
            "x": entry["x"], "y": entry["y"],
            "x3d": entry.get("x3d"), "y3d": entry.get("y3d"), "z3d": entry.get("z3d"),
        },
        "neighbors": neighbors,
    }


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
        if chord.get("nodes"):
            # The channel(s) DO appear in the retrieved rules - they just
            # never co-occur with another channel in the same rule's
            # antecedent. Without this distinction spelled out, the LLM
            # has been observed over-extrapolating "no edges" into the
            # stronger, false claim "this channel isn't in any rule" -
            # give it the actual node list so it can't make that leap.
            names = ", ".join(sorted(n["channel"] for n in chord["nodes"]))
            return (
                "No co-occurring channel pairs were found - none of these "
                "channels appear together with another channel in the "
                f"same rule's antecedent. However, these channel(s) DO "
                f"appear individually in the retrieved rules (each alone, "
                f"or only with channels outside the requested set): {names}. "
                "State plainly that no connection exists between them, "
                "not that the channel is absent from the rule set."
            )
        return "No matching channels were found in the retrieved rules to connect."
    lines = [
        f"- {e['source']} <-> {e['target']} (rule {e['rule_id']}, "
        f"predicts {e['consequent']}, accuracy={e.get('accuracy', 'n/a')})"
        for e in chord["edges"]
    ]
    return "Chord diagram rendered with these connections:\n" + "\n".join(lines)


def summarize_neighbors_for_llm(payload: dict) -> str:
    if not payload.get("found"):
        requested = payload.get("channel") or "the requested name"
        return (
            f"'{requested}' does not match any channel in this study's "
            "montage - do not invent or guess neighbor channels; state "
            "plainly that this channel isn't part of the montage."
        )
    if not payload["neighbors"]:
        return (
            f"{payload['channel']} has no recorded neighbors in this "
            "study's montage graph - say plainly that no adjacency data "
            "is available for it, do not guess neighbors from general "
            "10-10/10-5 EEG knowledge."
        )
    names = ", ".join(n["channel"] for n in payload["neighbors"])
    return (
        f"Channel-neighbor map rendered for {payload['channel']}. Its "
        f"physically adjacent channels in this study's own montage graph "
        f"(NOT a rule-derived relationship) are: {names}. These are the "
        "ONLY channels you may state as neighbors - do not add any "
        "channel name not in this list, even if it seems plausible from "
        "general EEG/fNIRS layout knowledge."
    )