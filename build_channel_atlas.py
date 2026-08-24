# """
# build_channel_atlas.py

# Generates channel_atlas.json - the single static source of truth for
# "where does this channel sit on the head, and what region/hemisphere is
# it in" used by BOTH the topomap and chord-diagram visualizations.

# This replaces the two previously-disconnected sources:
#   - brain_mapping.json's region table (used by the LLM prompt's hedged
#     Tier 1/2/3 glossary)
#   - the notebook's hand-typed eeg_location_data table (used only for the
#     PyCirclize chord diagram, with NO hedging language)
# By generating positions from MNE's own standard_1005 montage (the same
# method mne.viz uses internally for its topomaps) and pairing them with
# brain_mapping.json's already-hedged region labels, the diagram and the
# LLM's prose can never disagree about what region a channel is in.

# Run once (or whenever the channel list / brain_mapping.json regions
# change):
#     python build_channel_atlas.py
# Produces:
#     channel_atlas.json

# Positions are 2D, normalised to roughly [-1, 1], generated via the exact
# same projection MNE uses for `raw.plot_sensors(kind='topomap')` - so this
# matches the notebook's section 5 visually, just precomputed once instead
# of recomputed with mne+pyvista on every request.
# """

# import json
# import re

# import mne
# import numpy as np
# from mne.channels.layout import _find_topomap_coords

# # All 36 physical electrode sites in this project's montage (bare names,
# # i.e. without the tmb_s1_ch/tmb_s2_ch prefix - each site carries BOTH an
# # HbO (s1) and HbR (s2) signal at the same physical location).
# CHANNELS = [
#     "AF1", "AF2", "AF5h", "AF6h", "AF7", "AF8", "AFF3h", "AFF4h", "AFF5",
#     "AFF6", "AFFz", "AFp3", "AFp4", "AFp7", "AFp8", "AFpz", "C3h", "C4h",
#     "C5h", "C6h", "CCP3", "CCP4", "CPP3", "CPP4", "FCC3", "FCC4", "P3h",
#     "P4h", "P5h", "P6h", "PO1", "PO2", "POOz", "PPO3", "PPO4", "PPOz",
# ]

# # Region-code / hemisphere-suffix split - same pattern used elsewhere in
# # this project (rag_core.py's pre_translate_rule) so a channel parses the
# # same way everywhere.
# CHANNEL_PATTERN = re.compile(r"^([a-zA-Z]+)(\d+[a-zA-Z]*|z)$")


# def generate_positions(channels: list) -> dict:
#     info = mne.create_info(ch_names=channels, sfreq=100, ch_types="eeg")
#     montage = mne.channels.make_standard_montage("standard_1005")
#     info.set_montage(montage)

#     coords = _find_topomap_coords(info, picks=list(range(len(channels))))
#     max_abs = float(np.max(np.abs(coords)))
#     norm = coords / max_abs

#     return {
#         ch: {"x": round(float(x), 4), "y": round(float(y), 4)}
#         for ch, (x, y) in zip(channels, norm)
#     }


# def classify_hemisphere(hemi_code: str, hemi_conv: dict) -> str:
#     if "z" in hemi_code.lower():
#         key = "z"
#     elif any(c in hemi_code for c in "13579"):
#         key = "odd"
#     elif any(c in hemi_code for c in "02468"):
#         key = "even"
#     else:
#         key = "unknown"
#     return hemi_conv.get(key, key)


# def build_atlas(channels: list, brain_mapping_path: str = "brain_mapping.json") -> dict:
#     with open(brain_mapping_path, "r") as f:
#         brain_map = json.load(f)

#     regions = brain_map.get("regions", {})
#     hemi_conv = brain_map.get("hemisphere_convention", {})

#     positions = generate_positions(channels)

#     atlas = {}
#     unmapped = []
#     for ch, pos in positions.items():
#         m = CHANNEL_PATTERN.match(ch)
#         if not m:
#             unmapped.append(ch)
#             continue
#         region_code, hemi_code = m.group(1), m.group(2)
#         region_info = regions.get(region_code, {})
#         if not region_info:
#             unmapped.append(ch)

#         atlas[ch] = {
#             "x": pos["x"],
#             "y": pos["y"],
#             "region_code": region_code,
#             # Tier-1 sourced label - safe to state plainly, matches the
#             # LLM prompt's own glossary exactly (same source file).
#             "functional_region": region_info.get("functional_region", "unknown"),
#             "functional_region_basis": region_info.get(
#                 "functional_region_basis", "unmapped - review brain_mapping.json"
#             ),
#             "hemisphere": hemi_conv.get(
#                 "z" if "z" in hemi_code.lower()
#                 else "odd" if any(c in hemi_code for c in "13579")
#                 else "even" if any(c in hemi_code for c in "02468")
#                 else "unknown",
#                 "unknown",
#             ),
#         }

#     if unmapped:
#         print(f"WARNING: {len(unmapped)} channel(s) had no region match: {unmapped}")
#         print("These will render with functional_region='unknown' - check "
#               "brain_mapping.json's 'regions' keys against CHANNEL_PATTERN output.")

#     return atlas


# if __name__ == "__main__":
#     atlas = build_atlas(CHANNELS)
#     with open("channel_atlas.json", "w") as f:
#         json.dump(atlas, f, indent=2)
#     print(f"Saved {len(atlas)} channels to channel_atlas.json")

"""
build_channel_atlas.py

Generates channel_atlas.json - the single static source of truth for
"where does this channel sit on the head, and what region/hemisphere is
it in" used by BOTH the topomap and chord-diagram visualizations.

This replaces the two previously-disconnected sources:
  - brain_mapping.json's region table (used by the LLM prompt's hedged
    Tier 1/2/3 glossary)
  - the notebook's hand-typed eeg_location_data table (used only for the
    PyCirclize chord diagram, with NO hedging language)
By generating positions from MNE's own standard_1005 montage (the same
method mne.viz uses internally for its topomaps) and pairing them with
brain_mapping.json's already-hedged region labels, the diagram and the
LLM's prose can never disagree about what region a channel is in.

Run once (or whenever the channel list / brain_mapping.json regions
change):
    python build_channel_atlas.py
Produces:
    channel_atlas.json

Positions are 2D, normalised to roughly [-1, 1], generated via the exact
same projection MNE uses for `raw.plot_sensors(kind='topomap')` - so this
matches the notebook's section 5 visually, just precomputed once instead
of recomputed with mne+pyvista on every request.
"""

import json
import re

import mne
import numpy as np
from mne.channels.layout import _find_topomap_coords

# All 36 physical electrode sites in this project's montage (bare names,
# i.e. without the tmb_s1_ch/tmb_s2_ch prefix - each site carries BOTH an
# HbO (s1) and HbR (s2) signal at the same physical location).
CHANNELS = [
    "AF1", "AF2", "AF5h", "AF6h", "AF7", "AF8", "AFF3h", "AFF4h", "AFF5",
    "AFF6", "AFFz", "AFp3", "AFp4", "AFp7", "AFp8", "AFpz", "C3h", "C4h",
    "C5h", "C6h", "CCP3", "CCP4", "CPP3", "CPP4", "FCC3", "FCC4", "P3h",
    "P4h", "P5h", "P6h", "PO1", "PO2", "POOz", "PPO3", "PPO4", "PPOz",
]

# Region-code / hemisphere-suffix split - same pattern used elsewhere in
# this project (rag_core.py's pre_translate_rule) so a channel parses the
# same way everywhere.
CHANNEL_PATTERN = re.compile(r"^([a-zA-Z]+)(\d+[a-zA-Z]*|z)$")


def generate_positions(channels: list) -> dict:
    """2D positions for the flat topomap view - same projection MNE uses
    internally for raw.plot_sensors(kind='topomap')."""
    info = mne.create_info(ch_names=channels, sfreq=100, ch_types="eeg")
    montage = mne.channels.make_standard_montage("standard_1005")
    info.set_montage(montage)

    coords = _find_topomap_coords(info, picks=list(range(len(channels))))
    max_abs = float(np.max(np.abs(coords)))
    norm = coords / max_abs

    return {
        ch: {"x": round(float(x), 4), "y": round(float(y), 4)}
        for ch, (x, y) in zip(channels, norm)
    }


def generate_3d_positions(channels: list) -> dict:
    """Raw 3D head-frame coordinates (MNE 'head' coordinate system: +x =
    right, +y = anterior/front, +z = up/superior), normalised to a unit
    sphere. Used for the 3D electrode-marker rendering in React/Three.js -
    NOT the same values as the 2D topomap projection above, which uses a
    different (flattening) formula and is not just this with z dropped."""
    montage = mne.channels.make_standard_montage("standard_1005")
    pos_3d = montage.get_positions()["ch_pos"]
    coords3d = np.array([pos_3d[ch] for ch in channels])
    r = float(np.max(np.linalg.norm(coords3d, axis=1)))

    return {
        ch: {"x3d": round(float(x) / r, 4), "y3d": round(float(y) / r, 4),
             "z3d": round(float(z) / r, 4)}
        for ch, (x, y, z) in zip(channels, coords3d)
    }


def classify_hemisphere(hemi_code: str, hemi_conv: dict) -> str:
    if "z" in hemi_code.lower():
        key = "z"
    elif any(c in hemi_code for c in "13579"):
        key = "odd"
    elif any(c in hemi_code for c in "02468"):
        key = "even"
    else:
        key = "unknown"
    return hemi_conv.get(key, key)


def build_atlas(channels: list, brain_mapping_path: str = "brain_mapping.json") -> dict:
    with open(brain_mapping_path, "r") as f:
        brain_map = json.load(f)

    regions = brain_map.get("regions", {})
    hemi_conv = brain_map.get("hemisphere_convention", {})

    positions = generate_positions(channels)
    positions_3d = generate_3d_positions(channels)

    atlas = {}
    unmapped = []
    for ch, pos in positions.items():
        m = CHANNEL_PATTERN.match(ch)
        if not m:
            unmapped.append(ch)
            continue
        region_code, hemi_code = m.group(1), m.group(2)
        region_info = regions.get(region_code, {})
        if not region_info:
            unmapped.append(ch)

        atlas[ch] = {
            "x": pos["x"],
            "y": pos["y"],
            **positions_3d[ch],
            "region_code": region_code,
            # Tier-1 sourced label - safe to state plainly, matches the
            # LLM prompt's own glossary exactly (same source file).
            "functional_region": region_info.get("functional_region", "unknown"),
            "functional_region_basis": region_info.get(
                "functional_region_basis", "unmapped - review brain_mapping.json"
            ),
            "hemisphere": hemi_conv.get(
                "z" if "z" in hemi_code.lower()
                else "odd" if any(c in hemi_code for c in "13579")
                else "even" if any(c in hemi_code for c in "02468")
                else "unknown",
                "unknown",
            ),
        }

    if unmapped:
        print(f"WARNING: {len(unmapped)} channel(s) had no region match: {unmapped}")
        print("These will render with functional_region='unknown' - check "
              "brain_mapping.json's 'regions' keys against CHANNEL_PATTERN output.")

    return atlas


def build_mesh_edges(atlas: dict, k: int = 3) -> list:
    """K-nearest-neighbour graph over 2D positions - gives the montage
    diagram its 'grid' look (like a standard 10-20 layout illustration)
    without needing a hand-maintained adjacency list. k=3 keeps it
    readable; raise it for a denser mesh."""
    channels = list(atlas.keys())
    coords = np.array([[atlas[c]["x"], atlas[c]["y"]] for c in channels])

    edges = set()
    for i, ch in enumerate(channels):
        dists = np.linalg.norm(coords - coords[i], axis=1)
        nearest = np.argsort(dists)[1:k + 1]  # skip self at index 0
        for j in nearest:
            edges.add(tuple(sorted([ch, channels[int(j)]])))

    return [{"a": a, "b": b} for a, b in sorted(edges)]


if __name__ == "__main__":
    atlas = build_atlas(CHANNELS)
    mesh_edges = build_mesh_edges(atlas, k=3)
    output = {"channels": atlas, "mesh_edges": mesh_edges}
    with open("channel_atlas.json", "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved {len(atlas)} channels and {len(mesh_edges)} mesh edges "
          f"(each channel now includes 2D x/y AND 3D x3d/y3d/z3d) to channel_atlas.json")