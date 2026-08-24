"""
viz_components.py

Turns the JSON payloads returned by backend.py's /ask and /visualize/*
endpoints into things Streamlit can actually render.

Two different rendering strategies on purpose (see the planning
discussion this came out of):
  - Topomap -> Vega-Lite spec. Pure declarative JSON, rendered natively
    by st.vega_lite_chart with zero custom JS. The exact same spec dict
    renders in a React app later via react-vega/vega-embed - no rewrite
    needed, just a different host component.
  - Chord diagram -> hand-written D3, embedded via st.components.v1.html
    (an iframe running real JS). Vega-Lite has no native chord/arc-ribbon
    mark, and this is the one piece of genuinely custom visualization
    logic in the project. The SAME script, unmodified, becomes a React
    component later (useEffect + a ref instead of a Streamlit iframe) -
    it's isolated here specifically so that migration is a host-swap, not
    a rewrite.
"""

import json

REGION_COLORS = {
    "frontal": "#4C78A8",
    "motor": "#F58518",
    "parietal": "#54A24B",
    "occipital": "#E45756",
    "AMBIGUOUS - PPOz clusters near occipital POz; PPO3/PPO4 cluster near parietal P3/P4. Do not assert a single region for 'PPO' codes without the exact suffix.": "#B279A2",
    "unknown": "#999999",
}

LEVEL_COLORS = {"Low": "#4C78A8", "Medium": "#B0B0B0", "High": "#E45756"}


def _region_color(functional_region: str) -> str:
    return REGION_COLORS.get(functional_region, REGION_COLORS["unknown"])


# ---------------------------------------------------------
# Topomap - Vega-Lite spec
# ---------------------------------------------------------
def build_topomap_spec(points: list, title: str = "", atlas: dict = None) -> dict:
    """points: the `topomap` list from AskResponse.visualization or
    /visualize/topomap - list of dicts with feature/bare_channel/signal/
    level/x/y/functional_region/hemisphere/rule_id/consequent/accuracy.

    atlas: optional full-montage background from GET /atlas
    ({"channels": {...}, "mesh_edges": [...]}). When provided, draws
    every one of the 36 electrodes faded gray with the neighbour mesh
    grid behind them - like a standard electrode-layout diagram - with
    `points` highlighted in color on top. Without it, only the
    highlighted points render (the original, simpler look)."""

    # Simple head-outline guide so bare scattered dots have spatial
    # context - a circle plus a small nose marker, not an anatomical
    # claim, purely an orientation aid.
    import math
    circle_pts = [
        {"cx": round(math.cos(t), 4), "cy": round(math.sin(t), 4)}
        for t in [i * (2 * math.pi / 60) for i in range(61)]
    ]

    layers = [
        {
            "data": {"values": circle_pts},
            "mark": {"type": "line", "interpolate": "linear-closed", "color": "#ccc", "strokeWidth": 1},
            "encoding": {
                "x": {"field": "cx", "type": "quantitative", "scale": {"domain": [-1.3, 1.3]}, "axis": None},
                "y": {"field": "cy", "type": "quantitative", "scale": {"domain": [-1.3, 1.3]}, "axis": None},
            },
        },
    ]

    if atlas:
        chans = atlas["channels"]
        # Mesh grid lines - each edge as its own two-point line segment,
        # grouped by an edge id so Vega-Lite doesn't connect unrelated
        # segments together.
        mesh_rows = []
        for i, e in enumerate(atlas.get("mesh_edges", [])):
            a, b = chans.get(e["a"]), chans.get(e["b"])
            if not a or not b:
                continue
            mesh_rows.append({"edge_id": i, "x": a["x"], "y": a["y"]})
            mesh_rows.append({"edge_id": i, "x": b["x"], "y": b["y"]})

        if mesh_rows:
            layers.append({
                "data": {"values": mesh_rows},
                "mark": {"type": "line", "color": "#e0e0e0", "strokeWidth": 1},
                "encoding": {
                    "x": {"field": "x", "type": "quantitative", "scale": {"domain": [-1.3, 1.3]}, "axis": None},
                    "y": {"field": "y", "type": "quantitative", "scale": {"domain": [-1.3, 1.3]}, "axis": None},
                    "detail": {"field": "edge_id", "type": "nominal"},
                },
            })

        # All 36 electrodes, faded, as spatial context. Skips channels
        # already in `points` so a highlighted electrode doesn't get a
        # dim duplicate sitting under it.
        highlighted = {p["bare_channel"] for p in points}
        background_pts = [
            {"bare_channel": ch, "x": v["x"], "y": v["y"]}
            for ch, v in chans.items() if ch not in highlighted
        ]
        if background_pts:
            layers.append({
                "data": {"values": background_pts},
                "mark": {"type": "circle", "size": 90, "color": "#d8d8d8", "stroke": "#bbb", "strokeWidth": 0.5},
                "encoding": {
                    "x": {"field": "x", "type": "quantitative", "scale": {"domain": [-1.3, 1.3]}, "axis": None},
                    "y": {"field": "y", "type": "quantitative", "scale": {"domain": [-1.3, 1.3]}, "axis": None},
                    "tooltip": [{"field": "bare_channel", "title": "Channel"}],
                },
            })
            layers.append({
                "data": {"values": background_pts},
                "mark": {"type": "text", "dy": -12, "fontSize": 7, "color": "#aaa"},
                "encoding": {
                    "x": {"field": "x", "type": "quantitative", "scale": {"domain": [-1.3, 1.3]}, "axis": None},
                    "y": {"field": "y", "type": "quantitative", "scale": {"domain": [-1.3, 1.3]}, "axis": None},
                    "text": {"field": "bare_channel", "type": "nominal"},
                },
            })

    layers += [
            {
                "data": {"values": points},
                "mark": {"type": "circle", "size": 220, "stroke": "#333", "strokeWidth": 0.7},
                "encoding": {
                    "x": {"field": "x", "type": "quantitative", "scale": {"domain": [-1.3, 1.3]}, "axis": None},
                    "y": {"field": "y", "type": "quantitative", "scale": {"domain": [-1.3, 1.3]}, "axis": None},
                    "color": {
                        "field": "level", "type": "nominal",
                        "scale": {"domain": ["Low", "Medium", "High"],
                                  "range": [LEVEL_COLORS["Low"], LEVEL_COLORS["Medium"], LEVEL_COLORS["High"]]},
                        "legend": {"title": "Activation level"},
                    },
                    "shape": {
                        "field": "signal", "type": "nominal",
                        "scale": {"domain": ["HbO", "HbR"], "range": ["circle", "diamond"]},
                        "legend": {"title": "Signal"},
                    },
                    "tooltip": [
                        {"field": "bare_channel", "title": "Channel"},
                        {"field": "signal", "title": "Signal"},
                        {"field": "level", "title": "Level"},
                        {"field": "functional_region", "title": "Region"},
                        {"field": "rule_id", "title": "Rule"},
                        {"field": "consequent", "title": "Predicts"},
                        {"field": "accuracy", "title": "Rule accuracy", "format": ".2f"},
                    ],
                },
            },
            {
                "data": {"values": points},
                "mark": {"type": "text", "dy": -14, "fontSize": 9, "color": "#333"},
                "encoding": {
                    "x": {"field": "x", "type": "quantitative", "scale": {"domain": [-1.3, 1.3]}, "axis": None},
                    "y": {"field": "y", "type": "quantitative", "scale": {"domain": [-1.3, 1.3]}, "axis": None},
                    "text": {"field": "bare_channel", "type": "nominal"},
                },
            },
    ]

    return {
        "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
        "title": title,
        "width": 380,
        "height": 380,
        "layer": layers,
        "config": {"view": {"stroke": None}},
    }


# ---------------------------------------------------------
# Chord diagram - hand-written D3, isolated on purpose (see module docstring)
# ---------------------------------------------------------
_CHORD_TEMPLATE = """
<div id="chord-root" style="position:relative; font-family: sans-serif;">
  <div id="chord-tooltip" style="position:absolute; display:none; background:white;
       border:1px solid #ccc; border-radius:4px; padding:6px 10px; font-size:12px;
       box-shadow:0 2px 6px rgba(0,0,0,0.15); pointer-events:none; z-index:10;"></div>
  <svg id="chord-svg"></svg>
</div>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script>
(function() {
  const DATA = __DATA_JSON__;
  const regionColor = {
    "frontal": "#4C78A8", "motor": "#F58518", "parietal": "#54A24B",
    "occipital": "#E45756", "unknown": "#999999"
  };

  const nodes = DATA.nodes.slice();
  const regionOrder = ["frontal", "motor", "parietal", "occipital"];
  nodes.sort((a, b) => {
    const ra = regionOrder.indexOf(a.functional_region);
    const rb = regionOrder.indexOf(b.functional_region);
    return (ra === -1 ? 99 : ra) - (rb === -1 ? 99 : rb) || a.channel.localeCompare(b.channel);
  });

  const idx = new Map(nodes.map((n, i) => [n.channel, i]));
  const n = nodes.length;
  const matrix = Array.from({ length: n }, () => new Array(n).fill(0));
  const edgeLookup = new Map();

  DATA.edges.forEach(e => {
    const i = idx.get(e.source), j = idx.get(e.target);
    if (i === undefined || j === undefined) return;
    const w = (e.dominance_score != null ? e.dominance_score : 0.05) * 150;
    matrix[i][j] += w;
    matrix[j][i] += w;
    const key = i < j ? i + "-" + j : j + "-" + i;
    if (!edgeLookup.has(key)) edgeLookup.set(key, []);
    edgeLookup.get(key).push(e);
  });

  const width = 460, height = 460;
  const outerRadius = Math.min(width, height) * 0.5 - 60;
  const innerRadius = outerRadius - 12;

  const svg = d3.select("#chord-svg")
    .attr("width", width).attr("height", height)
    .append("g")
    .attr("transform", "translate(" + width / 2 + "," + height / 2 + ")");

  const tooltip = d3.select("#chord-tooltip");

  if (n === 0) {
    d3.select("#chord-root").append("p").text("No channels to display.");
    return;
  }

  const chordLayout = d3.chord().padAngle(0.04).sortSubgroups(d3.descending);
  const chords = chordLayout(matrix);
  const arcGen = d3.arc().innerRadius(innerRadius).outerRadius(outerRadius);
  const ribbonGen = d3.ribbon().radius(innerRadius);

  function showTooltip(event, html) {
    tooltip.style("display", "block").html(html)
      .style("left", (event.offsetX + 12) + "px")
      .style("top", (event.offsetY + 12) + "px");
  }
  function hideTooltip() { tooltip.style("display", "none"); }

  const group = svg.append("g").selectAll("g").data(chords.groups).join("g");

  group.append("path")
    .attr("fill", d => regionColor[nodes[d.index].functional_region] || regionColor.unknown)
    .attr("stroke", "#222")
    .attr("stroke-width", 0.6)
    .attr("d", arcGen)
    .on("mouseover", (event, d) => {
      const node = nodes[d.index];
      showTooltip(event, "<b>" + node.channel + "</b><br/>Region: " + node.functional_region +
        "<br/>" + node.hemisphere);
    })
    .on("mouseout", hideTooltip);

  group.append("text")
    .each(d => { d.angle = (d.startAngle + d.endAngle) / 2; })
    .attr("dy", "0.35em")
    .attr("transform", d => {
      const rot = (d.angle * 180 / Math.PI - 90);
      const flip = d.angle > Math.PI ? "rotate(180)" : "";
      return "rotate(" + rot + ") translate(" + (outerRadius + 8) + ") " + flip;
    })
    .attr("text-anchor", d => d.angle > Math.PI ? "end" : null)
    .style("font-size", "9px")
    .text(d => nodes[d.index].channel);

  svg.append("g")
    .attr("fill-opacity", 0.75)
    .selectAll("path")
    .data(chords)
    .join("path")
    .attr("d", ribbonGen)
    .attr("fill", d => regionColor[nodes[d.source.index].functional_region] || regionColor.unknown)
    .attr("stroke", "#000")
    .attr("stroke-width", 0.25)
    .on("mouseover", (event, d) => {
      const key = d.source.index < d.target.index
        ? d.source.index + "-" + d.target.index
        : d.target.index + "-" + d.source.index;
      const edges = edgeLookup.get(key) || [];
      const lines = edges.map(e =>
        "Rule " + e.rule_id + " &rarr; " + e.consequent +
        " (acc=" + (e.accuracy != null ? e.accuracy.toFixed(2) : "n/a") + ")"
      ).join("<br/>");
      showTooltip(event, "<b>" + nodes[d.source.index].channel + " &harr; " +
        nodes[d.target.index].channel + "</b><br/>" + lines);
    })
    .on("mouseout", hideTooltip);
})();
</script>
"""


def build_chord_html(chord: dict) -> str:
    """chord: {"nodes": [...], "edges": [...]} from AskResponse.visualization
    or /visualize/chord. Returns a self-contained HTML fragment for
    st.components.v1.html()."""
    return _CHORD_TEMPLATE.replace("__DATA_JSON__", json.dumps(chord))


# ---------------------------------------------------------
# 3D brain view - vanilla Three.js, same embedding pattern as the chord
# diagram (st.components.v1.html running a real <script>, not a Streamlit-
# specific mechanism). This is a RENDERING CHOICE over the same topomap
# payload the 2D Vega-Lite view uses - x3d/y3d/z3d are already present on
# every point backend.py returns, so no backend change is needed.
#
# The surface is the REAL fsaverage pial (cortical) surface, exported once
# by export_brain_mesh.py into the EXACT normalized coordinate frame that
# channel_atlas.json's x3d/y3d/z3d live in (pial_mm / 1000 / r, with the
# head->gltf axis swap already baked into the vertices). That is what
# replaces the old hand-built "balloon head": that mesh put its face and
# nose at height 0 - the same height as the ear-level coordinate origin -
# so every scalp electrode rendered at eye level, and its 1.28x horizontal
# stretch sank markers inside the surface. The coordinates were never
# wrong; the stylized mesh was simply never aligned to them.
#
# Electrodes sit on the SCALP (radius 0.73-1.0 in these units) and the
# mesh is the CORTEX (max radius ~0.6), so the markers already float
# above the surface at their true coordinates - MARKER_INFLATE stays at
# 1.0 and exists only as an escape hatch. A faint translucent scalp shell
# is drawn at radius 1.0 so the sensors visibly read as cap positions
# over the cortex, the same look as the notebook's section 6 figure.
# ---------------------------------------------------------

# Where the iframe fetches the GLB from. Streamlit serves ./static/* at
# /app/static/* once `enableStaticServing = true` is set in
# .streamlit/config.toml. Several candidates are tried in order so the
# same file also works when the app is mounted under a path prefix or
# opened as a plain HTML file during development.
BRAIN_MESH_URL = "/app/static/fsaverage_brain.glb"
BRAIN_MESH_FALLBACK_URLS = [
    "./static/fsaverage_brain.glb",
    "static/fsaverage_brain.glb",
]

# How far outward (x radius) the markers are pushed from the origin so
# scalp sensors clear the cortical surface instead of intersecting it.
MARKER_INFLATE = 1.0

_BRAIN3D_TEMPLATE = """
<div id="brain3d-root" style="position:relative; font-family: sans-serif;">
  <div id="brain3d-tooltip" style="position:absolute; display:none; background:white;
       border:1px solid #ccc; border-radius:4px; padding:6px 10px; font-size:12px;
       box-shadow:0 2px 6px rgba(0,0,0,0.15); pointer-events:none; z-index:10;"></div>
  <div id="brain3d-canvas" style="width:460px; height:460px;"></div>
  <div id="brain3d-status" style="color:#a33; font-size:11px; margin-top:4px; display:none;"></div>
  <p style="color:#888; font-size:11px; margin-top:4px;">Drag to rotate, scroll to zoom.</p>
</div>
<script type="importmap">
{
  "imports": {
    "three": "https://unpkg.com/three@0.160.0/build/three.module.js",
    "three/addons/": "https://unpkg.com/three@0.160.0/examples/jsm/"
  }
}
</script>
<script type="module">
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const POINTS = __DATA_JSON__;
const MESH_URLS = __MESH_URLS_JSON__;
const INFLATE = __MARKER_INFLATE__;
const LEVEL_COLOR = { Low: 0x4C78A8, Medium: 0xB0B0B0, High: 0xE45756 };

const container = document.getElementById("brain3d-canvas");
const status = document.getElementById("brain3d-status");
const width = 460, height = 460;

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xffffff);

const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
// Marker mapping below sends anterior (front, +y3d) to NEGATIVE three.z,
// so a camera at negative z starts looking at the FRONT of the head -
// this dataset is frontal-heavy (16/36 channels), so that's the more
// useful default view than starting from the back.
camera.position.set(0, 0.55, -2.4);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(width, height);
container.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 0.15, 0);

scene.add(new THREE.AmbientLight(0xffffff, 0.75));
const dirLight = new THREE.DirectionalLight(0xffffff, 0.55);
dirLight.position.set(2, 3, -3);
scene.add(dirLight);
const backLight = new THREE.DirectionalLight(0xffffff, 0.3);
backLight.position.set(-2, -1, 3);
scene.add(backLight);

// ---- real cortical surface -------------------------------------------
// The GLB's vertices are already in the markers' coordinate frame, so it
// is added with no scaling, rotation or offset whatsoever.
const brainMat = new THREE.MeshPhongMaterial({
  color: 0xd9c3bd, specular: 0x222222, shininess: 12,
  side: THREE.DoubleSide, flatShading: false,
});

const loader = new GLTFLoader();

function loadMesh(index) {
  if (index >= MESH_URLS.length) {
    status.style.display = "block";
    status.textContent =
      "Brain surface (fsaverage_brain.glb) not found - showing sensor " +
      "positions only. Put the file in static/ next to streamlit_app.py " +
      "and set enableStaticServing = true in .streamlit/config.toml.";
    return;
  }
  loader.load(
    MESH_URLS[index],
    (gltf) => {
      gltf.scene.traverse((child) => {
        if (child.isMesh) {
          child.material = brainMat;
          child.geometry.computeVertexNormals();
        }
      });
      scene.add(gltf.scene);
    },
    undefined,
    () => loadMesh(index + 1),
  );
}
loadMesh(0);

// ---- faint scalp shell ----------------------------------------------
// Orientation aid only, NOT an anatomical claim: a translucent sphere at
// the electrode normalization radius (1.0), so markers read as sitting on
// a cap above the cortex rather than floating in empty space.
const scalp = new THREE.Mesh(
  new THREE.SphereGeometry(1.0, 48, 32),
  new THREE.MeshPhongMaterial({
    color: 0xe7bda7, transparent: true, opacity: 0.10,
    side: THREE.DoubleSide, depthWrite: false,
  }),
);
scene.add(scalp);

// ---- sensor markers ---------------------------------------------------
const raycaster = new THREE.Raycaster();
const mouse = new THREE.Vector2();
const markers = [];

POINTS.forEach(p => {
  if (p.x3d == null || p.y3d == null || p.z3d == null) return;
  const isHbO = p.signal === "HbO";
  const geo = isHbO
    ? new THREE.SphereGeometry(0.038, 14, 14)
    : new THREE.BoxGeometry(0.06, 0.06, 0.06);
  const mat = new THREE.MeshPhongMaterial({
    color: LEVEL_COLOR[p.level] || 0x999999,
  });
  const marker = new THREE.Mesh(geo, mat);
  // Head coords: +x=right, +y=anterior/front, +z=superior/up.
  // Three.js coords: +x=right, +y=up, +z=toward viewer.
  // So: head x -> three x, head z(up) -> three y(up), head y(front) -> three -z(front).
  // Electrodes are scalp positions on a unit sphere; the mesh is cortex,
  // so push each marker outward along its own radius to sit just above it.
  // HbO and HbR share one physical electrode, so nudge them slightly in
  // and out along the radius - otherwise one marker hides inside the
  // other (same idea as the 2D view's SIGNAL_OFFSET).
  const s = INFLATE * (isHbO ? 0.975 : 1.025);
  marker.position.set(p.x3d * s, p.z3d * s, -p.y3d * s);
  marker.userData = p;
  scene.add(marker);
  markers.push(marker);
});

const tooltip = document.getElementById("brain3d-tooltip");

function onPointerMove(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

  raycaster.setFromCamera(mouse, camera);
  const hits = raycaster.intersectObjects(markers);

  if (hits.length > 0) {
    const p = hits[0].object.userData;
    tooltip.style.display = "block";
    tooltip.style.left = (event.clientX - rect.left + 12) + "px";
    tooltip.style.top = (event.clientY - rect.top + 12) + "px";
    tooltip.innerHTML = "<b>" + p.bare_channel + "</b> (" + p.signal + ")<br/>" +
      "Level: " + p.level + "<br/>Region: " + p.functional_region + "<br/>" +
      "Rule " + p.rule_id + " &rarr; " + p.consequent +
      " (acc=" + (p.accuracy != null ? p.accuracy.toFixed(2) : "n/a") + ")";
  } else {
    tooltip.style.display = "none";
  }
}
renderer.domElement.addEventListener("pointermove", onPointerMove);

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();
</script>
"""


def build_brain3d_html(points: list, mesh_urls: list = None) -> str:
    """points: the `topomap` list from AskResponse.visualization or
    /visualize/topomap - SAME data as build_topomap_spec(), just a
    different renderer. Points missing x3d/y3d/z3d are silently skipped
    (older cached responses before the 3D fields existed, defensive
    only - a freshly-generated channel_atlas.json always has them).

    mesh_urls: optional override for where the iframe fetches
    fsaverage_brain.glb from - each URL is tried in order and the first
    that loads wins (a base64 `data:model/gltf-binary;base64,...` URL
    works here too, if static file serving isn't available). Defaults to
    Streamlit's static folder plus relative fallbacks."""
    urls = mesh_urls or ([BRAIN_MESH_URL] + BRAIN_MESH_FALLBACK_URLS)
    return (
        _BRAIN3D_TEMPLATE
        .replace("__DATA_JSON__", json.dumps(points))
        .replace("__MESH_URLS_JSON__", json.dumps(urls))
        .replace("__MARKER_INFLATE__", repr(MARKER_INFLATE))
    )
