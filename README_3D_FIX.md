# 3D brain view fix - where to put these files

Your coordinates were never wrong. `channel_atlas.json`'s `x3d/y3d/z3d` are
standard 10-5 head coordinates normalized by the max electrode radius
(r = 0.11463 m), origin at ear level. The old `build_brain3d_html` drew a
hand-built "balloon head" whose face/nose sat at height 0 - the same height as
that origin - so scalp electrodes rendered at eye level, and its 1.28x
horizontal stretch buried markers inside the surface.

## Files

| File | Where it goes |
|---|---|
| `fsaverage_brain.glb` | `static/fsaverage_brain.glb` (folder next to `streamlit_app.py`) |
| `viz_components.py` | replaces your current `viz_components.py` |
| `export_brain_mesh.py` | project root; only needed to regenerate the GLB |

## One config line

`.streamlit/config.toml`:

```toml
[server]
enableStaticServing = true
```

Streamlit then serves `static/fsaverage_brain.glb` at `/app/static/fsaverage_brain.glb`,
which is the first URL the component tries. It also falls back to
`./static/...` and `static/...`, and if none load it shows an inline message and
still renders the sensors. If static serving is unavailable you can pass a
base64 URL instead:

```python
import base64, pathlib
b64 = base64.b64encode(pathlib.Path("static/fsaverage_brain.glb").read_bytes()).decode()
html = build_brain3d_html(points, mesh_urls=[f"data:model/gltf-binary;base64,{b64}"])
```

No changes are needed in `streamlit_app.py`, `backend.py`, `rag_core.py` or
`viz_tools.py` - `build_brain3d_html(points)` keeps its signature.

## What changed in viz_components.py

- Real fsaverage pial surface loaded via three.js `GLTFLoader` (same pinned
  0.160.0 release, plus the loader addon). Vertices are pre-baked into the
  markers' frame (`pial_mm / 1000 / r`, head->gltf axis swap applied), so the
  mesh is added with no scale, rotation or offset.
- Balloon head, ears, neck and nose cone removed. A faint translucent shell at
  radius 1.0 (the electrode normalization radius) remains purely as an
  orientation aid.
- Markers stay at their true coordinates `(x3d, z3d, -y3d)`; electrodes are at
  radius 0.73-1.0 and the cortex maxes out at ~0.6, so they naturally float
  above the surface. HbO/HbR at the same site are nudged +/-2.5% radially so
  neither hides inside the other.
- OrbitControls, hover tooltips, Low/Medium/High colours and HbO spheres /
  HbR boxes are unchanged.

## Notes on the notebook

Section 6's `*750, -28, -28` hand-tuning was compensating for the wrong
transform. The `standard_1005` montage's nasion/LPA/RPA are numerically
identical to fsaverage's MRI-frame fiducials, so no fudge factors are needed -
a plain unit conversion aligns cortex and electrodes. MNE's shipped
`fsaverage-trans.fif` is a different head-frame convention and is deliberately
not applied (using it is what shifts the brain 3-4 cm off the electrodes).

## Verified

Rendered headlessly with all 36 channels x HbO/HbR (72 markers) from front,
side, top and back: mesh loads (no fallback message), every marker sits on the
scalp above the cortex, frontal AF channels over the frontal poles, C3h/C4h at
the crown. Screenshots included: `v_front.png`, `v_side.png`, `v_top.png`.

Mesh stats: lh+rh pial merged 655,360 faces -> decimated to 60,000; 1.1 MB GLB;
bounds x -0.600..0.612, y -0.915..0.604, z -0.417..0.689.
