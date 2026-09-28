# fNIRS Fuzzy Rule Explainer — React frontend

Replaces `Streamlit_app.py`. Talks ONLY to `backend.py`'s FastAPI service
via `src/api.js`, which mirrors `api_client.py`'s contract exactly
(`/health`, `/rules`, `/atlas`, `POST /ask`, `/visualize/topomap`,
`/visualize/chord`). It does not import `rag_core.py` or `viz_tools.py` —
all retrieval/LLM/visualization logic stays server-side.

## Prerequisites

- Node.js 18+ and npm
- `backend.py` running (see the main project's setup — `uvicorn backend:app --reload`)
- `public/fsaverage_brain.glb` present for the React 3D brain view

## Setup

```bash
cd react-frontend
npm install
cp .env.example .env   # edit VITE_BACKEND_URL if backend isn't on localhost:8000
npm run dev
```

Opens at `http://localhost:5173` by default. `backend.py`'s
`CORSMiddleware` already allows this origin.

## Building for deployment

```bash
npm run build
```

Outputs a static `dist/` folder — this is what you'd deploy to Azure
Static Web Apps (or any static host). Remember to set `VITE_BACKEND_URL`
to point at the deployed backend before building, since Vite bakes
`import.meta.env` values in at build time, not runtime.

## What's here vs. not yet ported

- **Ported**: model overview, CV performance metrics, full rule list,
  2D/3D channel maps (all-rules overview + per-answer), connectivity
  chord diagram (all-rules overview + per-answer), chat with fallback
  and degraded-backend handling, out-of-scope guard behaviour (handled
  server-side by `backend.py`, this just renders whatever comes back).
- **3D view**: uses Three.js, OrbitControls, and the same `x3d`/`y3d`/`z3d`
  coordinate mapping as `viz_components.py`'s `build_brain3d_html`.
  Drag to rotate, scroll to zoom, and hover markers for channel/rule details.

## File map

```
src/
  api.js                    – backend HTTP client (mirrors api_client.py)
  App.jsx                   – loads backend state on mount, top-level layout
  index.css                 – "Amber Desk" theme, ported from Streamlit_app.py's CSS
  components/
    Sidebar.jsx              – model overview, performance, rule list, overview visualizations
    ChatPanel.jsx            – chat history + input, calls POST /ask
    Message.jsx              – single chat turn (answer, retrieved rules, visualization)
    VisualizationPanel.jsx   – switches between Topomap/ChordDiagram by visualization.type
    Topomap.jsx              – D3 2D scalp map (Low/Medium/High colors, HbO circle / HbR diamond)
    Brain3D.jsx              – responsive Three.js cortical surface and sensor markers
    ChordDiagram.jsx         – D3 connectivity diagram
```
