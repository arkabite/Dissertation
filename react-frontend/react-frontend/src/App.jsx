import { useEffect, useState } from "react";
import Sidebar from "./components/Sidebar.jsx";
import ChatPanel from "./components/ChatPanel.jsx";
import { health, getRules, getAtlas, getOverviewTopomap, getOverviewChord, BackendError } from "./api.js";

export default function App() {
  const [state, setState] = useState({ status: "loading" });

  useEffect(() => {
    let cancelled = false;

    async function load() {
      try {
        const [h, data, atlas, overviewTopomap, overviewChord] = await Promise.all([
          health(),
          getRules(),
          getAtlas(),
          getOverviewTopomap(),
          getOverviewChord(),
        ]);
        if (cancelled) return;
        setState({ status: "ready", health: h, data, atlas, overviewTopomap, overviewChord });
      } catch (err) {
        if (cancelled) return;
        const message =
          err instanceof BackendError
            ? err.message
            : `Unexpected error loading the model: ${err.message}`;
        setState({ status: "error", message });
      }
    }

    load();
    return () => {
      cancelled = true;
    };
  }, []);

  if (state.status === "loading") {
    return (
      <div className="main-panel">
        <h1>fNIRS Fuzzy Rule Explainer</h1>
        <p className="caption">Connecting to the backend...</p>
      </div>
    );
  }

  if (state.status === "error") {
    return (
      <div className="main-panel">
        <h1>fNIRS Fuzzy Rule Explainer</h1>
        <div className="error-banner">
          Can't reach the backend API.
          <br />
          {state.message}
          <br />
          Make sure it's running: <code>uvicorn backend:app --reload</code> (default
          expected at http://127.0.0.1:8000, or set VITE_BACKEND_URL).
        </div>
      </div>
    );
  }

  const degraded = state.health.status !== "ok";

  return (
    <div className="app-layout">
      <Sidebar
        data={state.data}
        atlas={state.atlas}
        overviewTopomap={state.overviewTopomap}
        overviewChord={state.overviewChord}
      />
      <div>
        {degraded && (
          <div className="warning-banner warning-banner-spaced">
            Backend is reachable but degraded: {state.health.llm_client_error}. Questions
            will still work using rule-matching only, without natural-language
            explanations or visualizations.
          </div>
        )}
        <ChatPanel atlas={state.atlas} />
      </div>
    </div>
  );
}
