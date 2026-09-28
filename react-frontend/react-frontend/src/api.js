/**
 * api.js
 *
 * Thin fetch wrapper for backend.py's FastAPI service — the React
 * equivalent of api_client.py. Same endpoints, same response shapes.
 * Nothing here talks to rag_core.py or viz_tools.py directly; this file
 * is the ONLY thing the rest of the app talks to, mirroring the
 * frontend/backend split api_client.py's docstring describes.
 */

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "http://127.0.0.1:8000";

export class BackendError extends Error {}

async function get(path) {
  let resp;
  try {
    resp = await fetch(`${BACKEND_URL}${path}`);
  } catch (e) {
    throw new BackendError(`Could not reach backend at ${BACKEND_URL}${path}: ${e.message}`);
  }
  if (!resp.ok) {
    throw new BackendError(`Backend returned ${resp.status} for ${path}`);
  }
  return resp.json();
}

async function post(path, body) {
  let resp;
  try {
    resp = await fetch(`${BACKEND_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (e) {
    throw new BackendError(`Could not reach backend at ${BACKEND_URL}${path}: ${e.message}`);
  }
  if (!resp.ok) {
    throw new BackendError(`Backend returned ${resp.status} for ${path}`);
  }
  return resp.json();
}

export const health = () => get("/health");

/** Returns {target_name, n_rules, rules, cv_performance_estimate, n_seeds}. */
export const getRules = () => get("/rules");

/** Returns the static channel atlas used as the topomap background. */
export const getAtlas = () => get("/atlas");

/**
 * Returns {answer, retrieved, in_scope, used_fallback, latency_ms,
 * visualization}. `visualization` is null or
 * {type, topomap: [...] | null, chord: {...} | null}.
 */
export const ask = (question) => post("/ask", { question });

/** Topomap points across EVERY rule in the model (sidebar overview). */
export const getOverviewTopomap = () => get("/visualize/topomap");

/** Chord {nodes, edges} across EVERY rule in the model (sidebar overview). */
export const getOverviewChord = () => get("/visualize/chord");
