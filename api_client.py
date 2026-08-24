"""
api_client.py

Thin HTTP client for backend.py's FastAPI service. This is the ONLY thing
Streamlit_app.py talks to now - it no longer imports rag_core directly.
That's a deliberate architecture decision, not just a style choice: it
means backend.py is the single source of truth for retrieval/prompting/
visualization logic, and swapping Streamlit for a React frontend later
means writing a new client against these same endpoints, not touching
the backend at all.

Set BACKEND_URL as an environment variable if the API isn't running on
the default localhost:8000 (e.g. a deployed backend).
"""

import os
from typing import Optional

import requests

BACKEND_URL = os.environ.get("BACKEND_URL", "http://127.0.0.1:8000")
TIMEOUT_S = 30  # generous - cloud LLM calls (occasionally two, for
                 # visualization turns) can take a while; see rag_core.py's
                 # ask_with_visualization for why latency isn't tightly
                 # budgeted here.


class BackendError(Exception):
    """Raised when the backend is unreachable or returns an error. Kept
    as one exception type so Streamlit_app.py has a single except clause
    to handle, rather than needing to know about requests' internals."""


def _get(path: str, params: Optional[dict] = None) -> dict:
    try:
        resp = requests.get(f"{BACKEND_URL}{path}", params=params, timeout=TIMEOUT_S)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        raise BackendError(f"Could not reach backend at {BACKEND_URL}{path}: {e}") from e


def _post(path: str, json_body: dict) -> dict:
    try:
        resp = requests.post(f"{BACKEND_URL}{path}", json=json_body, timeout=TIMEOUT_S)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.RequestException as e:
        raise BackendError(f"Could not reach backend at {BACKEND_URL}{path}: {e}") from e


def health() -> dict:
    return _get("/health")


def get_rules() -> dict:
    """Returns {target_name, n_rules, rules, cv_performance_estimate, n_seeds}."""
    return _get("/rules")


def get_atlas() -> dict:
    """Returns the static channel atlas used as the topomap background."""
    return _get("/atlas")


def ask(question: str) -> dict:
    """Returns {answer, retrieved, in_scope, used_fallback, latency_ms,
    visualization}. `visualization` is None or
    {type, topomap: [...] | None, chord: {...} | None}."""
    return _post("/ask", {"question": question})


def get_overview_topomap() -> list:
    """Topomap points across EVERY rule in the model - for the sidebar
    overview, independent of any chat question."""
    return _get("/visualize/topomap")


def get_overview_chord() -> dict:
    """Chord {nodes, edges} across EVERY rule in the model - for the
    sidebar overview, independent of any chat question."""
    return _get("/visualize/chord")
