# """
# backend.py

# FastAPI layer over rag_core.py's retrieval + prompt + LLM + visualization
# logic. This decouples the "explain a fuzzy rule" service from any
# particular frontend (Streamlit today, potentially React later) and gives
# the system an actual measurable API - relevant for the proposal's own
# "Backend Latency < 500ms" and "Response time < 3s for 95% of queries"
# success criteria, neither of which can be measured against a bare
# Streamlit script.

# What this adds on top of what rag_core.py already does:
#   - /ask, /rules, /health as real, documented endpoints (see /docs)
#   - scope-guarding: off-domain questions get a canned response instead
#     of spending an LLM call and risking an improvised answer
#   - no-LLM fallback: if the Ollama call errors/times out, the API still
#     returns something useful (the matching rules verbatim) instead of
#     failing outright
#   - visualization tool-calling: /ask now uses ask_with_visualization(),
#     which may run a second LLM call when the model requests a topomap or
#     chord diagram (see rag_core.py's module docstring). The response
#     schema's `visualization` field carries structured JSON (channel
#     points / chord nodes+edges), NOT a rendered image - deliberately, so
#     a Streamlit frontend and a future React+D3 frontend can both consume
#     the same endpoint without the backend caring which one asked.
#   - latency timing on every request, in the response body
#   - every interaction logged to SQLite (interactions.db) - question,
#     whether it was in scope, whether the fallback fired, which rule IDs
#     were retrieved, whether/which visualization was shown, the answer,
#     and latency. This is the data you'll need later for task-completion /
#     response-accuracy / latency reporting, so it's captured from day one
#     rather than reconstructed after the fact.

# Run with:
#     uvicorn backend:app --reload
# Then see interactive docs at:
#     http://127.0.0.1:8000/docs
# """

# import os
# import sqlite3
# import time
# from contextlib import asynccontextmanager
# from datetime import datetime, timezone
# from typing import Optional

# from fastapi import FastAPI, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# from rag_core import (
#     RULES_PATH, CLOUD_MODEL,
#     load_rules, load_domain_background, load_brain_mapping, get_client,
#     retrieve_rules, build_prompt, build_stability_lookup,
#     is_in_scope, classify_scope, build_fallback_answer,
#     ask_with_visualization, ask_general_neuro,
# )
# from viz_tools import (
#     load_channel_atlas, resolve_requested_channels,
#     build_topomap_payload, build_chord_payload,
# )

# DB_PATH = os.environ.get("DB_PATH", "interactions.db")

# # Loaded once at process startup - the FastAPI equivalent of
# # streamlit_app.py's @st.cache_resource init(). A plain dict is enough
# # here; swap for a proper DI container if this grows further.
# state: dict = {}


# # ---------------------------------------------------------
# # SQLite logging
# # ---------------------------------------------------------
# def init_db() -> None:
#     db_directory = os.path.dirname(DB_PATH)
#     if db_directory:
#         os.makedirs(db_directory, exist_ok=True)
#     conn = sqlite3.connect(DB_PATH)
#     conn.execute(
#         """
#         CREATE TABLE IF NOT EXISTS interactions (
#             id INTEGER PRIMARY KEY AUTOINCREMENT,
#             timestamp TEXT NOT NULL,
#             question TEXT NOT NULL,
#             in_scope INTEGER NOT NULL,
#             used_fallback INTEGER NOT NULL,
#             retrieved_rule_ids TEXT,
#             answer TEXT,
#             latency_ms REAL NOT NULL,
#             viz_shown INTEGER NOT NULL DEFAULT 0,
#             viz_type TEXT
#         )
#         """
#     )
#     # Additive migration for databases created before viz columns existed -
#     # CREATE TABLE IF NOT EXISTS above is a no-op on an existing table, so
#     # older interactions.db files need these added explicitly rather than
#     # silently missing the columns on every insert.
#     existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(interactions)")}
#     if "viz_shown" not in existing_cols:
#         conn.execute("ALTER TABLE interactions ADD COLUMN viz_shown INTEGER NOT NULL DEFAULT 0")
#     if "viz_type" not in existing_cols:
#         conn.execute("ALTER TABLE interactions ADD COLUMN viz_type TEXT")
#     conn.commit()
#     conn.close()


# def log_interaction(question: str, in_scope: bool, used_fallback: bool,
#                      retrieved_rule_ids: list, answer: str,
#                      latency_ms: float, viz_shown: bool = False,
#                      viz_type: Optional[str] = None) -> None:
#     conn = sqlite3.connect(DB_PATH)
#     conn.execute(
#         "INSERT INTO interactions "
#         "(timestamp, question, in_scope, used_fallback, retrieved_rule_ids, "
#         "answer, latency_ms, viz_shown, viz_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
#         (
#             datetime.now(timezone.utc).isoformat(),
#             question,
#             int(in_scope),
#             int(used_fallback),
#             ",".join(str(r) for r in retrieved_rule_ids),
#             answer,
#             latency_ms,
#             int(viz_shown),
#             viz_type,
#         ),
#     )
#     conn.commit()
#     conn.close()


# # ---------------------------------------------------------
# # App lifecycle
# # ---------------------------------------------------------
# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     state["data"] = load_rules(RULES_PATH)
#     state["domain_background"] = load_domain_background()
#     state["brain_mapping"] = load_brain_mapping()
#     try:
#         state["channel_atlas"] = load_channel_atlas()
#     except FileNotFoundError:
#         # Don't crash startup over a missing atlas - /ask still works for
#         # plain text answers, visualization tool calls just degrade to
#         # "no matching channels" rather than 500ing.
#         state["channel_atlas"] = {}
#     try:
#         state["client"] = get_client()
#         state["client_error"] = None
#     except RuntimeError as e:
#         # Don't crash the whole API just because OLLAMA_API_KEY isn't set
#         # yet - /health should be able to report this, and /ask should
#         # still serve the no-LLM fallback rather than 500ing on startup.
#         state["client"] = None
#         state["client_error"] = str(e)
#     init_db()
#     yield
#     state.clear()


# app = FastAPI(title="fNIRS Fuzzy Rule Explainer API", lifespan=lifespan)
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:5173",
#         "http://127.0.0.1:5173",
#         "http://localhost:5174",
#         "http://127.0.0.1:5174",
#         "http://localhost:8501",   # Streamlit's default dev port
#         "http://127.0.0.1:8501",
#         "https://fnirsragweb7f3a26.z1.web.core.windows.net",
#     ],
#     allow_credentials=True,
#     allow_methods=["GET", "POST"],
#     allow_headers=["Content-Type"],
# )


# # ---------------------------------------------------------
# # Schemas
# # ---------------------------------------------------------
# class AskRequest(BaseModel):
#     question: str


# class RuleOut(BaseModel):
#     rule_id: int | str  # stored as int in extracted_rules_final_raw.json
#     antecedent: str
#     consequent: str
#     accuracy: Optional[float] = None
#     dominance_score: Optional[float] = None


# class TopomapPoint(BaseModel):
#     feature: str
#     bare_channel: str
#     signal: str
#     level: str
#     x: float
#     y: float
#     x3d: Optional[float] = None
#     y3d: Optional[float] = None
#     z3d: Optional[float] = None
#     functional_region: str
#     hemisphere: str
#     rule_id: int | str
#     consequent: str
#     accuracy: Optional[float] = None


# class ChordNode(BaseModel):
#     channel: str
#     functional_region: str
#     hemisphere: str


# class ChordEdge(BaseModel):
#     source: str
#     target: str
#     rule_id: int | str
#     consequent: str
#     dominance_score: Optional[float] = None
#     accuracy: Optional[float] = None


# class ChordData(BaseModel):
#     nodes: list[ChordNode]
#     edges: list[ChordEdge]


# class NeighborChannel(BaseModel):
#     channel: str
#     x: float
#     y: float
#     x3d: Optional[float] = None
#     y3d: Optional[float] = None
#     z3d: Optional[float] = None
#     functional_region: str
#     hemisphere: str


# class NeighborPosition(BaseModel):
#     x: float
#     y: float
#     x3d: Optional[float] = None
#     y3d: Optional[float] = None
#     z3d: Optional[float] = None


# class NeighborsData(BaseModel):
#     channel: Optional[str] = None
#     found: bool
#     position: Optional[NeighborPosition] = None
#     neighbors: list[NeighborChannel] = []


# class VisualizationOut(BaseModel):
#     type: str  # "topomap" | "chord" | "neighbors"
#     topomap: Optional[list[TopomapPoint]] = None
#     chord: Optional[ChordData] = None
#     neighbors: Optional[NeighborsData] = None


# class Citation(BaseModel):
#     citation: str
#     url: str
#     summary: str


# class AskResponse(BaseModel):
#     answer: str
#     retrieved: list[RuleOut]
#     in_scope: bool
#     scope_tier: str  # "grounded" | "general" | "out_of_scope"
#     used_fallback: bool
#     latency_ms: float
#     visualization: Optional[VisualizationOut] = None
#     citations: list[Citation] = []


# class RulesResponse(BaseModel):
#     target_name: str
#     n_rules: int
#     rules: list[RuleOut]
#     cv_performance_estimate: Optional[dict] = None
#     n_seeds: int


# # ---------------------------------------------------------
# # Endpoints
# # ---------------------------------------------------------
# @app.get("/health")
# def health():
#     return {
#         "status": "ok" if state.get("client") else "degraded",
#         "llm_client_error": state.get("client_error"),
#         "n_rules": len(state["data"]["rules"]) if state.get("data") else 0,
#         "channel_atlas_loaded": bool(state.get("channel_atlas")),
#     }


# @app.get("/rules", response_model=RulesResponse)
# def get_rules():
#     data = state["data"]
#     return RulesResponse(
#         target_name=data["target_name"],
#         n_rules=len(data["rules"]),
#         rules=data["rules"],
#         cv_performance_estimate=data.get("cv_performance_estimate"),
#         n_seeds=data.get("n_seeds", 1),
#     )


# @app.get("/atlas")
# def get_atlas():
#     """Full static 36-channel layout (position, region, hemisphere) plus
#     a nearest-neighbour mesh graph for the montage's grid lines. This is
#     NOT rule-dependent - it's the same every time by design, so the
#     frontend should cache it (see api_client.py's ttl on this call) and
#     use it as a fixed background layer, with the per-question/overview
#     topomap points drawn on top as highlights."""
#     return state["channel_atlas"]


# @app.get("/visualize/topomap", response_model=list[TopomapPoint])
# def visualize_topomap_overview():
#     """Topomap over EVERY rule in the model, not just ones matching a
#     question - this is what the frontend's 'all rules' overview panel
#     uses, since it needs to exist without a chat turn ever happening.
#     Same builder as the per-question path in ask_with_visualization, so
#     there's exactly one code path that decides what a topomap point is."""
#     data = state["data"]
#     requested = resolve_requested_channels([], data["rules"])
#     return build_topomap_payload(data["rules"], requested, state["channel_atlas"])


# @app.get("/visualize/chord", response_model=ChordData)
# def visualize_chord_overview():
#     """Chord diagram over EVERY rule in the model - see visualize_topomap_overview."""
#     data = state["data"]
#     requested = resolve_requested_channels([], data["rules"])
#     return build_chord_payload(data["rules"], requested, state["channel_atlas"])


# @app.post("/ask", response_model=AskResponse)
# def ask_endpoint(req: AskRequest):
#     question = req.question.strip()
#     if not question:
#         raise HTTPException(status_code=422, detail="question must not be empty")

#     data = state["data"]
#     start = time.perf_counter()

#     # --- Three-way scope classification (was a binary in/out check) ---
#     # "grounded"     -> unchanged path: answer from this model's own rules/
#     #                   channels/performance, with viz tools available.
#     # "general"      -> NEW: neuroscience/fNIRS-adjacent but not about this
#     #                   specific model - answered with clearly-labeled
#     #                   general background instead of a flat refusal.
#     # "out_of_scope" -> unchanged: canned refusal, no LLM call spent.
#     tier = classify_scope(question, data["rules"], data["feature_names"])

#     if tier == "out_of_scope":
#         answer = (
#             "I can only answer questions about this model's rules, its "
#             "channels, and its performance, or general neuroscience/fNIRS "
#             "background - that question looks outside both, so I don't "
#             "have a grounded answer for it."
#         )
#         latency_ms = (time.perf_counter() - start) * 1000
#         log_interaction(question, False, False, [], answer, latency_ms)
#         return AskResponse(answer=answer, retrieved=[], in_scope=False,
#                             scope_tier="out_of_scope", used_fallback=False,
#                             latency_ms=latency_ms, visualization=None)

#     if tier == "general":
#         citations = []
#         if state.get("client") is None:
#             answer = (
#                 "That's a general neuroscience background question, but "
#                 "the explanation model isn't available right now, so I "
#                 "can't answer it without a plain refusal or a possibly "
#                 "unhedged guess - neither of which I want to give you. "
#                 "Please try again once the model is reachable."
#             )
#             used_fallback = True
#         else:
#             try:
#                 result = ask_general_neuro(
#                     question, state["client"],
#                     domain_background=state["domain_background"],
#                 )
#                 answer = result["answer"]
#                 citations = result.get("citations", [])
#                 used_fallback = False
#             except Exception:
#                 answer = (
#                     "The explanation model is unavailable right now, so I "
#                     "can't answer this general background question "
#                     "reliably at the moment - please try again shortly."
#                 )
#                 used_fallback = True
#         latency_ms = (time.perf_counter() - start) * 1000
#         log_interaction(question, True, used_fallback, [], answer, latency_ms)
#         return AskResponse(answer=answer, retrieved=[], in_scope=True,
#                             scope_tier="general", used_fallback=used_fallback,
#                             latency_ms=latency_ms, visualization=None,
#                             citations=citations)

#     # tier == "grounded" - unchanged behaviour from here down.
#     used_fallback = False
#     visualization = None

#     if state.get("client") is None:
#         # No LLM available at all - deterministic retrieval-only fallback,
#         # no tool-calling possible without a client to call.
#         retrieved = retrieve_rules(question, data["rules"], data["feature_names"])
#         answer = build_fallback_answer(retrieved, data["target_name"])
#         used_fallback = True
#     else:
#         try:
#             result = ask_with_visualization(
#                 question, data, state["client"],
#                 domain_background=state["domain_background"],
#                 brain_mapping=state["brain_mapping"],
#                 channel_atlas=state["channel_atlas"],
#             )
#             answer = result["answer"]
#             retrieved = result["retrieved"]
#             visualization = result["visualization"]
#         except Exception:
#             # LLM API down / timed out / rate-limited - degrade instead
#             # of returning a 500 to the frontend. Retrieval alone doesn't
#             # need the client, so it's re-run standalone here.
#             retrieved = retrieve_rules(question, data["rules"], data["feature_names"])
#             answer = build_fallback_answer(retrieved, data["target_name"])
#             used_fallback = True

#     latency_ms = (time.perf_counter() - start) * 1000
#     viz_shown = visualization is not None
#     log_interaction(
#         question, True, used_fallback,
#         [r["rule_id"] for r in retrieved], answer, latency_ms,
#         viz_shown=viz_shown,
#         viz_type=visualization["type"] if visualization else None,
#     )
#     return AskResponse(
#         answer=answer, retrieved=retrieved, in_scope=True,
#         scope_tier="grounded", used_fallback=used_fallback,
#         latency_ms=latency_ms, visualization=visualization,
#     )

"""
backend.py

FastAPI layer over rag_core.py's retrieval + prompt + LLM + visualization
logic. This decouples the "explain a fuzzy rule" service from any
particular frontend (Streamlit today, potentially React later) and gives
the system an actual measurable API - relevant for the proposal's own
"Backend Latency < 500ms" and "Response time < 3s for 95% of queries"
success criteria, neither of which can be measured against a bare
Streamlit script.

What this adds on top of what rag_core.py already does:
  - /ask, /rules, /health as real, documented endpoints (see /docs)
  - scope-guarding: off-domain questions get a canned response instead
    of spending an LLM call and risking an improvised answer
  - no-LLM fallback: if the Ollama call errors/times out, the API still
    returns something useful (the matching rules verbatim) instead of
    failing outright
  - visualization tool-calling: /ask now uses ask_with_visualization(),
    which may run a second LLM call when the model requests a topomap or
    chord diagram (see rag_core.py's module docstring). The response
    schema's `visualization` field carries structured JSON (channel
    points / chord nodes+edges), NOT a rendered image - deliberately, so
    a Streamlit frontend and a future React+D3 frontend can both consume
    the same endpoint without the backend caring which one asked.
  - latency timing on every request, in the response body
  - every interaction logged to SQLite (interactions.db) - question,
    whether it was in scope, whether the fallback fired, which rule IDs
    were retrieved, whether/which visualization was shown, the answer,
    and latency. This is the data you'll need later for task-completion /
    response-accuracy / latency reporting, so it's captured from day one
    rather than reconstructed after the fact.

Run with:
    uvicorn backend:app --reload
Then see interactive docs at:
    http://127.0.0.1:8000/docs
"""

import sqlite3
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from rag_core import (
    RULES_PATH, CLOUD_MODEL,
    load_rules, load_domain_background, load_brain_mapping, get_client,
    retrieve_rules, build_prompt, build_stability_lookup,
    is_in_scope, classify_scope, build_fallback_answer,
    ask_with_visualization, ask_general_neuro, describe_issue,
)
from viz_tools import (
    load_channel_atlas, resolve_requested_channels,
    build_topomap_payload, build_chord_payload,
)

DB_PATH = "interactions.db"

# Loaded once at process startup - the FastAPI equivalent of
# streamlit_app.py's @st.cache_resource init(). A plain dict is enough
# here; swap for a proper DI container if this grows further.
state: dict = {}


# ---------------------------------------------------------
# SQLite logging
# ---------------------------------------------------------
def init_db() -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            question TEXT NOT NULL,
            in_scope INTEGER NOT NULL,
            used_fallback INTEGER NOT NULL,
            retrieved_rule_ids TEXT,
            answer TEXT,
            latency_ms REAL NOT NULL,
            viz_shown INTEGER NOT NULL DEFAULT 0,
            viz_type TEXT
        )
        """
    )
    # Additive migration for databases created before viz columns existed -
    # CREATE TABLE IF NOT EXISTS above is a no-op on an existing table, so
    # older interactions.db files need these added explicitly rather than
    # silently missing the columns on every insert.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(interactions)")}
    if "viz_shown" not in existing_cols:
        conn.execute("ALTER TABLE interactions ADD COLUMN viz_shown INTEGER NOT NULL DEFAULT 0")
    if "viz_type" not in existing_cols:
        conn.execute("ALTER TABLE interactions ADD COLUMN viz_type TEXT")
    conn.commit()
    conn.close()


def log_interaction(question: str, in_scope: bool, used_fallback: bool,
                     retrieved_rule_ids: list, answer: str,
                     latency_ms: float, viz_shown: bool = False,
                     viz_type: Optional[str] = None) -> None:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO interactions "
        "(timestamp, question, in_scope, used_fallback, retrieved_rule_ids, "
        "answer, latency_ms, viz_shown, viz_type) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            datetime.now(timezone.utc).isoformat(),
            question,
            int(in_scope),
            int(used_fallback),
            ",".join(str(r) for r in retrieved_rule_ids),
            answer,
            latency_ms,
            int(viz_shown),
            viz_type,
        ),
    )
    conn.commit()
    conn.close()


# ---------------------------------------------------------
# App lifecycle
# ---------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    state["data"] = load_rules(RULES_PATH)
    state["domain_background"] = load_domain_background()
    state["brain_mapping"] = load_brain_mapping()
    try:
        state["channel_atlas"] = load_channel_atlas()
    except FileNotFoundError:
        # Don't crash startup over a missing atlas - /ask still works for
        # plain text answers, visualization tool calls just degrade to
        # "no matching channels" rather than 500ing.
        state["channel_atlas"] = {}
    try:
        state["client"] = get_client()
        state["client_error"] = None
    except RuntimeError as e:
        # Don't crash the whole API just because OLLAMA_API_KEY isn't set
        # yet - /health should be able to report this, and /ask should
        # still serve the no-LLM fallback rather than 500ing on startup.
        state["client"] = None
        state["client_error"] = str(e)
    init_db()
    yield
    state.clear()


app = FastAPI(title="fNIRS Fuzzy Rule Explainer API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:5174",
        "http://127.0.0.1:5174",
        "http://localhost:8501",   # Streamlit's default dev port
        "http://127.0.0.1:8501",
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ---------------------------------------------------------
# Schemas
# ---------------------------------------------------------
class AskRequest(BaseModel):
    question: str


class RuleOut(BaseModel):
    rule_id: int | str  # stored as int in extracted_rules_final_raw.json
    antecedent: str
    consequent: str
    accuracy: Optional[float] = None
    dominance_score: Optional[float] = None


class TopomapPoint(BaseModel):
    feature: str
    bare_channel: str
    signal: str
    level: str
    x: float
    y: float
    x3d: Optional[float] = None
    y3d: Optional[float] = None
    z3d: Optional[float] = None
    functional_region: str
    hemisphere: str
    rule_id: int | str
    consequent: str
    accuracy: Optional[float] = None


class ChordNode(BaseModel):
    channel: str
    functional_region: str
    hemisphere: str


class ChordEdge(BaseModel):
    source: str
    target: str
    rule_id: int | str
    consequent: str
    dominance_score: Optional[float] = None
    accuracy: Optional[float] = None


class ChordData(BaseModel):
    nodes: list[ChordNode]
    edges: list[ChordEdge]


class NeighborChannel(BaseModel):
    channel: str
    x: float
    y: float
    x3d: Optional[float] = None
    y3d: Optional[float] = None
    z3d: Optional[float] = None
    functional_region: str
    hemisphere: str


class NeighborPosition(BaseModel):
    x: float
    y: float
    x3d: Optional[float] = None
    y3d: Optional[float] = None
    z3d: Optional[float] = None


class NeighborsData(BaseModel):
    channel: Optional[str] = None
    found: bool
    position: Optional[NeighborPosition] = None
    neighbors: list[NeighborChannel] = []


class VisualizationOut(BaseModel):
    type: str  # "topomap" | "chord" | "neighbors"
    topomap: Optional[list[TopomapPoint]] = None
    chord: Optional[ChordData] = None
    neighbors: Optional[NeighborsData] = None


class Citation(BaseModel):
    citation: str
    url: str
    summary: str


class AskResponse(BaseModel):
    answer: str
    retrieved: list[RuleOut]
    in_scope: bool
    scope_tier: str  # "grounded" | "general" | "out_of_scope"
    used_fallback: bool
    latency_ms: float
    visualization: Optional[VisualizationOut] = None
    citations: list[Citation] = []
    # Set when the post-generation consistency check found (and could not
    # fully fix) a contradiction between the answer and the rule data.
    consistency_warning: Optional[str] = None
    # True when the first draft was flagged and automatically corrected.
    consistency_repaired: bool = False


class RulesResponse(BaseModel):
    target_name: str
    n_rules: int
    rules: list[RuleOut]
    cv_performance_estimate: Optional[dict] = None
    n_seeds: int


# ---------------------------------------------------------
# Endpoints
# ---------------------------------------------------------
@app.get("/health")
def health():
    return {
        "status": "ok" if state.get("client") else "degraded",
        "llm_client_error": state.get("client_error"),
        "n_rules": len(state["data"]["rules"]) if state.get("data") else 0,
        "channel_atlas_loaded": bool(state.get("channel_atlas")),
    }


@app.get("/rules", response_model=RulesResponse)
def get_rules():
    data = state["data"]
    return RulesResponse(
        target_name=data["target_name"],
        n_rules=len(data["rules"]),
        rules=data["rules"],
        cv_performance_estimate=data.get("cv_performance_estimate"),
        n_seeds=data.get("n_seeds", 1),
    )


@app.get("/atlas")
def get_atlas():
    """Full static 36-channel layout (position, region, hemisphere) plus
    a nearest-neighbour mesh graph for the montage's grid lines. This is
    NOT rule-dependent - it's the same every time by design, so the
    frontend should cache it (see api_client.py's ttl on this call) and
    use it as a fixed background layer, with the per-question/overview
    topomap points drawn on top as highlights."""
    return state["channel_atlas"]


@app.get("/visualize/topomap", response_model=list[TopomapPoint])
def visualize_topomap_overview():
    """Topomap over EVERY rule in the model, not just ones matching a
    question - this is what the frontend's 'all rules' overview panel
    uses, since it needs to exist without a chat turn ever happening.
    Same builder as the per-question path in ask_with_visualization, so
    there's exactly one code path that decides what a topomap point is."""
    data = state["data"]
    requested = resolve_requested_channels([], data["rules"])
    return build_topomap_payload(data["rules"], requested, state["channel_atlas"])


@app.get("/visualize/chord", response_model=ChordData)
def visualize_chord_overview():
    """Chord diagram over EVERY rule in the model - see visualize_topomap_overview."""
    data = state["data"]
    requested = resolve_requested_channels([], data["rules"])
    return build_chord_payload(data["rules"], requested, state["channel_atlas"])


@app.post("/ask", response_model=AskResponse)
def ask_endpoint(req: AskRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")

    data = state["data"]
    start = time.perf_counter()

    # --- Three-way scope classification (was a binary in/out check) ---
    # "grounded"     -> unchanged path: answer from this model's own rules/
    #                   channels/performance, with viz tools available.
    # "general"      -> NEW: neuroscience/fNIRS-adjacent but not about this
    #                   specific model - answered with clearly-labeled
    #                   general background instead of a flat refusal.
    # "out_of_scope" -> unchanged: canned refusal, no LLM call spent.
    tier = classify_scope(question, data["rules"], data["feature_names"])

    if tier == "out_of_scope":
        answer = (
            "I can only answer questions about this model's rules, its "
            "channels, and its performance, or general neuroscience/fNIRS "
            "background - that question looks outside both, so I don't "
            "have a grounded answer for it."
        )
        latency_ms = (time.perf_counter() - start) * 1000
        log_interaction(question, False, False, [], answer, latency_ms)
        return AskResponse(answer=answer, retrieved=[], in_scope=False,
                            scope_tier="out_of_scope", used_fallback=False,
                            latency_ms=latency_ms, visualization=None)

    if tier == "general":
        citations = []
        if state.get("client") is None:
            answer = (
                "That's a general neuroscience background question, but "
                "the explanation model isn't available right now, so I "
                "can't answer it without a plain refusal or a possibly "
                "unhedged guess - neither of which I want to give you. "
                "Please try again once the model is reachable."
            )
            used_fallback = True
        else:
            try:
                result = ask_general_neuro(
                    question, state["client"],
                    domain_background=state["domain_background"],
                )
                answer = result["answer"]
                citations = result.get("citations", [])
                used_fallback = False
            except Exception:
                answer = (
                    "The explanation model is unavailable right now, so I "
                    "can't answer this general background question "
                    "reliably at the moment - please try again shortly."
                )
                used_fallback = True
        latency_ms = (time.perf_counter() - start) * 1000
        log_interaction(question, True, used_fallback, [], answer, latency_ms)
        return AskResponse(answer=answer, retrieved=[], in_scope=True,
                            scope_tier="general", used_fallback=used_fallback,
                            latency_ms=latency_ms, visualization=None,
                            citations=citations)

    # tier == "grounded" - unchanged behaviour from here down.
    used_fallback = False
    visualization = None
    consistency_warning = None
    consistency_repaired = False

    if state.get("client") is None:
        # No LLM available at all - deterministic retrieval-only fallback,
        # no tool-calling possible without a client to call.
        retrieved = retrieve_rules(question, data["rules"], data["feature_names"])
        answer = build_fallback_answer(retrieved, data["target_name"])
        used_fallback = True
    else:
        try:
            result = ask_with_visualization(
                question, data, state["client"],
                domain_background=state["domain_background"],
                brain_mapping=state["brain_mapping"],
                channel_atlas=state["channel_atlas"],
            )
            answer = result["answer"]
            retrieved = result["retrieved"]
            visualization = result["visualization"]
            consistency = result.get("consistency") or {}
            consistency_repaired = bool(consistency.get("repaired"))
            remaining = consistency.get("issues") or []
            if remaining:
                consistency_warning = (
                    "Automatic check: this answer may contain an error - "
                    + "; ".join(describe_issue(i) for i in remaining)
                    + ". Please rely on the rules shown under 'Rules used'."
                )
        except Exception:
            # LLM API down / timed out / rate-limited - degrade instead
            # of returning a 500 to the frontend. Retrieval alone doesn't
            # need the client, so it's re-run standalone here.
            retrieved = retrieve_rules(question, data["rules"], data["feature_names"])
            answer = build_fallback_answer(retrieved, data["target_name"])
            used_fallback = True

    latency_ms = (time.perf_counter() - start) * 1000
    viz_shown = visualization is not None
    log_interaction(
        question, True, used_fallback,
        [r["rule_id"] for r in retrieved], answer, latency_ms,
        viz_shown=viz_shown,
        viz_type=visualization["type"] if visualization else None,
    )
    return AskResponse(
        answer=answer, retrieved=retrieved, in_scope=True,
        scope_tier="grounded", used_fallback=used_fallback,
        latency_ms=latency_ms, visualization=visualization,
        consistency_warning=consistency_warning,
        consistency_repaired=consistency_repaired,
    )