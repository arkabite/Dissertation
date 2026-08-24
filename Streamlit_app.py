# """
# streamlit_app.py

# Chat UI for the ExFuzzy rule explainer. Talks ONLY to backend.py's
# FastAPI service via api_client.py - it does not import rag_core, does
# not construct an Ollama client, and does not render matplotlib/mne
# figures locally. Every answer, retrieved rule, and visualization comes
# from the API, so this file only needs to know how to display JSON.

# That's a deliberate choice, not an accident: it's what makes this
# frontend swappable for React later without touching the backend at all
# (see api_client.py's module docstring).

# Run the backend first:
#     uvicorn backend:app --reload
# Then run this:
#     streamlit run streamlit_app.py
# """

# import streamlit as st

# import api_client
# from api_client import BackendError
# from viz_components import build_topomap_spec, build_chord_html, build_brain3d_html

# st.set_page_config(page_title="Fuzzy Rule Explainer", layout="wide")

# # ---------------------------------------------------------
# # Light theming touch - kept minimal on purpose, this is a research
# # usability-study tool, not a marketing site. Just enough to stop it
# # looking like an untouched default Streamlit page.
# # ---------------------------------------------------------
# st.markdown(
#     """
#     <style>
#     .block-container { padding-top: 2rem; }
#     [data-testid="stChatMessage"] { border-radius: 10px; }
#     .viz-caption { color: #666; font-size: 0.85rem; margin-top: -0.5rem; }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# st.title("fNIRS Fuzzy Rule Explainer")


# # ---------------------------------------------------------
# # Backend connectivity check - fail loudly and specifically rather than
# # letting every widget below throw its own confusing error.
# # ---------------------------------------------------------
# @st.cache_resource
# def check_backend():
#     return api_client.health()


# try:
#     health = check_backend()
# except BackendError as e:
#     st.error(
#         f"Can't reach the backend API.\n\n{e}\n\n"
#         f"Make sure it's running: `uvicorn backend:app --reload` "
#         f"(default expected at {api_client.BACKEND_URL})."
#     )
#     st.stop()

# if health.get("status") != "ok":
#     st.warning(
#         f"Backend is reachable but degraded: {health.get('llm_client_error')}. "
#         "Questions will still work using rule-matching only, without "
#         "natural-language explanations or visualizations."
#     )


# @st.cache_data(ttl=300)
# def load_rules_overview():
#     return api_client.get_rules()


# @st.cache_data(ttl=3600)  # static layout - safe to cache much longer
# def load_atlas():
#     return api_client.get_atlas()


# @st.cache_data(ttl=300)
# def load_overview_topomap():
#     return api_client.get_overview_topomap()


# @st.cache_data(ttl=300)
# def load_overview_chord():
#     return api_client.get_overview_chord()


# data = load_rules_overview()

# # ---------------------------------------------------------
# # Sidebar: model overview + CV performance + all-rules visualizations
# # ---------------------------------------------------------
# with st.sidebar:
#     st.subheader("Model overview")
#     st.write(f"**{data['n_rules']} rules** over target `{data['target_name']}`")

#     cv = data.get("cv_performance_estimate")
#     if cv:
#         st.metric("Test MCC", f"{cv['test_mcc_mean']:.3f}", f"± {cv['test_mcc_std']:.3f}")
#         st.metric("Test accuracy", f"{cv['test_accuracy_mean']:.1%}", f"± {cv['test_accuracy_std']:.1%}")
#         st.caption(
#             f"{cv['n_folds']}-fold subject-grouped cross-validation. "
#             "MCC is shown first since accuracy alone sits close to the "
#             "majority-class baseline here."
#         )

#     with st.expander("View all rules"):
#         for r in data["rules"]:
#             st.markdown(
#                 f"**Rule {r['rule_id']}**: IF {r['antecedent']} "
#                 f"THEN {data['target_name']} = {r['consequent']}  \n"
#                 f"DS={r.get('dominance_score', float('nan')):.3f}, "
#                 f"ACC={r.get('accuracy', float('nan')):.3f}"
#             )

#     with st.expander("Channel map (all rules)"):
#         st.caption(
#             "Every channel referenced across the full rule set, coloured "
#             "by activation level. Shape distinguishes HbO (circle/sphere) "
#             "from HbR (diamond/cube)."
#         )
#         view = st.radio("View", ["2D scalp map", "3D headspace"],
#                          horizontal=True, key="overview_view_mode")
#         try:
#             points = load_overview_topomap()
#             if points:
#                 if view == "2D scalp map":
#                     atlas = load_atlas()
#                     st.vega_lite_chart(build_topomap_spec(points, atlas=atlas), use_container_width=True)
#                 else:
#                     st.components.v1.html(build_brain3d_html(points), height=520, scrolling=False)
#             else:
#                 st.info("No channel data available.")
#         except BackendError as e:
#             st.warning(f"Couldn't load the overview topomap: {e}")

#     with st.expander("Connectivity (all rules)"):
#         st.caption(
#             "Which channels co-occur together within the same rule. A "
#             "connection means the two channels appear ANDed in a rule's "
#             "condition - not a measured physiological connectivity claim."
#         )
#         try:
#             chord = load_overview_chord()
#             if chord.get("edges"):
#                 st.components.v1.html(build_chord_html(chord), height=520, scrolling=False)
#             else:
#                 st.info("No co-occurring channel pairs found across the rule set.")
#         except BackendError as e:
#             st.warning(f"Couldn't load the overview chord diagram: {e}")

#     st.divider()
#     if st.button("Clear chat"):
#         st.session_state.history = []
#         st.rerun()


# # ---------------------------------------------------------
# # Chat history + rendering
# # ---------------------------------------------------------
# if "history" not in st.session_state:
#     st.session_state.history = []


# def render_visualization(visualization: dict, key_prefix: str) -> None:
#     if visualization is None:
#         return
#     viz_type = visualization.get("type")
#     if viz_type == "topomap" and visualization.get("topomap"):
#         view = st.radio("View", ["2D scalp map", "3D headspace"],
#                          horizontal=True, key=f"{key_prefix}_view_mode")
#         if view == "2D scalp map":
#             st.vega_lite_chart(
#                 build_topomap_spec(visualization["topomap"], title="Channels behind this answer",
#                                     atlas=load_atlas()),
#                 use_container_width=True,
#             )
#             st.markdown(
#                 '<p class="viz-caption">Shape = signal (HbO circle / HbR diamond). '
#                 "Color = activation level.</p>",
#                 unsafe_allow_html=True,
#             )
#         else:
#             st.components.v1.html(build_brain3d_html(visualization["topomap"]), height=520, scrolling=False)
#             st.markdown(
#                 '<p class="viz-caption">Drag to rotate. Shape = signal '
#                 "(HbO sphere / HbR cube). Color = activation level.</p>",
#                 unsafe_allow_html=True,
#             )
#     elif viz_type == "chord" and visualization.get("chord", {}).get("edges"):
#         st.components.v1.html(build_chord_html(visualization["chord"]), height=520, scrolling=False)
#         st.markdown(
#             '<p class="viz-caption">Hover a ribbon for the rule(s) behind that connection.</p>',
#             unsafe_allow_html=True,
#         )


# def render_message(role: str, content: str, retrieved: list = None,
#                     visualization: dict = None, key_prefix: str = "live") -> None:
#     with st.chat_message(role):
#         st.markdown(content)
#         if visualization is not None:
#             render_visualization(visualization, key_prefix)
#         if retrieved:
#             with st.expander(f"Rules used ({len(retrieved)})"):
#                 for r in retrieved:
#                     st.caption(
#                         f"Rule {r['rule_id']}: {r['antecedent']} → {r['consequent']} "
#                         f"(ACC={r.get('accuracy', float('nan')):.3f})"
#                     )


# for i, entry in enumerate(st.session_state.history):
#     render_message(entry["role"], entry["content"], entry.get("retrieved"),
#                     entry.get("visualization"), key_prefix=f"hist_{i}")


# # ---------------------------------------------------------
# # Chat input
# # ---------------------------------------------------------
# if question := st.chat_input("Ask about the model's rules..."):
#     st.session_state.history.append({"role": "user", "content": question})
#     with st.chat_message("user"):
#         st.markdown(question)

#     with st.spinner("Thinking..."):
#         try:
#             result = api_client.ask(question)
#             answer = result["answer"]
#             retrieved = result.get("retrieved", [])
#             visualization = result.get("visualization")
#             if result.get("used_fallback"):
#                 st.toast("The explanation model was unavailable - showing matched rules directly.")
#         except BackendError as e:
#             answer = f"Something went wrong reaching the backend: {e}"
#             retrieved, visualization = [], None

#     render_message("assistant", answer, retrieved, visualization)
#     st.session_state.history.append({
#         "role": "assistant", "content": answer,
#         "retrieved": retrieved, "visualization": visualization,
#     })

"""
streamlit_app.py

Chat UI for the ExFuzzy rule explainer. This single-app version calls
rag_core.py and viz_tools.py directly, so it does not require a separate
FastAPI service or Uvicorn process. The app owns model loading, retrieval,
LLM calls, fallback answers, and visualization payload construction.

Run this:
    streamlit run Streamlit_app.py

Theme note: the palette lives in .streamlit/config.toml (warm ivory /
amber-brass "Amber Desk" theme) so Streamlit's own widgets pick it up.
The CSS block below only handles the few surfaces config.toml can't
reach (chat bubbles, expander borders, caption tone, fonts).
"""

import streamlit as st
import streamlit.components.v1 as components

from rag_core import (
    ask_with_visualization,
    build_fallback_answer,
    get_client,
    is_in_scope,
    load_brain_mapping,
    load_domain_background,
    load_rules,
    retrieve_rules,
)
from viz_tools import (
    build_chord_payload,
    build_topomap_payload,
    load_channel_atlas,
    resolve_requested_channels,
)
from viz_components import build_topomap_spec, build_chord_html, build_brain3d_html

st.set_page_config(page_title="Fuzzy Rule Explainer", layout="wide")

VIZ_HEIGHT = 520

# ---------------------------------------------------------
# Light theming touch - kept restrained on purpose, this is a research
# usability-study tool, not a marketing site. Warm paper tones + serif
# headings so it reads academic rather than default-Streamlit.
# ---------------------------------------------------------
# st.markdown(
#     """
#     <link rel="preconnect" href="https://fonts.googleapis.com">
#     <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
#     <link href="https://fonts.googleapis.com/css2?family=Lora:ital,wght@0,400;0,600;1,400&family=Nunito+Sans:wght@300;400;600;700&display=swap" rel="stylesheet">
#     <style>
#     :root {
#         --ad-bg: #FCF8F2;
#         --ad-panel: #EFE3CF;
#         --ad-accent: #B5761F;
#         --ad-ink: #332E28;
#         --ad-muted: #7A6E60;
#         --ad-line: #E2D6C2;
#     }

#     html, body, [class*="css"], .stMarkdown, .stTextInput, .stRadio label {
#         font-family: "Nunito Sans", "Helvetica Neue", Arial, sans-serif;
#         color: var(--ad-ink);
#     }

#     h1, h2, h3, h4,
#     [data-testid="stSidebar"] h1,
#     [data-testid="stSidebar"] h2,
#     [data-testid="stSidebar"] h3 {
#         font-family: "Lora", Georgia, serif;
#         font-weight: 600;
#         letter-spacing: -0.005em;
#         color: var(--ad-ink);
#     }

#     .block-container { padding-top: 2.25rem; max-width: 1180px; }

#     /* Page header block */
#     .ad-title { margin-bottom: 0.15rem; }
#     .ad-subtitle {
#         font-family: "Lora", Georgia, serif;
#         font-style: italic;
#         color: var(--ad-muted);
#         font-size: 1.02rem;
#         margin: 0 0 0.9rem 0;
#     }

#     /* Chat bubbles: warm paper cards with a hairline border */
#     [data-testid="stChatMessage"] {
#         border-radius: 12px;
#         border: 1px solid var(--ad-line);
#         background: #FFFDF9;
#         padding: 0.85rem 1.05rem;
#         box-shadow: 0 1px 2px rgba(51, 46, 40, 0.04);
#         margin-bottom: 0.7rem;
#     }
#     /* Assistant turns get an amber rule so the thread reads at a glance */
#     [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
#         border-left: 3px solid var(--ad-accent);
#         background: #FBF6EC;
#     }

#     /* Expanders / metrics: match the paper palette */
#     [data-testid="stExpander"] details {
#         border: 1px solid var(--ad-line);
#         border-radius: 10px;
#         background: #FFFDF9;
#     }
#     [data-testid="stExpander"] summary { font-weight: 600; }
#     [data-testid="stMetricValue"] {
#         font-family: "Lora", Georgia, serif;
#         color: var(--ad-ink);
#     }

#     .viz-caption {
#         color: var(--ad-muted);
#         font-size: 0.85rem;
#         line-height: 1.35;
#         margin: 0.35rem 0 0.9rem 0;
#     }

#     /* Aligned rule list entries in the sidebar */
#     .ad-rule { margin-bottom: 0.7rem; }
#     .ad-rule-head { font-weight: 700; font-size: 0.9rem; }
#     .ad-rule-body { font-size: 0.86rem; line-height: 1.4; }
#     .ad-rule-stats {
#         color: var(--ad-muted);
#         font-size: 0.78rem;
#         font-variant-numeric: tabular-nums;
#     }

#     [data-testid="stSidebar"] { border-right: 1px solid var(--ad-line); }
#     hr { border-color: var(--ad-line); }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

st.markdown('<h1 class="ad-title">fNIRS Fuzzy Rule Explainer</h1>', unsafe_allow_html=True)
st.markdown(
    '<p class="ad-subtitle">Ask about the fuzzy rules behind the classifier — answers are '
    "grounded in the extracted rule set, with the channels involved shown alongside.</p>",
    unsafe_allow_html=True,
)
st.divider()


# ---------------------------------------------------------
# Small formatting helper: the API may omit a stat or send null, and a
# bare f"{None:.3f}" raises. Keep display robust, not clever.
# ---------------------------------------------------------
def fmt_num(value, spec: str = ".3f", fallback: str = "n/a") -> str:
    try:
        if value is None:
            return fallback
        return format(float(value), spec)
    except (TypeError, ValueError):
        return fallback


def viz_caption(text: str) -> None:
    st.markdown(f'<p class="viz-caption">{text}</p>', unsafe_allow_html=True)


# ---------------------------------------------------------
# Chat history is initialised before the sidebar renders, so the
# "Clear chat" button can never touch an unset session key.
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []


# ---------------------------------------------------------
# Load the model directly in this Streamlit process. This replaces the
# separate FastAPI service for the single-app deployment.
# ---------------------------------------------------------
@st.cache_resource
def load_app_state():
    try:
        api_key = st.secrets.get("OLLAMA_API_KEY", "")
    except Exception:
        api_key = ""

    try:
        client = get_client(api_key=api_key)
        client_error = None
    except RuntimeError as e:
        client = None
        client_error = str(e)

    data = load_rules()
    atlas = load_channel_atlas()
    return {
        "data": data,
        "domain_background": load_domain_background(),
        "brain_mapping": load_brain_mapping(),
        "atlas": atlas,
        "client": client,
        "client_error": client_error,
    }


try:
    app_state = load_app_state()
except Exception as e:
    st.error(f"The model could not be loaded: {e}")
    st.stop()

data = app_state["data"]
target_name = data.get("target_name", "target")
rules = data.get("rules", [])
atlas = app_state["atlas"]

if app_state["client"] is None:
    st.warning(
        f"The explanation model is unavailable: {app_state['client_error']}. "
        "Questions will still use deterministic rule matching."
    )

overview_requested = resolve_requested_channels([], rules)
overview_topomap = build_topomap_payload(rules, overview_requested, atlas)
overview_chord = build_chord_payload(rules, overview_requested, atlas)

# ---------------------------------------------------------
# Sidebar: model overview + CV performance + all-rules visualizations
# ---------------------------------------------------------
with st.sidebar:
    st.subheader("Model overview")
    st.write(f"**{len(rules)} rules** over target `{target_name}`")

    cv = data.get("cv_performance_estimate") or {}
    if cv:
        st.subheader("Performance")
        left, right = st.columns(2)
        with left:
            st.metric(
                "Test MCC",
                fmt_num(cv.get("test_mcc_mean")),
                f"± {fmt_num(cv.get('test_mcc_std'))}",
                delta_color="off",  # a std-dev is not an improvement
            )
        with right:
            st.metric(
                "Test accuracy",
                fmt_num(cv.get("test_accuracy_mean"), ".1%"),
                f"± {fmt_num(cv.get('test_accuracy_std'), '.1%')}",
                delta_color="off",
            )
        st.caption(
            f"{cv.get('n_folds', 'n/a')}-fold subject-grouped cross-validation. "
            "MCC is shown first since accuracy alone sits close to the "
            "majority-class baseline here."
        )

    st.subheader("Rule set")
    with st.expander("View all rules"):
        for r in data.get("rules", []):
            st.markdown(
                '<div class="ad-rule">'
                f'<div class="ad-rule-head">Rule {r.get("rule_id", "?")}</div>'
                f'<div class="ad-rule-body">IF {r.get("antecedent", "")}<br>'
                f'THEN {target_name} = {r.get("consequent", "")}</div>'
                f'<div class="ad-rule-stats">DS {fmt_num(r.get("dominance_score"))} '
                f'&nbsp;·&nbsp; ACC {fmt_num(r.get("accuracy"))}</div>'
                "</div>",
                unsafe_allow_html=True,
            )

    st.subheader("Visualizations")
    with st.expander("Channel map (all rules)"):
        viz_caption(
            "Every channel referenced across the full rule set, coloured "
            "by activation level. Shape distinguishes HbO (circle/sphere) "
            "from HbR (diamond/cube)."
        )
        view = st.radio(
            "View",
            ["2D scalp map", "3D headspace"],
            horizontal=True,
            key="overview_view_mode",
        )
        if overview_topomap:
            if view == "2D scalp map":
                st.vega_lite_chart(
                    build_topomap_spec(overview_topomap, atlas=atlas),
                    use_container_width=True,
                )
            else:
                components.html(
                    build_brain3d_html(overview_topomap),
                    height=VIZ_HEIGHT,
                    scrolling=False,
                )
        else:
            st.info("No channel data available.")

    with st.expander("Connectivity (all rules)"):
        viz_caption(
            "Which channels co-occur together within the same rule. A "
            "connection means the two channels appear ANDed in a rule's "
            "condition — not a measured physiological connectivity claim."
        )
        if overview_chord.get("edges"):
            components.html(
                build_chord_html(overview_chord), height=VIZ_HEIGHT, scrolling=False
            )
        else:
            st.info("No co-occurring channel pairs found across the rule set.")

    st.divider()
    if st.button("Clear chat", use_container_width=True):
        st.session_state.history = []
        st.rerun()


# ---------------------------------------------------------
# Chat rendering
# ---------------------------------------------------------
def render_visualization(visualization: dict, key_prefix: str) -> None:
    if not visualization:
        return
    viz_type = visualization.get("type")

    if viz_type == "topomap" and visualization.get("topomap"):
        view = st.radio(
            "View",
            ["2D scalp map", "3D headspace"],
            horizontal=True,
            key=f"{key_prefix}_view_mode",
        )
        if view == "2D scalp map":
            st.vega_lite_chart(
                build_topomap_spec(
                    visualization["topomap"],
                    title="Channels behind this answer",
                    atlas=atlas,
                ),
                use_container_width=True,
            )
            viz_caption(
                "Shape = signal (HbO circle / HbR diamond). Color = activation level."
            )
        else:
            components.html(
                build_brain3d_html(visualization["topomap"]),
                height=VIZ_HEIGHT,
                scrolling=False,
            )
            viz_caption(
                "Drag to rotate. Shape = signal (HbO sphere / HbR cube). "
                "Color = activation level."
            )

    elif viz_type == "chord" and (visualization.get("chord") or {}).get("edges"):
        components.html(
            build_chord_html(visualization["chord"]), height=VIZ_HEIGHT, scrolling=False
        )
        viz_caption("Hover a ribbon for the rule(s) behind that connection.")


def render_message(
    role: str,
    content: str,
    retrieved: list = None,
    visualization: dict = None,
    key_prefix: str = "live",
) -> None:
    with st.chat_message(role):
        st.markdown(content)
        if visualization:
            render_visualization(visualization, key_prefix)
        if retrieved:
            with st.expander(f"Rules used ({len(retrieved)})"):
                for r in retrieved:
                    st.caption(
                        f"Rule {r.get('rule_id', '?')}: {r.get('antecedent', '')} → "
                        f"{r.get('consequent', '')} (ACC={fmt_num(r.get('accuracy'))})"
                    )


# ---------------------------------------------------------
# Chat input
#
# The new turn is appended to history and then the whole thread is
# rendered from history in one place. Rendering the fresh answer inline
# with a "live" key prefix and again as "hist_N" on the next run gave
# the 2D/3D toggle two different widget keys, so the user's choice reset
# itself on the following interaction.
# ---------------------------------------------------------
if question := st.chat_input("Ask about the model's rules..."):
    st.session_state.history.append({"role": "user", "content": question})

    with st.spinner("Thinking..."):
        if not is_in_scope(question, rules, data.get("feature_names", [])):
            answer = (
                "I can only answer questions about this model's rules, its "
                "channels, and its performance - that question looks outside "
                "that scope, so I don't have a grounded answer for it."
            )
            retrieved, visualization, used_fallback = [], None, False
        elif app_state["client"] is None:
            retrieved = retrieve_rules(question, rules, data["feature_names"])
            answer = build_fallback_answer(retrieved, target_name)
            visualization, used_fallback = None, True
        else:
            try:
                result = ask_with_visualization(
                    question,
                    data,
                    app_state["client"],
                    domain_background=app_state["domain_background"],
                    brain_mapping=app_state["brain_mapping"],
                    channel_atlas=atlas,
                )
                answer = result.get("answer") or "The model returned an empty answer."
                retrieved = result.get("retrieved", [])
                visualization = result.get("visualization")
                used_fallback = False
            except Exception as e:
                retrieved = retrieve_rules(question, rules, data["feature_names"])
                answer = build_fallback_answer(retrieved, target_name)
                visualization, used_fallback = None, True

    st.session_state.history.append(
        {
            "role": "assistant",
            "content": answer,
            "retrieved": retrieved,
            "visualization": visualization,
        }
    )

    if used_fallback:
        st.toast("The explanation model was unavailable - showing matched rules directly.")

for i, entry in enumerate(st.session_state.history):
    render_message(
        entry["role"],
        entry["content"],
        entry.get("retrieved"),
        entry.get("visualization"),
        key_prefix=f"hist_{i}",
    )
