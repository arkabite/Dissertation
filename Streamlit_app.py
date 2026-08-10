"""
streamlit_app.py

Chat UI for the ExFuzzy rule explainer. Wraps the already-verified
retrieval + prompt logic in rag_core.py.

Run with:
    streamlit run streamlit_app.py
"""

import os
from pathlib import Path

import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError

from rag_core import load_rules, get_client, ask, RULES_PATH


def load_api_key_from_local_file() -> str:
    candidates = [
        Path(__file__).resolve().parent / ".streamlit" / "secrets.toml",
        Path(__file__).resolve().parent / "secrets.toml",
        Path(__file__).resolve().parent / "streamlitlol" / "secrets.toml",
        Path.home() / ".streamlit" / "secrets.toml",
    ]

    for path in candidates:
        if not path.exists():
            continue
        import tomllib

        try:
            with path.open("rb") as fh:
                data = tomllib.load(fh)
        except Exception:
            continue

        value = data.get("OLLAMA_API_KEY", "")
        if value:
            return str(value)

    return ""


# Pull the API key from Streamlit secrets if available, but do not crash if
# Streamlit cannot parse the file or no secrets are configured.
try:
    if "OLLAMA_API_KEY" in st.secrets:
        os.environ["OLLAMA_API_KEY"] = st.secrets["OLLAMA_API_KEY"]
except (StreamlitSecretNotFoundError, Exception):
    os.environ.pop("OLLAMA_API_KEY", None)

if not os.environ.get("OLLAMA_API_KEY"):
    fallback_key = load_api_key_from_local_file()
    if fallback_key:
        os.environ["OLLAMA_API_KEY"] = fallback_key

st.set_page_config(page_title="Fuzzy Rule Explainer", layout="wide")
st.title("fNIRS Fuzzy Rule Explainer")


@st.cache_resource
def init():
    data = load_rules(RULES_PATH)
    client = get_client()
    return data, client


try:
    data, client = init()
except Exception as e:
    st.error(f"Could not start the app: {e}")
    st.stop()

# ---------------------------------------------------------
# Sidebar: model overview + CV performance
# ---------------------------------------------------------
with st.sidebar:
    st.subheader("Model overview")
    st.write(f"**{len(data['rules'])} rules** over target `{data['target_name']}`")

    cv = data.get("cv_performance_estimate")
    if cv:
        st.metric("Test MCC", f"{cv['test_mcc_mean']:.3f}",
                   f"± {cv['test_mcc_std']:.3f}")
        st.metric("Test accuracy", f"{cv['test_accuracy_mean']:.1%}",
                   f"± {cv['test_accuracy_std']:.1%}")
        st.caption(
            f"{cv['n_folds']}-fold subject-grouped cross-validation. "
            "MCC is shown first since accuracy alone sits close to the "
            "majority-class baseline here."
        )

    with st.expander("View all rules"):
        for r in data["rules"]:
            st.markdown(
                f"**Rule {r['rule_id']}**: IF {r['antecedent']} "
                f"THEN {data['target_name']} = {r['consequent']}  \n"
                f"DS={r['dominance_score']:.3f}, ACC={r['accuracy']:.3f}"
            )

    st.divider()
    if st.button("Clear chat"):
        st.session_state.history = []
        st.rerun()

# ---------------------------------------------------------
# Chat history
# ---------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

for role, content in st.session_state.history:
    with st.chat_message(role):
        st.markdown(content)

# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------
if question := st.chat_input("Ask about the model's rules..."):
    st.session_state.history.append(("user", question))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving relevant rules..."):
            try:
                answer, retrieved = ask(question, data, client)
            except Exception as e:
                answer = f"Something went wrong calling the model: {e}"
                retrieved = []
        st.markdown(answer)
        if retrieved:
            with st.expander(f"Rules used ({len(retrieved)})"):
                for r in retrieved:
                    st.caption(
                        f"Rule {r['rule_id']}: {r['antecedent']} → {r['consequent']} "
                        f"(ACC={r['accuracy']:.3f})"
                    )
    st.session_state.history.append(("assistant", answer))