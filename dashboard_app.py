"""
dashboard_app.py

Static / traditional-visualisation condition for the RQ2 comparative
usability study. Participants browse fuzzy rules and model performance
through tables, filters, and charts only — no natural-language query,
no LLM call, no chat.

Built to read the SAME rule file as the conversational app
(Streamlit_app.py / rag_core.py), so any difference measured in the
comparative study is attributable to the *interface*, not to different
underlying content. Nothing in this file calls Ollama or needs an API key.

Run with:
    streamlit run dashboard_app.py
"""

import json

import pandas as pd
import streamlit as st

RULES_PATH = "extracted_rules_final.json"


@st.cache_data
def load_rules(path: str = RULES_PATH) -> dict:
    with open(path, "r") as f:
        return json.load(f)


st.set_page_config(page_title="Fuzzy Rule Dashboard", layout="wide")
st.title("fNIRS Fuzzy Rule Dashboard")
st.caption(
    "Traditional static-visualisation condition — browse rules and model "
    "performance directly. No chat, no natural-language query."
)

try:
    data = load_rules()
except Exception as e:
    st.error(f"Could not load rules from {RULES_PATH}: {e}")
    st.stop()

rules = data["rules"]
target_name = data["target_name"]
cv = data.get("cv_performance_estimate")

df = pd.DataFrame(rules).rename(columns={
    "rule_id": "Rule ID",
    "antecedent": "Antecedent",
    "consequent": "Consequent",
    "dominance_score": "Dominance Score",
    "accuracy": "Accuracy",
})

# ---------------------------------------------------------
# Top-level model performance — identical figures to the chat app's
# sidebar, kept identical here so the comparison is fair
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Rules in model", len(rules))
with col2:
    if cv:
        st.metric(
            "Test accuracy (5-fold CV)",
            f"{cv['test_accuracy_mean']:.1%}",
            f"± {cv['test_accuracy_std']:.1%}",
        )
with col3:
    if cv:
        st.metric(
            "Test MCC (5-fold CV)",
            f"{cv['test_mcc_mean']:.3f}",
            f"± {cv['test_mcc_std']:.3f}",
        )

if cv:
    st.caption(
        f"Cross-validated across {cv['n_folds']} subject-grouped folds. "
        "MCC is the more informative figure here, since raw accuracy sits "
        "close to the majority-class baseline for this dataset."
    )

st.divider()

# ---------------------------------------------------------
# Channel filter — the "traditional dashboard" equivalent of asking
# "what does channel AF8 predict?" in the chat app. Participants have to
# find and select it themselves rather than typing a question.
# ---------------------------------------------------------
all_channels = sorted({
    tok for text in df["Antecedent"]
    for tok in text.replace("IS", " ").replace("AND", " ").split()
    if tok.startswith("tmb_")
})

st.subheader("Filter rules by channel")
selected_channel = st.selectbox("Channel", ["(all rules)"] + all_channels)

filtered_df = df
if selected_channel != "(all rules)":
    filtered_df = df[df["Antecedent"].str.contains(selected_channel, regex=False)]

# ---------------------------------------------------------
# Rule table — sortable by clicking column headers
# ---------------------------------------------------------
st.subheader("Rule table")
st.dataframe(
    filtered_df[["Rule ID", "Antecedent", "Consequent", "Dominance Score", "Accuracy"]],
    use_container_width=True,
    hide_index=True,
)

# ---------------------------------------------------------
# Charts
# ---------------------------------------------------------
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Dominance score by rule")
    st.bar_chart(filtered_df.set_index("Rule ID")[["Dominance Score"]])

with chart_col2:
    st.subheader("Accuracy by rule")
    st.bar_chart(filtered_df.set_index("Rule ID")[["Accuracy"]])

st.subheader("Rules grouped by predicted class")
st.bar_chart(filtered_df["Consequent"].value_counts())

with st.expander("Full feature name list (all 72 channels)"):
    st.write(data["feature_names"])