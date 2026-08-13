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

# ---------------------------------------------------------
# Glossary — kept word-for-word identical to the copy in Streamlit_app.py
# and Section 8 of the study protocol. If you edit this, edit all three,
# or the two study conditions stop being a fair comparison.
# ---------------------------------------------------------
GLOSSARY_MD = """
- **Channel** — A sensor location on the scalp that recorded brain-activity data (e.g. AF7, PO2). The model looks at readings from many channels, not just one.
- **0 back / 2-3 back** — The two outcomes the model predicts. They refer to how demanding the memory task was: "0 back" = low demand, "2/3 back" = higher demand.
- **Rule** — A statement of the form "IF [certain channels show certain patterns] THEN [predict 0 back / 2/3 back]." The model is made up of several such rules, not one single formula.
- **Antecedent / Consequent** — The "IF" part of a rule is its antecedent (the condition); the "THEN" part is its consequent (the prediction).
- **Dominance score** — A number showing how much a rule contributes to the model's overall decisions, relative to the other rules. Higher = the rule covers more of the data.
- **Accuracy (per rule)** — How often that specific rule was correct, when it applied, in the data it was built from.
- **Accuracy / MCC (model-level)** — A separate estimate of how well the whole model performs on new data, tested by holding out different groups of people in turn. MCC (Matthews Correlation Coefficient) is a stricter measure than accuracy and should be treated as at least as important.

If you don't know an answer, or the interface doesn't give you one, say so — that is a valid, correct response.
"""



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

with st.expander("📖 Glossary — click to expand definitions"):
    st.markdown(GLOSSARY_MD)

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