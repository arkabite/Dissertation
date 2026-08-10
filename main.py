import json
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ex_fuzzy.evolutionary_fit import BaseFuzzyRulesClassifier
from ex_fuzzy.eval_tools import eval_fuzzy_model
import ex_fuzzy.utils as ef_utils
import ex_fuzzy.fuzzy_sets as fs
import ex_fuzzy.temporal as temporal

# Compatibility fix for pickling: the library redefines FUZZY_SETS in temporal.py
# but does not expose it as an attribute on that module, which breaks pickle.
if not hasattr(temporal, "FUZZY_SETS"):
    temporal.FUZZY_SETS = fs.FUZZY_SETS

# ---------------------------------------------------------
# 1. Load & Preprocess Data
# ---------------------------------------------------------
df = pd.read_csv("Titanic-Dataset.csv")

df["Age"] = df["Age"].fillna(df["Age"].median())
df["Fare"] = df["Fare"].fillna(df["Fare"].median())
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["Embarked"] = df["Embarked"].map({"S": 0, "C": 1, "Q": 2})

features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
X_df = df[features].astype(float)
y = df["Survived"].values

categorical_mask = np.array([0, 1, 0, 0, 0, 0, 1])

linguistic_variables = ef_utils.construct_partitions(
    X_df,
    fz_type_studied=fs.FUZZY_SETS.t1,
    categorical_mask=categorical_mask,
    n_partitions=3
)

X = X_df.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# ---------------------------------------------------------
# 2. Fit Classifier (Increased nRules/nAnts for better coverage)
# ---------------------------------------------------------
classifier = BaseFuzzyRulesClassifier(
    nRules=20,          # Increased from 10 to give richer rule diversity
    nAnts=3,
    linguistic_variables=linguistic_variables,
    backend="pymoo"
)

classifier.fit(X_train, y_train)

# ---------------------------------------------------------
# 3. Evaluate Model
# ---------------------------------------------------------
try:
    eval_fuzzy_model(classifier, X_train, y_train, X_test, y_test,
                     plot_rules=True, plot_partitions=True)
except ModuleNotFoundError as e:
    print(f"Warning: {e}. Running evaluation without plotting.")
    eval_fuzzy_model(classifier, X_train, y_train, X_test, y_test,
                     plot_rules=False, plot_partitions=False)

# ---------------------------------------------------------
# 4. Extract and Save Structured Rules (.json)
# ---------------------------------------------------------
import re

rules_list = []

# First try to extract actual rule objects
if hasattr(classifier, "rule_base") and hasattr(classifier.rule_base, "rules"):

    for idx, rule in enumerate(classifier.rule_base.rules):

        rules_list.append({
            "rule_id": idx + 1,
            "antecedent": str(
                getattr(rule, "antecedents",
                    getattr(rule, "antecedent", ""))
            ),
            "consequent": str(getattr(rule, "consequent", "")),
            "confidence": float(
                getattr(rule, "confidence",
                    getattr(rule, "score", 0.0))
            ),
            "accuracy": float(getattr(rule, "accuracy", 0.0))
        })

else:
    # Generic fallback that parses the printed rule base
    raw_text = str(classifier.rule_base)

    current_consequent = None

    for line in raw_text.splitlines():

        line = line.strip()

        # Match "Rules for consequent: X"
        m = re.match(r"Rules for consequent:\s*(.+)", line)

        if m:
            current_consequent = m.group(1)
            continue

        # Match IF rule lines
        m = re.match(
            r"IF\s+(.*?)\s+WITH\s+DS\s+([\d.eE+-]+),\s*ACC\s+([\d.eE+-]+)",
            line,
        )

        if m:

            antecedent, ds, acc = m.groups()

            rules_list.append({
                "rule_id": len(rules_list) + 1,
                "antecedent": antecedent,
                "consequent": current_consequent,
                "dominance_score": float(ds),
                "accuracy": float(acc)
            })

    # Final fallback if nothing was extracted
    if not rules_list:
        rules_list.append({
            "raw_rules_text": raw_text
        })

# # Save metadata + rules
# output = {
#     "feature_names": features,
#     "rules": rules_list
# }

target_name = "Survived"

output = {
    "feature_names": list(X_df.columns),
    "target_name": target_name,
    "rules": rules_list
}

with open("extracted_rules.json", "w") as f:
    json.dump(output, f, indent=4)

print("Saved structured rules to extracted_rules.json")

# ---------------------------------------------------------
# 5. Save the Trained Model (.pkl)
# ---------------------------------------------------------
try:
    with open("exfuzzy_model.pkl", "wb") as f:
        pickle.dump(classifier, f)
    print("Saved classifier model to exfuzzy_model.pkl")
except Exception as exc:
    print(f"Pickle failed: {exc}")
    print("Falling back to saving a lightweight state dictionary.")
    fallback_payload = {
        "classifier_type": type(classifier).__name__,
        "feature_names": features,
        "rules": rules_list,
        "train_score": getattr(classifier, "train_score", None),
    }
    with open("exfuzzy_model.pkl", "wb") as f:
        pickle.dump(fallback_payload, f)
    print("Saved fallback model payload to exfuzzy_model.pkl")