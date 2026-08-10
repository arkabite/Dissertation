"""
Grouped k-fold cross-validation for the ExFuzzy fNIRS classifier.

Unlike seed_loop_eval.py (which repeats many independent random splits and
just MEASURES variance), this script partitions all 26 subjects into K folds
using sklearn's GroupKFold, so every subject appears in the test set exactly
once across the K folds. This uses 100% of your subjects for testing overall,
rather than risking an unlucky held-out group - the standard fix for the
small-sample split instability you saw between seed=1 and seed=2.

Run:
    python grouped_kfold_eval.py
Outputs:
    grouped_kfold_results.csv - per-fold metrics
Prints the mean +/- std across folds, which is the number you should report
as your headline performance metric instead of a single train/test split.
"""

import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.model_selection import GroupKFold

from ex_fuzzy.evolutionary_fit import BaseFuzzyRulesClassifier
import ex_fuzzy.utils as ef_utils
import ex_fuzzy.fuzzy_sets as fs
import ex_fuzzy.temporal as temporal

if not hasattr(temporal, "FUZZY_SETS"):
    temporal.FUZZY_SETS = fs.FUZZY_SETS

# ---------------------------------------------------------
# Config
# ---------------------------------------------------------
N_FOLDS = 5  # 26 subjects / 5 folds -> folds of 5 or 6 subjects each.
             # Set to 26 for full leave-one-subject-out (much slower: one
             # GA fit per subject instead of per fold).

fz_type_studied = fs.FUZZY_SETS.t1
vl = 3
n_gen = 100
n_pop = 75
nRules = 10
nAnts = 3
tolerance = 0.05
classifier_random_state = 6  # fixed, as in Mainfnirs.py - only the fold changes

target_name = "stimulus"
class_names = ["0 back", "2/3 back"]

tmb_all = pd.read_csv("block_avg_tmb.csv")
feature_names = list(tmb_all.columns[3:75])


def extract_rules_from_classifier(classifier) -> list:
    raw_text = str(classifier.rule_base)
    rules_list = []
    current_consequent = None

    for line in raw_text.splitlines():
        line = line.strip()

        m = re.match(r"Rules for consequent:\s*(.+)", line)
        if m:
            current_consequent = m.group(1)
            continue

        m = re.match(
            r"IF\s+(.*?)\s+WITH\s+DS\s+([\d.eE+-]+),\s*ACC\s+([\d.eE+-]+)",
            line,
        )
        if m:
            antecedent, ds, acc = m.groups()
            rules_list.append({
                "antecedent": antecedent,
                "consequent": current_consequent,
                "dominance_score": float(ds),
                "accuracy": float(acc),
            })

    return rules_list


# ---------------------------------------------------------
# Grouped K-fold split - groups = subject, so no subject's trials
# ever appear in both train and test within a fold
# ---------------------------------------------------------
subjects_col = tmb_all["subject"].values
X_all_df = tmb_all[feature_names]
y_all = tmb_all[target_name].values

gkf = GroupKFold(n_splits=N_FOLDS)

fold_results = []
all_fold_rules = {}

for fold_idx, (train_idx, test_idx) in enumerate(
    gkf.split(X_all_df, y_all, groups=subjects_col), start=1
):
    X_train_df = X_all_df.iloc[train_idx]
    y_train = y_all[train_idx]
    X_test_df = X_all_df.iloc[test_idx]
    y_test = y_all[test_idx]

    fold_test_subjects = sorted(set(subjects_col[test_idx]))

    precomputed_partitions = ef_utils.construct_partitions(
        X_train_df, fz_type_studied, n_partitions=vl
    )

    classifier = BaseFuzzyRulesClassifier(
        nRules=nRules,
        nAnts=nAnts,
        linguistic_variables=precomputed_partitions,
        n_linguistic_variables=vl,
        fuzzy_type=fz_type_studied,
        class_names=class_names,
        verbose=False,
        tolerance=tolerance,
        runner=2,
        allow_unknown=True,
        ds_mode=0,
    )

    classifier.fit(
        X_train_df, y_train,
        n_gen=n_gen, pop_size=n_pop,
        random_state=classifier_random_state,
    )

    y_train_pred = classifier.predict(X_train_df)
    y_test_pred = classifier.predict(X_test_df)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_mcc = matthews_corrcoef(y_train, y_train_pred)
    test_mcc = matthews_corrcoef(y_test, y_test_pred)

    fold_rules = extract_rules_from_classifier(classifier)
    n_rules = len(fold_rules)
    all_fold_rules[fold_idx] = fold_rules

    fold_results.append({
        "fold": fold_idx,
        "n_test_subjects": len(fold_test_subjects),
        "test_subjects": ",".join(map(str, fold_test_subjects)),
        "n_rules_found": n_rules,
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_mcc": train_mcc,
        "test_mcc": test_mcc,
    })

    print(
        f"[fold {fold_idx}/{N_FOLDS}] test_subjects={fold_test_subjects}  "
        f"train_acc={train_acc:.3f}  test_acc={test_acc:.3f}  "
        f"train_mcc={train_mcc:.3f}  test_mcc={test_mcc:.3f}  "
        f"n_rules={n_rules}"
    )

# ---------------------------------------------------------
# Save + summarise - THIS mean+/-std is your headline number
# ---------------------------------------------------------
results_df = pd.DataFrame(fold_results)
results_df.to_csv("grouped_kfold_results.csv", index=False)

print("\n================ SUMMARY ACROSS FOLDS ================")
summary = results_df[["train_accuracy", "test_accuracy", "train_mcc", "test_mcc"]].agg(
    ["mean", "std", "min", "max"]
)
print(summary)

print(
    f"\nHeadline test accuracy to report: "
    f"{results_df['test_accuracy'].mean():.3f} +/- {results_df['test_accuracy'].std():.3f}"
)
print(
    f"Headline test MCC to report: "
    f"{results_df['test_mcc'].mean():.3f} +/- {results_df['test_mcc'].std():.3f}"
)
print("\nSaved per-fold metrics to grouped_kfold_results.csv")

# ---------------------------------------------------------
# Save the actual rules found in every fold, so you can directly compare
# how much they vary (rule instability is itself a legitimate finding)
# ---------------------------------------------------------
import json

with open("grouped_kfold_rules_per_fold.json", "w") as f:
    json.dump(all_fold_rules, f, indent=2)
print("Saved every fold's actual rules to grouped_kfold_rules_per_fold.json")

# ---------------------------------------------------------
# IMPORTANT: the folds above exist only to give an honest performance
# estimate. None of those 5 fold-models is "the" model. For the rule set
# you actually write up / feed to the Ollama explainer, refit once more on
# ALL subjects (no held-out test set, since your generalisation estimate
# already comes from the CV run above).
# ---------------------------------------------------------
print("\n================ FITTING FINAL MODEL ON ALL DATA ================")
print("(this rule set - not any individual fold - is what you report/explain)")

final_partitions = ef_utils.construct_partitions(X_all_df, fz_type_studied, n_partitions=vl)

final_classifier = BaseFuzzyRulesClassifier(
    nRules=nRules,
    nAnts=nAnts,
    linguistic_variables=final_partitions,
    n_linguistic_variables=vl,
    fuzzy_type=fz_type_studied,
    class_names=class_names,
    verbose=False,
    tolerance=tolerance,
    runner=2,
    allow_unknown=True,
    ds_mode=0,
)
final_classifier.fit(
    X_all_df, y_all,
    n_gen=n_gen, pop_size=n_pop,
    random_state=classifier_random_state,
)

final_rules = extract_rules_from_classifier(final_classifier)
final_output = {
    "feature_names": feature_names,
    "target_name": target_name,
    "cv_performance_estimate": {
        "test_accuracy_mean": float(results_df["test_accuracy"].mean()),
        "test_accuracy_std": float(results_df["test_accuracy"].std()),
        "test_mcc_mean": float(results_df["test_mcc"].mean()),
        "test_mcc_std": float(results_df["test_mcc"].std()),
        "n_folds": N_FOLDS,
    },
    "rules": [
        {"rule_id": i + 1, "antecedent": r["antecedent"],
         "consequent": r["consequent"], "dominance_score": r["dominance_score"],
         "accuracy": r["accuracy"]}
        for i, r in enumerate(final_rules)
    ],
}

with open("extracted_rules_final.json", "w") as f:
    json.dump(final_output, f, indent=2)

import pickle
with open("exfuzzy_model_final.pkl", "wb") as f:
    pickle.dump(final_classifier, f)

print(f"Final model found {len(final_rules)} rules on the full dataset.")
print("Saved to extracted_rules_final.json and exfuzzy_model_final.pkl")
print("Use extracted_rules_final.json with ollamaRuleExplainer.py going forward -")
print("it carries the CV performance estimate alongside the rules, so your")
print("explainer/report can cite honest accuracy figures next to each rule.")