# import json
# import pickle
# import random
# import re

# import numpy as np
# import pandas as pd

# from ex_fuzzy.evolutionary_fit import BaseFuzzyRulesClassifier
# from ex_fuzzy.eval_tools import eval_fuzzy_model
# import ex_fuzzy.utils as ef_utils
# import ex_fuzzy.fuzzy_sets as fs
# import ex_fuzzy.temporal as temporal

# # Compatibility fix for pickling: the library redefines FUZZY_SETS in temporal.py
# # but does not expose it as an attribute on that module, which breaks pickle.
# if not hasattr(temporal, "FUZZY_SETS"):
#     temporal.FUZZY_SETS = fs.FUZZY_SETS

# # ---------------------------------------------------------
# # 1. Load Data (block-averaged fNIRS features)
# # ---------------------------------------------------------
# tmb_all = pd.read_csv("block_avg_tmb.csv")

# target_name = "stimulus"
# class_names = ["0 back", "2/3 back"]

# # Feature columns are the 72 fNIRS channel columns (cols 3:75),
# # same slice used in 003_Fuzzy_Model_Development_and_Validation.ipynb
# feature_names = list(tmb_all.columns[3:75])

# # ---------------------------------------------------------
# # 2. Train/Test Split by Subject (prevents leakage across trials
# #    from the same subject) - matches notebook 003, seed=1
# # ---------------------------------------------------------
# random.seed(2)
# subjects = list(range(1, 27))
# test_subjects = random.sample(subjects, 6)
# train_subjects = [s for s in subjects if s not in test_subjects]

# df_train = tmb_all[tmb_all["subject"].isin(train_subjects)]
# X_train_df = df_train[feature_names]
# y_train = df_train[target_name].values

# df_test = tmb_all[tmb_all["subject"].isin(test_subjects)]
# X_test_df = df_test[feature_names]
# y_test = df_test[target_name].values

# # ---------------------------------------------------------
# # 3. Fuzzy Partitions + Classifier
# #    Hyperparameters are the best-performing combination found
# #    via grid search in notebook 003 (n_gen=100, pop=75, nRules=10,
# #    nAnts=3, vl=3, tolerance=0.05, random_state=6, type-1 fuzzy sets)
# # ---------------------------------------------------------
# fz_type_studied = fs.FUZZY_SETS.t1
# vl = 3
# n_gen = 100
# n_pop = 75
# nRules = 10
# nAnts = 3
# tolerance = 0.05
# random_state = 6

# precomputed_partitions = ef_utils.construct_partitions(
#     X_train_df, fz_type_studied, n_partitions=vl
# )

# classifier = BaseFuzzyRulesClassifier(
#     nRules=nRules,
#     nAnts=nAnts,
#     linguistic_variables=precomputed_partitions,
#     n_linguistic_variables=vl,
#     fuzzy_type=fz_type_studied,
#     class_names=class_names,
#     verbose=False,
#     tolerance=tolerance,
#     runner=2,
#     allow_unknown=True,
#     ds_mode=0,
# )

# classifier.fit(X_train_df, y_train, n_gen=n_gen, pop_size=n_pop, random_state=random_state)

# # ---------------------------------------------------------
# # 4. Evaluate
# # ---------------------------------------------------------
# try:
#     eval_fuzzy_model(
#         classifier, X_train_df, y_train, X_test_df, y_test,
#         plot_rules=False, print_rules=True, plot_partitions=False,
#     )
# except ModuleNotFoundError as e:
#     print(f"Warning: {e}. Running evaluation without plotting.")

# # ---------------------------------------------------------
# # 5. Extract and Save Structured Rules (.json)
# #    Same extraction logic as main.py (Titanic version) so the
# #    output shape matches what ollamaRuleExplainer.py expects.
# # ---------------------------------------------------------
# rules_list = []

# if hasattr(classifier, "rule_base") and hasattr(classifier.rule_base, "rules"):
#     for idx, rule in enumerate(classifier.rule_base.rules):
#         rules_list.append({
#             "rule_id": idx + 1,
#             "antecedent": str(
#                 getattr(rule, "antecedents", getattr(rule, "antecedent", ""))
#             ),
#             "consequent": str(getattr(rule, "consequent", "")),
#             "confidence": float(
#                 getattr(rule, "confidence", getattr(rule, "score", 0.0))
#             ),
#             "accuracy": float(getattr(rule, "accuracy", 0.0)),
#         })
# else:
#     raw_text = str(classifier.rule_base)
#     current_consequent = None

#     for line in raw_text.splitlines():
#         line = line.strip()

#         m = re.match(r"Rules for consequent:\s*(.+)", line)
#         if m:
#             current_consequent = m.group(1)
#             continue

#         m = re.match(
#             r"IF\s+(.*?)\s+WITH\s+DS\s+([\d.eE+-]+),\s*ACC\s+([\d.eE+-]+)",
#             line,
#         )
#         if m:
#             antecedent, ds, acc = m.groups()
#             rules_list.append({
#                 "rule_id": len(rules_list) + 1,
#                 "antecedent": antecedent,
#                 "consequent": current_consequent,
#                 "dominance_score": float(ds),
#                 "accuracy": float(acc),
#             })

#     if not rules_list:
#         rules_list.append({"raw_rules_text": raw_text})

# output = {
#     "feature_names": feature_names,
#     "target_name": target_name,
#     "rules": rules_list,
# }

# with open("extracted_rules.json", "w") as f:
#     json.dump(output, f, indent=4)

# print("Saved structured rules to extracted_rules.json")

# # ---------------------------------------------------------
# # 6. Save the Trained Model (.pkl)
# # ---------------------------------------------------------
# try:
#     with open("exfuzzy_model.pkl", "wb") as f:
#         pickle.dump(classifier, f)
#     print("Saved classifier model to exfuzzy_model.pkl")
# except Exception as exc:
#     print(f"Pickle failed: {exc}")
#     print("Falling back to saving a lightweight state dictionary.")
#     fallback_payload = {
#         "classifier_type": type(classifier).__name__,
#         "feature_names": feature_names,
#         "rules": rules_list,
#         "train_score": getattr(classifier, "train_score", None),
#     }
#     with open("exfuzzy_model.pkl", "wb") as f:
#         pickle.dump(fallback_payload, f)
#     print("Saved fallback model payload to exfuzzy_model.pkl")

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

    n_rules = len(extract_rules_from_classifier(classifier))

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