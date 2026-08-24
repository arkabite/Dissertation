"""
Grouped k-fold cross-validation for the ExFuzzy fNIRS classifier.

Unlike seed_loop_eval.py (which repeats many independent random splits and
just MEASURES variance), this script partitions all 26 subjects into K folds
using sklearn's GroupKFold, so every subject appears in the test set exactly
once across the K folds. This uses 100% of your subjects for testing overall,
rather than risking an unlucky held-out group - the standard fix for the
small-sample split instability you saw between seed=1 and seed=2.

Run:
    python MainFold.py
Outputs (filenames suffixed with FEATURE_MODE, e.g. "_raw"):
    grouped_kfold_results_<mode>.csv       - per-fold metrics (ensembled)
    grouped_kfold_rules_per_fold_<mode>.json - every seed's rules, per fold
    extracted_rules_final_<mode>.json      - representative rule set + CV
                                              estimate + cross-seed stability
    exfuzzy_model_final_<mode>.pkl         - representative fitted classifier
Prints the mean +/- std across folds, which is the number you should report
as your headline performance metric instead of a single train/test split.

FEATURE_MODE (see Config section below) switches between the original raw
tmb_s1_ch*/tmb_s2_ch* channels and engineered HbD (= s1 - s2) / HbT
(= s1 + s2) features. Empirically, hbd_hbt underperformed raw on this
dataset (test_mcc 0.29 vs 0.34, and notably less stable across folds), so
"raw" is the current default.

N_SEEDS (see Config section) addresses a separate issue: the GA is
stochastic, and MainFold.py originally fixed a single random_state per
fold, so any one fold's reported result was really "how did this one seed
happen to do here." (This is what caused fold 5 of the hbd_hbt run to
collapse to 3 surviving rules and 0.459 test accuracy - that was GA luck,
not necessarily the feature set.) With N_SEEDS > 1, each fold/final-model
fit is repeated with independent seeds; fold metrics are computed from the
MAJORITY-VOTE ensemble of those seeds, and the reported final rule set is
whichever seed's train MCC sits closest to the cross-seed median (a guard
against reporting a rule set that just got lucky, without literally
averaging N different symbolic rule sets into an unreadable mess).

NOTE: N_SEEDS > 1 multiplies total GA fitting time by roughly N_SEEDS.
"""

import re
import json
import pickle

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
classifier_random_state = 6  # base seed; SEED_LIST below is derived from this

# Confirmed by reading ex_fuzzy/evolutionary_fit.py + eval_rules.py directly:
# BaseFuzzyRulesClassifier's fitness_func returns classification_eval(),
# which is matthews_corrcoef (not accuracy_score), and the alpha/beta terms
# that would add rule-size penalties both default to 0 and aren't set here.
# So the GA already optimizes MCC, not raw accuracy - no separate
# "imbalance-aware fitness" change is needed on top of that.

N_SEEDS = 3  # independent GA fits per fold (and for the final model),
             # combined via majority vote. Set to 1 to reproduce the
             # original single-seed behaviour. Runtime scales ~linearly
             # with this.
SEED_LIST = [classifier_random_state + i for i in range(N_SEEDS)]

target_name = "stimulus"
class_names = ["0 back", "2/3 back"]

# ---------------------------------------------------------
# Feature engineering: tmb_s1_ch* and tmb_s2_ch* are HbO and HbR
# (oxy-/deoxy-hemoglobin) derived from the SAME NIRS channel via the
# modified Beer-Lambert law - not independent measurements. 34/36 channel
# pairs are negatively correlated (mean r ~ -0.26) and the sign flips with
# task load matching the source paper's Fig. 3 (HbO up / HbR down with
# n-back load) - classic neurovascular coupling.
#
# FEATURE_MODE controls what the classifier is trained on:
#   "raw"      - original 72 features (tmb_s1_ch*, tmb_s2_ch*). CURRENT
#                BEST performer on this dataset (test_mcc_mean=0.34);
#                default here.
#   "hbd_only" - 36 features: HbD = s1 - s2 per channel only.
#   "hbd_hbt"  - 72 features: HbD = s1 - s2 AND HbT = s1 + s2 per channel.
#                Tried already: underperformed raw (test_mcc_mean=0.29,
#                ~2x the fold-to-fold accuracy variance).
FEATURE_MODE = "raw"  # one of: "raw", "hbd_only", "hbd_hbt"


def engineer_hbd_hbt(df: pd.DataFrame, channel_names: list) -> pd.DataFrame:
    """Build per-channel HbD (= s1 - s2) and HbT (= s1 + s2) features."""
    engineered = {}
    for ch in channel_names:
        s1 = df[f"tmb_s1_ch{ch}"]
        s2 = df[f"tmb_s2_ch{ch}"]
        engineered[f"HbD_ch{ch}"] = s1 - s2
        engineered[f"HbT_ch{ch}"] = s1 + s2
    return pd.DataFrame(engineered, index=df.index)


tmb_all = pd.read_csv("block_avg_tmb.csv")
raw_feature_names = list(tmb_all.columns[3:75])
channel_names = [
    c.replace("tmb_s1_ch", "") for c in raw_feature_names if c.startswith("tmb_s1_ch")
]

if FEATURE_MODE == "raw":
    X_all_df = tmb_all[raw_feature_names]
elif FEATURE_MODE == "hbd_only":
    _hbd_hbt_df = engineer_hbd_hbt(tmb_all, channel_names)
    X_all_df = _hbd_hbt_df[[c for c in _hbd_hbt_df.columns if c.startswith("HbD_")]]
elif FEATURE_MODE == "hbd_hbt":
    X_all_df = engineer_hbd_hbt(tmb_all, channel_names)
else:
    raise ValueError(f"Unknown FEATURE_MODE: {FEATURE_MODE!r}")

feature_names = list(X_all_df.columns)
print(f"FEATURE_MODE='{FEATURE_MODE}' -> {len(feature_names)} features "
      f"(e.g. {feature_names[:4]})")
print(f"N_SEEDS={N_SEEDS} -> seeds {SEED_LIST} "
      f"(runtime is roughly {N_SEEDS}x a single-seed run)")


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


def build_classifier(linguistic_variables) -> BaseFuzzyRulesClassifier:
    """Fresh, unfitted classifier with the shared hyperparameters."""
    return BaseFuzzyRulesClassifier(
        nRules=nRules,
        nAnts=nAnts,
        linguistic_variables=linguistic_variables,
        n_linguistic_variables=vl,
        fuzzy_type=fz_type_studied,
        class_names=class_names,
        verbose=False,
        tolerance=tolerance,
        runner=2,
        allow_unknown=True,
        ds_mode=0,
    )


def ensemble_predict(classifiers: list, X_df: pd.DataFrame) -> np.ndarray:
    """Majority vote across N independently-fit classifiers' predictions."""
    preds = np.stack([clf.predict(X_df) for clf in classifiers], axis=0)  # (N, n_samples)
    majority = np.empty(preds.shape[1], dtype=preds.dtype)
    for i in range(preds.shape[1]):
        vals, counts = np.unique(preds[:, i], return_counts=True)
        majority[i] = vals[np.argmax(counts)]
    return majority


def fit_multi_seed(X_train_df, y_train, seeds: list):
    """Fit one classifier per seed on the same training fold. Returns
    (classifiers, per_seed_train_mcc, per_seed_rules)."""
    partitions = ef_utils.construct_partitions(X_train_df, fz_type_studied, n_partitions=vl)
    classifiers, train_mccs, seed_rules = [], [], {}
    for seed in seeds:
        clf = build_classifier(partitions)
        clf.fit(X_train_df, y_train, n_gen=n_gen, pop_size=n_pop, random_state=seed)
        classifiers.append(clf)
        train_mccs.append(matthews_corrcoef(y_train, clf.predict(X_train_df)))
        seed_rules[seed] = extract_rules_from_classifier(clf)
    return classifiers, train_mccs, seed_rules


# ---------------------------------------------------------
# Grouped K-fold split - groups = subject, so no subject's trials
# ever appear in both train and test within a fold
# ---------------------------------------------------------
subjects_col = tmb_all["subject"].values
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

    seed_classifiers, seed_train_mccs, seed_rules = fit_multi_seed(
        X_train_df, y_train, SEED_LIST
    )

    # Ensemble (majority vote) metrics - this is the fold's headline result
    y_train_pred = ensemble_predict(seed_classifiers, X_train_df)
    y_test_pred = ensemble_predict(seed_classifiers, X_test_df)

    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    train_mcc = matthews_corrcoef(y_train, y_train_pred)
    test_mcc = matthews_corrcoef(y_test, y_test_pred)

    # Diagnostic: how much would a SINGLE seed have varied on this fold?
    # (this is what fold 5's hbd_hbt collapse was actually measuring)
    single_seed_test_mccs = [
        matthews_corrcoef(y_test, clf.predict(X_test_df)) for clf in seed_classifiers
    ]
    n_rules_per_seed = [len(seed_rules[s]) for s in SEED_LIST]

    all_fold_rules[fold_idx] = {
        "seeds": {str(s): seed_rules[s] for s in SEED_LIST},
        "n_rules_per_seed": n_rules_per_seed,
        "single_seed_test_mcc": single_seed_test_mccs,
    }

    fold_results.append({
        "fold": fold_idx,
        "n_test_subjects": len(fold_test_subjects),
        "test_subjects": ",".join(map(str, fold_test_subjects)),
        "n_rules_mean": float(np.mean(n_rules_per_seed)),
        "n_rules_min": int(np.min(n_rules_per_seed)),
        "n_rules_max": int(np.max(n_rules_per_seed)),
        "train_accuracy": train_acc,
        "test_accuracy": test_acc,
        "train_mcc": train_mcc,
        "test_mcc": test_mcc,
        "single_seed_test_mcc_mean": float(np.mean(single_seed_test_mccs)),
        "single_seed_test_mcc_std": float(np.std(single_seed_test_mccs)),
        "single_seed_test_mcc_min": float(np.min(single_seed_test_mccs)),
        "single_seed_test_mcc_max": float(np.max(single_seed_test_mccs)),
    })

    print(
        f"[fold {fold_idx}/{N_FOLDS}] test_subjects={fold_test_subjects}\n"
        f"    ENSEMBLE  train_acc={train_acc:.3f}  test_acc={test_acc:.3f}  "
        f"train_mcc={train_mcc:.3f}  test_mcc={test_mcc:.3f}\n"
        f"    single-seed test_mcc: mean={np.mean(single_seed_test_mccs):.3f}  "
        f"std={np.std(single_seed_test_mccs):.3f}  "
        f"range=[{np.min(single_seed_test_mccs):.3f}, {np.max(single_seed_test_mccs):.3f}]  "
        f"n_rules per seed={n_rules_per_seed}"
    )

# ---------------------------------------------------------
# Save + summarise - THIS mean+/-std is your headline number
# ---------------------------------------------------------
results_df = pd.DataFrame(fold_results)
results_csv_path = f"grouped_kfold_results_{FEATURE_MODE}.csv"
results_df.to_csv(results_csv_path, index=False)

print("\n================ SUMMARY ACROSS FOLDS (ensemble) ================")
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
print(
    f"\nFor comparison, average SINGLE-seed test MCC std within a fold was "
    f"{results_df['single_seed_test_mcc_std'].mean():.3f} - that's the variance "
    f"the ensemble is absorbing that a one-seed run (like the original script) "
    f"would have been exposed to."
)
print(f"\nSaved per-fold metrics to {results_csv_path}")

# ---------------------------------------------------------
# Save the actual rules found by every seed in every fold, so you can
# compare both across-fold AND across-seed variation (rule instability is
# itself a legitimate finding)
# ---------------------------------------------------------
rules_per_fold_path = f"grouped_kfold_rules_per_fold_{FEATURE_MODE}.json"
with open(rules_per_fold_path, "w") as f:
    json.dump(all_fold_rules, f, indent=2)
print(f"Saved every fold's per-seed rules to {rules_per_fold_path}")

# ---------------------------------------------------------
# IMPORTANT: the folds above exist only to give an honest performance
# estimate. None of those fold-models is "the" model. For the rule set you
# actually write up / feed to the Ollama explainer, refit once more on ALL
# subjects (no held-out test set, since your generalisation estimate
# already comes from the CV run above).
#
# With N_SEEDS > 1, this final step also fits N_SEEDS candidates and
# reports the one closest to the cross-seed median train MCC, plus a
# stability tally of which rules recurred across seeds.
# ---------------------------------------------------------
print("\n================ FITTING FINAL MODEL ON ALL DATA ================")
print("(this rule set - not any individual fold - is what you report/explain)")

final_classifiers, final_train_mccs, final_seed_rules = fit_multi_seed(
    X_all_df, y_all, SEED_LIST
)

# Representative seed = closest to the median train MCC across seeds -
# a guard against reporting whichever seed just got GA-lucky, without
# averaging N different symbolic rule sets together.
median_mcc = float(np.median(final_train_mccs))
rep_idx = int(np.argmin([abs(m - median_mcc) for m in final_train_mccs]))
representative_seed = SEED_LIST[rep_idx]
final_classifier = final_classifiers[rep_idx]
final_rules = final_seed_rules[representative_seed]

print(f"Per-seed train MCC: {dict(zip(SEED_LIST, [round(m, 3) for m in final_train_mccs]))}")
print(f"Representative seed: {representative_seed} "
      f"(train_mcc={final_train_mccs[rep_idx]:.3f}, median={median_mcc:.3f})")

# Cross-seed rule stability: how many of the N final seeds independently
# found the same (antecedent, consequent) pair? Rules that recur across
# multiple seeds are the trustworthy ones to lead with when reporting.
from collections import Counter
rule_key_counts = Counter()
for seed in SEED_LIST:
    seen_this_seed = {(r["antecedent"], r["consequent"]) for r in final_seed_rules[seed]}
    rule_key_counts.update(seen_this_seed)

rule_stability = [
    {"antecedent": ant, "consequent": cons, "n_seeds_found": count, "n_seeds_total": N_SEEDS}
    for (ant, cons), count in sorted(rule_key_counts.items(), key=lambda kv: -kv[1])
]

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
    "n_seeds": N_SEEDS,
    "representative_seed": representative_seed,
    "seed_train_mcc": {str(s): float(m) for s, m in zip(SEED_LIST, final_train_mccs)},
    "rule_stability_across_seeds": rule_stability,
    "rules": [
        {"rule_id": i + 1, "antecedent": r["antecedent"],
         "consequent": r["consequent"], "dominance_score": r["dominance_score"],
         "accuracy": r["accuracy"]}
        for i, r in enumerate(final_rules)
    ],
}

rules_final_path = f"extracted_rules_final_{FEATURE_MODE}.json"
with open(rules_final_path, "w") as f:
    json.dump(final_output, f, indent=2)

model_pkl_path = f"exfuzzy_model_final_{FEATURE_MODE}.pkl"
with open(model_pkl_path, "wb") as f:
    pickle.dump(final_classifier, f)

print(f"\nFinal (representative) model found {len(final_rules)} rules on the full dataset.")
n_stable = sum(1 for r in rule_stability if r["n_seeds_found"] > 1)
print(f"{n_stable}/{len(rule_stability)} unique rules recurred in more than 1 of the "
      f"{N_SEEDS} final seeds - see 'rule_stability_across_seeds' in the JSON.")
print(f"Saved to {rules_final_path} and {model_pkl_path}")
print(f"Use {rules_final_path} with ollamaRuleExplainer.py going forward -")
print("it carries the CV performance estimate alongside the rules, so your")
print("explainer/report can cite honest accuracy figures next to each rule.")