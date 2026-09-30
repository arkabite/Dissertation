# Dissertation System Structure

This document describes the current implementation from model fitting to the
Streamlit interfaces. The main execution path is:

```text
block_avg_tmb.csv
        |
        v
MainFold.py
  |             \
  |              +--> grouped_kfold_results_raw.csv
  |              +--> grouped_kfold_rules_per_fold_raw.json
  |              +--> extracted_rules_final_raw.json
  |              +--> exfuzzy_model_final_raw.pkl
  |
  +--> brain_mapping.json --> build_channel_atlas.py --> channel_atlas.json
                                                        |
                                                        v
                        Streamlit_app.py --> rag_core.py --> Ollama response
                              |                 |
                              |                 +--> viz_tools.py
                              |                 +--> deterministic fallback
                              |
                              +--> viz_components.py --> charts and 3D views

                        dashboard_app.py
                              |
                              +--> extracted_rules_final.json

                        Baseline comparison notebook
                           |
                           +--> baseline_kfold_results_raw.csv
```

`backend.py` and `api_client.py` are legacy FastAPI code paths. They are not
used by the current Streamlit application.

## 1. Model fitting: MainFold.py

### Purpose

`MainFold.py` trains and evaluates the ExFuzzy fuzzy rule classifier. It has
two related responsibilities:

1. Estimate how well the modelling procedure generalises to unseen subjects.
2. Fit a final representative rule set on all available subjects for
   explanation and reporting.

These responsibilities are kept separate. The cross-validation models provide
the performance estimate, while the final all-data model provides the rules
that are shown to users and supplied to the RAG explainer.

### Input data

The script reads [block_avg_tmb.csv](block_avg_tmb.csv). The file contains
block-averaged fNIRS features, subject identifiers, and the target column
`stimulus`.

The current configuration uses `FEATURE_MODE = "raw"`. The raw feature matrix
contains 72 features:

- 36 physical fNIRS channels measured as oxygenated haemoglobin features,
  named `tmb_s1_ch...`.
- The same 36 channels measured as deoxygenated haemoglobin features, named
  `tmb_s2_ch...`.

In the project convention, `s1` is HbO and `s2` is HbR. The two signals belong
to the same underlying physical channel and should not be interpreted as 72
independent electrode locations.

The script also supports two engineered alternatives:

- `hbd_only`: HbD features calculated as `s1 - s2`.
- `hbd_hbt`: HbD and HbT features, where HbT is `s1 + s2`.

The raw representation is the current default because it performed better on
this dataset than the tested engineered alternatives.

### Fuzzy classifier configuration

The classifier is created by `build_classifier()` using the ExFuzzy
`BaseFuzzyRulesClassifier`. The important settings are:

- Type-1 fuzzy sets.
- Three linguistic partitions per feature.
- Up to 10 rules.
- Up to 3 antecedents per rule.
- 100 evolutionary generations.
- Population size 75.
- Classification tolerance 0.05.
- Target classes `0 back` and `2/3 back`.

The evolutionary algorithm is stochastic, so one fit can produce a different
rule set from another fit even on the same data.

### Subject-grouped five-fold cross-validation

`MainFold.py` uses `GroupKFold(n_splits=5)` with the subject identifier as the
group. This means that all trials from a subject stay together in either the
training or test partition for a fold. A subject therefore cannot contribute
training examples and test examples in the same fold.

For every fold, the script:

1. Splits the data by subject.
2. Fits three independent classifiers using seeds 6, 7, and 8.
3. Predicts the training and test partitions with majority voting across the
   three classifiers.
4. Calculates accuracy and Matthews Correlation Coefficient (MCC).
5. Stores the rules found by every seed and diagnostic single-seed metrics.

The ensemble majority vote is the headline fold prediction. The individual
seed results are retained to show how much the stochastic fitting process
varies within a fold.

### Why MCC is used

The ExFuzzy fitness function uses the project's classification evaluation,
which is MCC rather than raw accuracy. MCC is useful for binary classification
when class balance is imperfect because it accounts for all four confusion
matrix components. Accuracy is retained as an additional, easier-to-read
metric, but it is not sufficient on its own to characterise model quality.

### Final representative model

After the five folds have produced a generalisation estimate, the script fits
three further classifiers on all subjects. These final fits are not used to
claim an unbiased test score; they exist to produce the rule set used for
explanation.

The representative final seed is selected by comparing each seed's training
MCC with the median training MCC across the three seeds. The seed closest to
the median is selected. This avoids simply reporting whichever stochastic fit
happened to achieve the most favourable training score, without trying to
average several symbolic rule sets into an unreadable composite.

### Rule extraction and stability

The fitted ExFuzzy rule base is converted to structured dictionaries by
parsing the classifier's printed rule representation. Each rule contains:

- `rule_id`
- `antecedent`: the IF condition, including channel, signal, and fuzzy level
- `consequent`: the predicted class
- `dominance_score`: the rule's relative contribution/coverage measure
- `accuracy`: the rule's accuracy when it applies in the fitted data

The script also compares the exact `(antecedent, consequent)` pair across the
three final seeds. The resulting `n_seeds_found` and `n_seeds_total` values
indicate recurrence of that exact symbolic rule, not proof that an equivalent
biological pattern would always be rediscovered with different wording.

## 2. Files produced by MainFold.py

Filenames include the selected feature mode. With the current raw mode, the
outputs are:

### `grouped_kfold_results_raw.csv`

One row per cross-validation fold. It contains:

- Fold number and number/list of test subjects.
- Mean, minimum, and maximum rule counts across seeds.
- Ensemble train/test accuracy.
- Ensemble train/test MCC.
- Mean, standard deviation, minimum, and maximum single-seed test MCC.

The mean and standard deviation of the ensemble test metrics are the main
cross-validation performance estimates to report.

### `grouped_kfold_rules_per_fold_raw.json`

The rules found by every seed in every validation fold, together with the
number of rules per seed and the single-seed test MCC diagnostics. This file
makes fold-level and seed-level rule variation inspectable.

### `extracted_rules_final_raw.json`

The primary structured input for the current RAG application. It contains:

- `feature_names`: the 72 raw feature names.
- `target_name`: `stimulus`.
- `cv_performance_estimate`: mean and standard deviation of five-fold test
  accuracy and MCC, plus the fold count.
- `n_seeds`: the number of final independent fits.
- `representative_seed` and `seed_train_mcc`.
- `rule_stability_across_seeds`: exact rule recurrence information.
- `rules`: the representative all-data rule set.

The CV metrics and final rules have different meanings. CV metrics estimate
performance on unseen subjects; the final rules are fitted using all subjects
and are therefore not an independent test result.

### `exfuzzy_model_final_raw.pkl`

A pickle containing the representative fitted ExFuzzy classifier. It preserves
the trained model object for Python use, but the current Streamlit explainer
reads the structured JSON rules rather than loading this pickle to answer
questions.

### Other generated variants

If `FEATURE_MODE` is changed, the same four output types are created with the
corresponding suffix, such as `_hbd_only` or `_hbd_hbt`.

## 3. Brain mapping and channel atlas

### `brain_mapping.json`

`brain_mapping.json` is a tiered glossary for explaining opaque channel names.
It is used by `rag_core.py` when constructing the prompt.

#### Signal information

The glossary records:

- `s1` as oxygenated haemoglobin, HbO.
- `s2` as deoxygenated haemoglobin, HbR.
- The project's observed task-load direction: HbO generally increases and HbR
generally decreases with higher n-back load.

These are project-level signal definitions and feature-engineering facts. They
do not mean that every individual rule proves a physiological mechanism.

#### Hemisphere information

The standard naming convention is used:

- Odd numeric suffixes indicate the left hemisphere.
- Even numeric suffixes indicate the right hemisphere.
- `z` indicates the midline.

#### Region information

The glossary maps region prefixes such as `AF`, `C`, `P`, and `PO` to readable
scalp and functional-region labels. It also records general literature
associations, but those associations are explicitly marked as background and
must not be presented as findings demonstrated by a particular rule.

`PPO` is treated as ambiguous because the suffix changes whether the channel is
closer to the parietal or occipital grouping. The system is instructed not to
force one region when the source information does not support it.

### `build_channel_atlas.py`

`build_channel_atlas.py` creates the static [channel_atlas.json](channel_atlas.json)
file. It contains the 36 physical electrode sites, not 72 separate positions,
because HbO and HbR at the same electrode share a physical location.

The script:

1. Creates an MNE EEG information object for the 36 channel names.
2. Loads MNE's `standard_1005` montage.
3. Uses MNE's topomap coordinate calculation to obtain 2D positions.
4. Normalises the coordinates for plotting.
5. Splits each channel into a region prefix and hemisphere suffix.
6. Joins the channel with the region and hemisphere labels in
   `brain_mapping.json`.
7. Writes position, region code, functional region, evidence basis, and
   hemisphere to `channel_atlas.json`.

This makes the atlas the shared deterministic source for the visualisations.
The LLM can request a channel by name, but it cannot invent its plotted
position, region, or activation level.

Run it after changing the channel list or the region entries in
`brain_mapping.json`:

```text
python build_channel_atlas.py
```

## 4. RAG implementation: rag_core.py

`rag_core.py` contains the non-UI explanation logic. It does not import
Streamlit, start a server, or construct a client at import time.

### Files used by the RAG layer

- `extracted_rules_final_raw.json`: structured rules, feature names, CV
  statistics, seed count, and stability data.
- `domain_background.md`: curated project background supplied as general
  context, separate from the model's own rule findings.
- `brain_mapping.json`: signal, hemisphere, region, and evidence-tier glossary.
- `channel_atlas.json`: loaded indirectly through `viz_tools.py` for plotted
  positions and visual metadata.
- Ollama Cloud: the external LLM service used for natural-language responses.

### Loading and client construction

`load_rules()` reads the structured JSON and rejects an older raw-rule format.
`load_domain_background()` and `load_brain_mapping()` load optional grounding
files and return an empty value if the optional file is absent.

`get_client()` creates an Ollama client pointed at `https://ollama.com`. The
current Streamlit app first obtains `OLLAMA_API_KEY` from Streamlit secrets
and passes it to this function. `rag_core.py` also supports an environment
variable and local secrets-file fallbacks for non-Streamlit use.

### Retrieval

The default `TOP_K` is 5. `retrieve_rules()` uses a deliberately simple,
inspectable scoring method:

1. Tokenise the question and rule text with an alphanumeric regular
   expression. Numeric channel suffixes are preserved.
2. Remove common stopwords so words such as `is` do not match every fuzzy
   rule.
3. Count shared query/rule tokens.
4. Give extra weight to exact feature-name matches.
5. Add substring matches so a natural-language `AF8` query can match a stored
   token such as `chAF8`.
6. Sort by score and return the top rules.
7. If there is no meaningful overlap, rank by rule accuracy as a fallback.

This is keyword retrieval rather than a vector database. Its advantages here
are transparency, reproducibility, low operational complexity, and easy
verification against exact channel names. Its limitation is that semantic
paraphrases that share little vocabulary may retrieve less well.

### Scope guard

`is_in_scope()` runs before an LLM call. It checks the question against
feature names, rule vocabulary, and a controlled model-domain vocabulary. A
question about unrelated topics is refused deterministically, which prevents
an unnecessary model call and reduces the opportunity for an ungrounded
answer.

### Prompt construction

`build_prompt()` constructs a system message containing:

- The retrieved rules and the number retrieved versus the total rule count.
- Exact target labels.
- Per-rule dominance and accuracy values.
- Model-level cross-validation accuracy and MCC, explicitly separated from
  per-rule accuracy.
- Cross-seed recurrence information and a warning not to overclaim stability.
- Domain background, marked as general context rather than this model's own
  finding.
- A channel glossary restricted to region codes used by the retrieved rules.

Before a rule enters the prompt, `pre_translate_rule()` deterministically
replaces opaque channel tokens such as `tmb_s1_chAF5h` with bracketed labels
such as a hemisphere, scalp-region label, and signal. The model is instructed
to copy these labels rather than reconstructing hemisphere or signal details
from memory.

The prompt also separates evidence tiers:

- Tier 1: sourced signal, location, hemisphere, and functional-region labels.
- Tier 2: general literature associations, which must be explicitly hedged.
- Tier 3: ambiguous mappings such as PPO, which must not be resolved by
  guessing.

### Fallback answer

`build_fallback_answer()` creates a deterministic response from the retrieved
rules when the LLM is unavailable or raises an exception. It lists the raw
rule conditions, target class, accuracy, and dominance value. This preserves a
useful grounded response without pretending that a natural-language model
answer was produced.

## 5. Streamlit_app.py

`Streamlit_app.py` is the current conversational application. It runs all
orchestration in the Streamlit process and does not call `backend.py`.

Start it with:

```text
streamlit run Streamlit_app.py
```

### Initialisation

The app:

1. Sets the page configuration and displays the title and subtitle.
2. Initialises `st.session_state.history` so the conversation survives
   Streamlit reruns.
3. Uses cached `load_app_state()` to load rules, background, brain mapping,
   channel atlas, and the Ollama client once per Streamlit session/cache key.
4. Shows a warning if the API key is missing and keeps the deterministic path
   available.
5. Builds all-rule overview visualisations for the sidebar.

### Sidebar

The sidebar displays:

- Number of rules and target name.
- Five-fold subject-grouped test accuracy and MCC with standard deviations.
- The full rule set with antecedent, consequent, dominance score, and
  per-rule accuracy.
- An all-rule 2D scalp map or 3D headspace view.
- An all-rule channel co-occurrence diagram.
- A `Clear chat` button.

The CV metrics in the sidebar are model-level metrics. They should not be
read as the accuracy of any single symbolic rule.

### Question handling

When a user submits a question through `st.chat_input()`:

1. The user message is added to session history.
2. `is_in_scope()` checks whether it is plausibly about this model.
3. An out-of-scope question receives a fixed refusal and no LLM call.
4. If the Ollama client is unavailable, the app retrieves rules and builds the
   deterministic fallback answer.
5. Otherwise, the app calls `ask_with_visualization()`.
6. Any exception from the LLM or visualization request is caught. The app then
   retrieves rules and returns the deterministic fallback instead of exposing a
   traceback.
7. The assistant answer, retrieved rules, and optional visualization payload
   are stored in session history.
8. The full history is rendered on the next and subsequent Streamlit reruns.

Each assistant message can show a `Rules used` expander containing the rule
ID, antecedent, consequent, and accuracy. This lets a user inspect which rule
records supported the answer.

## 6. Visualization and response flow

Visualisations are optional. The LLM decides whether a topomap or chord diagram
would help, but the LLM does not provide the data used to draw it.

### Two-call visualization flow

For a normal question, `ask_with_visualization()` can use one LLM call. When
visualisation tools are enabled, the flow is:

1. Retrieve the top five relevant rules.
2. Build a grounded prompt with the retrieved rules and glossary.
3. Send the prompt to Ollama with the `show_topomap` and
   `show_chord_diagram` tool definitions.
4. Read the first tool call, if any, only as an intention and a rough channel
   request.
5. Resolve requested channels against channel names actually present in the
   retrieved rules.
6. If the request cannot be resolved, use all channels in the retrieved rules
   rather than trusting an invented channel.
7. Build the visualization payload deterministically from the rules and
   `channel_atlas.json`.
8. Send a second grounded message to Ollama containing a compact summary of
   exactly what was rendered.
9. Return the narrated answer plus the structured visualization payload.

If no tool call is made, the function returns a normal single-call answer. The
current implementation handles only the first tool call, so one visualization
is produced per answer at most.

### Topomap

`build_topomap_payload()` parses each retrieved rule antecedent into structured
facts containing:

- Full feature name.
- Bare physical channel name.
- HbO or HbR signal.
- Low, Medium, or High fuzzy activation level.
- Rule ID and predicted class.
- Per-rule accuracy.
- Atlas x/y position and region metadata.

`build_topomap_spec()` turns these facts into a Vega-Lite specification for the
2D scalp map. HbO and HbR receive a small horizontal offset when they share a
physical electrode so both points remain visible. The UI uses shape to
separate signal types and colour to show activation level.

The 3D alternative is rendered by `build_brain3d_html()` in an embedded HTML
component. It uses the same deterministic points, with spheres and cubes to
separate HbO and HbR.

### Chord diagram

`build_chord_payload()` creates nodes for channels and edges for channel pairs
that occur together in the same rule antecedent. An edge means literal
co-occurrence in an AND condition; it is not a claim of measured physiological
connectivity.

The payload includes the rule ID, consequent, dominance score, accuracy, and
channel metadata. `build_chord_html()` renders it in the Streamlit page.

### Response object

The successful RAG response contains an answer, the retrieved rules, and an
optional visualization object. Conceptually it has this structure:

```python
{
    "answer": "...",
    "retrieved": [
        {
            "rule_id": 1,
            "antecedent": "...",
            "consequent": "0 back",
            "dominance_score": 0.0,
            "accuracy": 0.0,
        }
    ],
    "visualization": {
        "type": "topomap" or "chord",
        "topomap": [...],
        "chord": {...},
    },
}
```

The actual shape has `None` for the unused visualization branch. On a
fallback response, `visualization` is `None`, while `retrieved` still shows
the rules used to build the raw answer.

## 7. Static comparison condition: dashboard_app.py

`dashboard_app.py` is the non-conversational condition for the RQ2 usability
comparison. It does not import `rag_core.py`, call Ollama, or require an API
key.

Start it with:

```text
streamlit run dashboard_app.py
```

The dashboard currently reads `extracted_rules_final.json`. It displays:

- Total rule count.
- Model-level five-fold CV accuracy and MCC.
- A plain-language glossary.
- A channel selector for filtering rules.
- A sortable rule table.
- Dominance-score and per-rule-accuracy charts.
- Rules grouped by consequent.
- The full feature-name list.

The dashboard is intentionally less adaptive than the chat app. Participants
must select a channel and inspect tables/charts rather than ask a natural-
language question. The two conditions should be compared as interfaces, while
keeping the underlying rule meaning and model-level metrics aligned.

## 8. Why the design is methodologically sound

The implementation has several safeguards that support a defensible
explainability workflow:

### No subject leakage in the main performance estimate

Grouped folds prevent trials from the same subject appearing in both training
and test data in one fold. Every subject is used as test data across the full
five-fold process, while the fold-level estimate remains separate from the
all-data rule fit.

### MCC-aware optimisation and reporting

The evolutionary classifier's fitness is MCC, which is more informative than
accuracy alone when class proportions are not perfectly balanced. Both metrics
are reported, with their scope clearly labelled.

### Seed variation is measured rather than hidden

Independent seeds are used for each fold and for the final fit. Majority voting
reduces the influence of a single lucky seed for predictive evaluation, while
saving every seed's rules makes instability visible. Exact rule recurrence is
reported as recurrence, not promoted to biological or causal certainty.

### Rule-grounded language generation

The LLM receives only the retrieved rules and explicitly supplied statistics
and background. It is instructed to refuse unsupported questions, disclose
partial retrieval, preserve exact class labels, distinguish rule-level from
model-level metrics, and hedge general literature claims.

### Deterministic channel transcription

Channel-to-signal, hemisphere, and region labels are generated by code before
the prompt is sent. The atlas and visualization payload are also generated by
code. This reduces the risk that an LLM silently changes a channel side,
signal type, activation value, or plotted position while paraphrasing.

### Graceful degradation

If the cloud model cannot be reached, the application still returns the
retrieved raw rules. This is more transparent than fabricating an explanation
or presenting an opaque error to a study participant.

## 9. Important limitations

The safeguards improve reliability but do not turn the system into a causal
neuroscience instrument.

- Per-rule accuracy and dominance are properties of the fitted rule set; they
  are not independent test scores for individual rules.
- Cross-validation estimates performance for the modelling procedure, not a
  guarantee of performance on every future population.
- A rule is a predictive association in this dataset, not proof that a brain
  region caused the predicted task state.
- The channel atlas uses standard montage geometry and dataset-supported region
  groupings. It is not a clinical localisation map.
- General literature associations in `brain_mapping.json` are background
  context, not findings established by this dataset.
- Keyword retrieval is transparent but can miss a semantically equivalent
  question that shares little vocabulary with the stored rules.
- The LLM can still produce a flawed paraphrase, so retrieved rules and exact
  labels should remain visible for audit.
- The dashboard and chat app now read the same structured rule file,
   `extracted_rules_final_raw.json`, so their rule content is aligned. Keep this
   path consistent if a new feature-mode output is generated.

## 10. Reproducibility checklist

1. Install the packages in `requirements.txt`.
2. Confirm `block_avg_tmb.csv` and the desired feature mode in `MainFold.py`.
3. Run `python MainFold.py` to regenerate model outputs.
4. Run `python build_channel_atlas.py` if the channel list or brain mapping
   changed.
5. Place `OLLAMA_API_KEY` in `.streamlit/secrets.toml` for the chat app.
6. Run `streamlit run Streamlit_app.py`.
7. Run `streamlit run dashboard_app.py` for the static comparison condition.
8. Re-run the ten grounding probes after changes to retrieval, prompt text,
   glossary content, model selection, or response formatting.
9. Record the generated file suffix and representative seed alongside any
   reported results.

## 11. Baseline model comparison

The baseline comparison notebook provides non-fuzzy reference models for the
same classification task. It is a separate evaluation stage from the RAG and
Streamlit system, but it uses the same raw feature representation and the
same subject-grouped outer folds as `MainFold.py`.

### Models

The comparison evaluates:

- Support Vector Machine (`SVC`).
- Random Forest (`RandomForestClassifier`).
- Linear Discriminant Analysis (`LinearDiscriminantAnalysis`).
- Multilayer Perceptron (`MLPClassifier`).

All models use the same 72 raw features from columns 3 through 74 of
`block_avg_tmb.csv`, the same `stimulus` target, and the same 26 subjects.
The outer split is `GroupKFold(n_splits=5)` with `subject` as the group. As
`GroupKFold` is deterministic when no shuffle or random state is supplied,
the held-out subjects line up with the folds used by `MainFold.py`, provided
the input file and feature slice are identical.

### Nested model selection

Within each outer training fold, the notebook uses a five-fold shuffled
`StratifiedKFold` with seed 42 for `GridSearchCV`. The grid search selects
hyperparameters using MCC. The selected estimator is then evaluated on the
outer training and held-out subject sets using accuracy and MCC.

This gives each baseline a model-selection step without using the outer test
subjects to choose hyperparameters. The resulting mean and standard deviation
across the five outer folds can therefore be compared with the corresponding
ensemble metrics in `grouped_kfold_results_raw.csv`.

### Output

The notebook writes `baseline_kfold_results_raw.csv`. Each row contains:

- Outer fold and held-out subjects.
- Baseline model name.
- Selected hyperparameters.
- Train/test accuracy.
- Train/test MCC.

The reported baseline summary is:

| Model | Test accuracy mean +/- std | Test MCC mean +/- std |
|---|---:|---:|
| LDA | 0.736 +/- 0.032 | 0.418 +/- 0.067 |
| MLP | 0.756 +/- 0.034 | 0.460 +/- 0.076 |
| RF | 0.776 +/- 0.025 | 0.472 +/- 0.068 |
| SVM | 0.768 +/- 0.031 | 0.474 +/- 0.067 |

These values are reference performance estimates, not inputs to the
Streamlit explainer. They help answer whether the fuzzy model offers useful
performance and interpretability relative to standard alternatives.

### Baseline evaluation caveat

The LDA grid includes combinations where shrinkage is requested with the
`svd` solver, although scikit-learn does not support that combination. The
notebook uses `error_score=np.nan`, so those invalid configurations produce
warnings and are excluded from ranking rather than stopping the search. The
valid LDA configurations still produce the reported result, but the warning
should be retained in the evaluation record and the grid can be cleaned up in
a future run by separating solver-compatible parameter combinations.

The baseline models are CPU-bound in scikit-learn. A GPU accelerator does not
speed up these estimators; any speed-up comes from the available CPU
allocation and `n_jobs=-1` parallelisation during grid search.
