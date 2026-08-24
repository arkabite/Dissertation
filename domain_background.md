# Domain background (from the project's research proposal literature review)

This is general context from the dissertation's own literature review, kept
separate from the fitted model's rules. It exists to help explain *why* this
project looks the way it does and what fNIRS is, in plain terms. It must
never be presented as something this specific classifier discovered, and it
contains no claim about which channel does what in this model's own rules.

## What fNIRS measures

Functional near-infrared spectroscopy (fNIRS) is a non-invasive
neuroimaging technique that shines near-infrared light into the scalp and
measures the reflected signal to track changes in oxygenated (HbO) and
deoxygenated (HbR) haemoglobin in the cerebral cortex. Because active brain
regions consume more oxygen, these haemoglobin changes are used as an
indirect proxy for local neural activity. Recordings are time-series in
nature, span many channels tied to different scalp locations, and are
commonly affected by noise, motion artefacts, and physiological variation —
which is why preprocessing (filtering, normalisation, feature extraction)
happens before model training.

## Why this project uses an interpretable model instead of a black box

fNIRS analysis is frequently carried out with black-box models (deep
networks, ensembles) that give little insight into which channels or
signal properties actually drove a prediction. This project's motivation is
to use an interpretable fuzzy rule model (ExFuzzy) instead, so predictions
can be inspected and questioned rather than accepted at face value, and to
let non-expert users query those rules conversationally rather than reading
a static plot or table.

## Prior related work (cited context, not this project's own finding)

The proposal cites earlier fNIRS work on cognitive workload / mental-state
classification using prefrontal signals, and on combining EEG with fNIRS
for workload measurement. This establishes fNIRS as a workable input for
workload-style classification *in general* — it does not specify which
individual channel matters for *this* project's dataset or classifier, and
should not be cited as if it did.

## What this background deliberately does NOT contain

- No per-channel neuroanatomical claims (e.g. no "this channel corresponds
  to working memory" mapping). If asked why a specific channel like AF7 or
  AF8 appears in a rule, the honest answer is that the fitted rule found a
  statistical association in this dataset — not that the literature
  predicted that specific channel in advance.
- No information about which channels this particular classifier's rules
  actually use — that only comes from the rules JSON.
- No claim that resolves or explains the low cross-seed rule stability
  reported separately — that is a property of this model's fitting
  procedure, not something the background literature addresses.
