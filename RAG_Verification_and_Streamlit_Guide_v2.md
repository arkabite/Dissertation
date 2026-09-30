# Part 1 — Methodology: Verifying Grounded Retrieval in the Rule Explainer

## Suggested write-up (methodology / evaluation section)

To verify that the LLM-based rule explainer (Stage 4) remained faithful to the
extracted fuzzy rule base rather than fabricating or omitting information, a
structured verification protocol was applied. A fixed set of probe questions
was designed to test properties of the retrieval-augmented generation (RAG)
pipeline: exact rule recall, channel-level disambiguation, refusal on
out-of-scope channels, cross-rule synthesis, numeric fidelity, refusal on
questions unanswerable from the rule base, correct handling of model-level
(cross-validation) statistics as distinct from individual rule statistics,
correct handling of cross-seed rule-stability statistics once the model was
extended to multi-seed ensembling, and — once a neurological explanation
layer was added — correct separation of sourced anatomical fact from
unsourced literature association, and correct verbatim transcription of
sourced channel labels into natural-language explanations. Each probe was
re-run after every code change to confirm the fix resolved the targeted
failure without regressing previously-passing behaviour.

This process surfaced ten distinct grounding failure modes across three
phases of development, none of which were apparent from casual use of the
system and each of which would have produced plausible-looking but incorrect
or misleading output if left unaddressed.

### Phase 1 — original single-seed rule explainer

1. **Digit-stripping tokenisation.** The keyword-overlap retrieval function
   tokenised text using `[a-zA-Z]+`, discarding numeric characters. Since
   fNIRS channel identity is frequently encoded numerically (e.g. `AF7` vs
   `AF8`, `PO1` vs `PO2`), this caused channel-specific queries to collapse
   onto a single ambiguous token (`chaf`), making the retriever unable to
   distinguish between electrodes. Fixed by extending the tokeniser regex to
   `[a-zA-Z0-9]+`.

2. **Masked filtering at small scale.** With `TOP_K=5` and an initial rule
   base of only four rules, the retrieval step returned the entire rule set
   on every query regardless of relevance scoring, meaning the system
   appeared to perform correct retrieval while the filtering logic was, in
   fact, never exercised. Identified by inspecting the retrieval function's
   return-size logic rather than by output alone; reported as a caution
   against validating retrieval systems solely on rule bases too small to
   expose their filtering behaviour.

3. **Underscore-delimited tokenisation mismatch.** Once the rule base grew
   past `TOP_K`, channel names stored as `tmb_s2_chAF8` tokenised to `{tmb,
   s2, chaf8}`, while a natural-language query such as "What does channel
   AF8 predict?" tokenised to `af8` — a token that never equalled `chaf8`
   under exact set intersection. This caused all channel-specific
   natural-language queries to silently fall back to an accuracy-ranked
   default rule set irrespective of the channel named. Fixed by adding a
   substring-containment fallback to the scoring function.

4. **Incomplete-result disclosure.** When the number of retrieved rules
   (`top_k=5`) was smaller than the total rule count, a query asking for
   "every rule" returned only the retrieved subset and presented it without
   qualification, giving the false impression of completeness. Fixed by
   passing the total rule count into the prompt and instructing the model to
   explicitly disclose partial results.

5. **Missing model-level context.** The structured rule file's
   `cv_performance_estimate` field was never injected into the prompt, so
   questions about overall model reliability could not be answered even
   though the information existed in the underlying data file. Fixed by
   passing this field through to the prompt as explicit model-level context,
   distinct from any individual rule's own accuracy figure.

6. **Stopword-driven false positives.** Common query words (e.g. "is", from
   fuzzy antecedent syntax "X IS High") produced non-zero overlap scores
   against every rule, preventing the retrieval function's zero-overlap
   fallback path from triggering correctly on genuinely out-of-scope
   queries. This produced a near-hallucination on a probe about a
   non-existent channel, where an irrelevant rule was surfaced with language
   implying relevance before the model self-corrected later in the same
   response. Fixed by excluding a stopword list from the token-overlap
   calculation.

### Phase 2 — multi-seed ensembling extension

After extending the fitting procedure from a single GA seed to a 3-seed
ensemble (`MainFold.py`), with rule-level `n_seeds_found` / `n_seeds_total`
recurrence statistics added to the extracted rule JSON, two further failure
modes emerged specific to how the explainer talked about this new stability
information:

7. **Cross-seed stability overclaim.** Before an explicit guard was added,
   the model asserted that rules found in only 1 of 3 independent refits
   were "stable enough to be meaningful" or that a pattern "remains present"
   across seeds — the opposite of what a low recurrence count indicates.
   This is a materially misleading claim for an XAI system: predictive
   performance across the ensemble is genuinely stable (CV test MCC = 0.408
   ± 0.111), but 0 of 17 unique rules recurred in more than one of three
   independent final-model refits, meaning individual symbolic rules should
   not be presented as stable findings. Fixed by adding an explicit
   instruction forbidding stability inference beyond the literal recurrence
   count, and prohibiting generalisation about what `dominance_score`
   "typically" implies beyond the specific number given for that rule.

### Phase 3 — neurological explanation layer

To make rule explanations more accessible to non-expert users, channel
codes were translated from opaque identifiers (`tmb_s1_chAF5h`) into
plain-English scalp locations and signal types. This introduced a new class
of risk — conflating genuinely sourced facts with unsourced or fabricated
neuroscience — which required its own sub-protocol and surfaced three
further failure modes:

8. **Unsourced neuroanatomical narration.** An initial version supplied the
   LLM with an AI-generated channel-to-region mapping containing specific
   cognitive-function claims (e.g. "anterior-frontal = focus, active
   decision-making") with no citation, and with the HbO/HbR task-load
   direction stated backwards relative to the project's own feature
   engineering code and cited source figure. Given a system prompt framing
   the assistant as an "expert neuroscience assistant" instructed to
   "explain what the brain is doing," the model narrated confident,
   specific cognitive claims per channel and per rule, and also fabricated
   an incorrect expansion for the MCC acronym in the same response —
   indicating that permissive framing degraded factual carefulness beyond
   just the neuroscience content. Fixed by replacing the ad-hoc mapping
   with a tiered glossary (`brain_mapping.json`): Tier 1 (channel location
   and HbO/HbR direction) sourced directly from this project's own
   feature-engineering code and, subsequently, from the dataset's own
   published montage description (functional grouping into
   frontal/motor/parietal/occipital regions, laterality convention
   confirmed as following the international 10-5 system); Tier 2
   (cognitive-function associations) retained as optional context but
   required to be explicitly hedged as general literature background, never
   presented as a finding of this rule or dataset; Tier 3, a small number of
   boundary-case channel codes (e.g. `PPO`, which spans both a
   parietal-adjacent and an occipital-adjacent cluster in the dataset's
   montage depending on the numeric suffix) explicitly flagged as
   unresolved rather than arbitrarily assigned to one region.

9. **Transcription fidelity under multi-condition load.** Channel labels
   were pre-computed deterministically (via regex, not the LLM) into
   `[Hemisphere Location (Signal)]` annotations and inserted directly into
   each rule's antecedent before the prompt reached the model. Despite this,
   when asked to explain multi-condition rules in natural language, the
   model silently swapped the hemisphere side and/or HbO/HbR signal type for
   2 of 5 rules in one response, while leaving other rules in the same
   response correct — indicating inconsistent recall/paraphrase of a
   pre-computed, verbatim-correct label rather than a systematic directional
   bug. This is a materially more dangerous failure than Phase 3's other
   modes, since the corrupted values sat inline with correctly-transcribed
   values in the same response, at the same apparent confidence level, with
   no textual signal distinguishing them. Fixed by instructing the model to
   copy bracketed labels verbatim rather than paraphrase them from memory;
   verified by cross-checking every hemisphere/signal pair in repeated
   probes against an independently-computed ground-truth table (i.e.
   re-running the deterministic labelling function directly on the rule
   file and diffing against the model's stated values, rather than
   eyeballing plausibility). A response format that echoed labels inside a
   two-column table, rather than free prose, empirically produced zero
   transcription errors across the runs tested and is recommended as the
   preferred response shape for this content.

10. **Tier 2 hedge scope too narrow.** The instruction requiring general
    literature associations to be explicitly hedged initially only reliably
    triggered on overtly interpretive claims (e.g. "linked to executive
    function"). An anatomy-flavoured functional gloss of a sourced location
    label (e.g. calling the occipital region "the visual cortex") slipped
    past the hedge requirement in one response, despite being the same
    category of unsourced-for-this-study claim. Fixed by widening the Tier 2
    instruction to explicitly cover functional shorthand for a region name,
    not only overtly cognitive claims, while leaving the bare
    `functional_region` label itself (frontal/motor/parietal/occipital) as
    Tier 1, since that grouping is sourced from the dataset's own montage
    description.

Following the tenth fix, repeated verification probes — including probes
specifically designed to re-trigger each of the nine prior failure modes,
cross-checked against an independently-computed ground-truth table for
every hemisphere/signal/region claim — passed cleanly and consistently
across multiple runs with varied phrasing. This iterative
probe-fix-reverify cycle, carried out across three distinct phases of the
project's evolution (single-seed → multi-seed ensemble → neurologically
explained), is presented as evidence that the explainer's output remains
grounded in the extracted rule base and the project's own sourced data,
rather than the underlying language model's prior knowledge or stylistic
tendencies. The failure modes themselves are reported as a contribution in
their own right: several (in particular #2, #6, #8, and #9) illustrate how
retrieval-augmented and glossary-augmented systems can appear correct under
casual testing while containing latent faults that only manifest under
specific data-scale, framing, or multi-condition conditions — and #9 in
particular demonstrates that even deterministically pre-computed, verbatim
ground-truth data can be silently corrupted by an LLM during natural-language
paraphrase, which is a materially different and arguably more concerning
risk than the model inventing content outright, since the surrounding
correct content provides no signal that anything is wrong.

## Suggested table for your dissertation appendix

| # | Failure mode | Symptom | Fix |
|---|---|---|---|
| 1 | Digit-stripping tokeniser | AF7/AF8 conflated to one token | Regex `[a-zA-Z]+` → `[a-zA-Z0-9]+` |
| 2 | Filtering masked by small rule count | Retrieval appeared correct but never filtered | Identified via code inspection; resolved implicitly once rule base exceeded `TOP_K` |
| 3 | Underscore-delimited token mismatch | Natural-language channel queries silently returned wrong/default rules | Added substring-containment scoring fallback |
| 4 | Incomplete-result disclosure | "List every rule" silently returned a partial list as if complete | Injected total rule count; instructed explicit disclosure |
| 5 | Missing model-level context | CV performance stats present in data file but never reached the model | Injected `cv_performance_estimate` into system prompt |
| 6 | Stopword-driven false positive | Out-of-scope channel query surfaced an irrelevant rule before self-correcting | Excluded stopwords from token-overlap scoring |
| 7 | Cross-seed stability overclaim | Model asserted 1/3-recurrence rules were "stable enough to be meaningful" | Instruction forbidding stability inference beyond literal recurrence count; forbade generalising `dominance_score` beyond the number given |
| 8 | Unsourced neuroanatomical narration | Model invented per-channel cognitive-function claims and inverted HbO/HbR task-load direction; also fabricated an MCC acronym expansion under permissive "expert neuroscience assistant" framing | Tiered glossary (`brain_mapping.json`): Tier 1 sourced from project code + dataset montage paper; Tier 2 general literature always hedged; Tier 3 explicitly-flagged ambiguous channels (`PPO`) |
| 9 | Transcription fidelity under multi-condition load | Hemisphere side and/or HbO/HbR signal silently swapped for 2 of 5 rules despite correct pre-computed labels being present in the prompt | Instruction to copy bracketed labels verbatim, not paraphrase from memory; verified against an independently-computed ground-truth table; tabular response format empirically eliminated the error |
| 10 | Tier 2 hedge scope too narrow | Anatomy-flavoured shorthand ("visual cortex" for occipital) slipped past the hedge requirement | Widened Tier 2 instruction to cover functional shorthand, not only overtly cognitive claims |

---

# Part 2 — Guideline for Stage 5/6 (Streamlit UI)

## Goal and current architecture

The verified `rag_core.py` logic (rule loading, retrieval, prompt
construction, channel-glossary translation, Ollama call, and deterministic
fallback) is used directly by the Streamlit chat interface. A separate
FastAPI service is not required for the current application. `Streamlit_app.py`
loads the data and model client in a cached Streamlit resource, calls
`rag_core.py` in the same process, and uses `viz_tools.py` to build
deterministic visualization payloads. This lets a non-technical user ask
questions about the fuzzy rule base — including plain-English neurological
framing — through a browser instead of a terminal loop.

The static-dashboard condition (`dashboard_app.py`) remains separate for the
RQ2 usability comparison. It reads the final structured rule data but does
not import `rag_core.py`, call Ollama, or require an API key. The comparison
is therefore between a conversational interface and a browse/filter
interface, not between two different explanation engines.

## 1. Project layout (current)

```
Dissertation/
├── Streamlit_app.py           # chat UI and direct RAG orchestration
├── dashboard_app.py           # static browse/filter UI (comparison condition)
├── rag_core.py                # retrieval, prompts, Ollama, and fallback
├── viz_tools.py               # deterministic visualization payload builders
├── viz_components.py           # chart, chord, and 3D rendering helpers
├── channel_atlas.json         # channel positions used by visualizations
├── extracted_rules_final_raw.json   # multi-seed ensembled rules + stability data
├── extracted_rules_final.json # rule file currently read by dashboard_app.py
├── domain_background.md        # curated, paraphrased project-background text
├── brain_mapping.json          # tiered channel/signal/region glossary
├── requirements.txt
└── .streamlit/
   ├── config.toml              # Amber Desk theme
   └── secrets.toml             # optional local OLLAMA_API_KEY, NOT committed
```

`backend.py` and `api_client.py` are legacy FastAPI code paths. They are not
part of the current Streamlit execution path and do not need to be started.

`requirements.txt`:
```
streamlit
ollama
python-dotenv
mne
fastapi
uvicorn
```

Run the current app with:

```text
streamlit run Streamlit_app.py
```

No separate `uvicorn backend:app` command is needed. The active app does not
import `python-dotenv`; that dependency remains only for older scripts.

## Current chat request flow

1. `Streamlit_app.py` initialises `st.session_state.history`, then loads the
   rules, channel atlas, domain background, brain mapping, and Ollama client
   through the cached `load_app_state()` function.
2. The API key is read from `st.secrets["OLLAMA_API_KEY"]`. If it is absent,
   the app remains usable with deterministic rule matching.
3. Each question first passes through `rag_core.is_in_scope()`. Out-of-scope
   questions are refused without an LLM call.
4. In-scope questions call `rag_core.ask_with_visualization()`. If the client
   is unavailable or the call raises an exception, the app uses
   `retrieve_rules()` and `build_fallback_answer()` instead.
5. The response is stored in session state and rendered on each Streamlit
   rerun. Retrieved rules appear in a `Rules used` expander.
6. For a requested topomap or chord diagram, `viz_tools.py` resolves channels
   against the retrieved rules and derives plotted data from those rules and
   `channel_atlas.json`; the LLM does not supply positions or activation
   values.

The sidebar shows model-level cross-validation metrics, the full rule list,
all-rule visualizations, and a `Clear chat` button. When the LLM is
unavailable, the app shows a warning and returns raw retrieved rules instead
of an unhandled traceback.

## 2. Design decisions worth making deliberately (not defaults)

- **Show retrieved rule IDs and available statistics per answer**, not just
   the answer text — this turns the verification work into a visible feature
   and lets a reader audit grounding themselves. The current
   `Streamlit_app.py` `Rules used` expander shows rule ID, antecedent,
   consequent, and per-rule accuracy. The current `dashboard_app.py` table
   shows rule ID, antecedent, consequent, dominance score, and accuracy.
   Cross-seed recurrence values remain available in the structured data and
   prompt context, but are not currently rendered in either UI.
- **Surface CV stats and the cross-seed stability limitation clearly.** The
   current chat sidebar and dashboard display model-level CV metrics. The
   cross-seed recurrence data is passed into the chat prompt, but the current
   UI does not display a separate permanent warning banner or recurrence
   column; users should not infer rule-level stability from accuracy alone.
- **Keep the neurological glossary aligned between UI and prompt.**
   `rag_core.py` uses the Tier 1 / Tier 2 / Tier 3 distinction encoded in
   `brain_mapping.json` when constructing prompts. The current chat UI does
   not expose that file in a dedicated glossary panel, while
   `dashboard_app.py` shows its own plain-language glossary. Aligning those
   two visible glossary experiences remains a usability improvement, rather
   than a prerequisite for the direct RAG path.
- **Treat the domain-background and brain-mapping files as citable data,
  not prose you wrote once and forgot about.** `domain_background.md` is
  paraphrased from the project's own research proposal; `brain_mapping.json`
  cites the project's feature-engineering code and, where available, the
  dataset's own published montage description. Both should be referenced
  by name in the dissertation's methodology section as the actual grounding
  sources, distinct from the LLM's own training knowledge.
- **Handle the API key via `st.secrets`**, not a hardcoded string or a
   committed `.env`. The active app passes the secret directly to
   `rag_core.get_client()`; there is no backend endpoint or health-check call.
- **Handle failures around `client.chat(...)`.** The current app catches
   failures from `ask_with_visualization()`, shows a toast, and returns a
   deterministic rule-based answer instead of an unhandled traceback during a
   demo or examiner session.

## 3. Suggested order of remaining work

1. Re-run the full 10-probe verification set (Parts 1 above) once more after
   any further prompt change, before treating the explainer as stable for
   write-up. Run the probes through the current direct Streamlit/RAG path,
   not through the legacy FastAPI files.
2. Confirm the citation for the NIRS montage description used in
   `brain_mapping.json`'s functional-region grouping (author/year/title) and
   fold it into `domain_background.md`'s citation trail.
3. Optionally log Q&A pairs (with retrieved rule IDs and stability figures
   attached) to a file for inclusion as evaluation transcripts in the
   dissertation appendix — this directly documents the failure-mode table
   above with real transcripts rather than paraphrased examples.
4. If accessible-outside-your-machine deployment is needed for a viva/demo,
   deploy via Streamlit Community Cloud, setting `OLLAMA_API_KEY` in the
   app's secrets panel rather than in code. Select `Streamlit_app.py` as the
   app entry point; do not configure a FastAPI process.
