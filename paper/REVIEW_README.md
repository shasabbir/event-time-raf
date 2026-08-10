# Peer Review — *Event-TimeRAF: Event-Aware Retrieval-Augmented Foundation Model for Explainable Air Quality Forecasting Under Concept Drift*

**Reviewer role assumed:** senior reviewer, IEEE Transactions–tier journal (TKDE / TNNLS / IEEE TGRS / *Environmental Modelling & Software* class).
**Material reviewed:** `main.tex` (754 lines), `references.bib` (43 entries), `figures/` (9 figures: 4 vector schematics + 5 raster result plots).
**Review date:** 2026-08-10.

---

## Part 0 — Recommendation

> **REJECT** as a methods contribution.
> **Encourage resubmission** as a reframed *negative-result / evaluation-protocol* paper, subject to the mandatory experiments in §3.

| Criterion | Score (1–5) | Note |
|---|---|---|
| Novelty of method | **1.5** | Every named module reduces to standard practice (data hygiene, backtest embargo, hand-weighted similarity, forecast averaging, logging). |
| Technical soundness | **2.0** | Two concrete mathematical defects; retrieval pool is under-designed by ~200×; inference statistics are anticonservative. |
| Experimental rigour | **1.5** | Zero deep-learning baselines, one under-powered TSFM checkpoint, no comparison to the method being extended (TimeRAF). |
| Support for claims | **1.0** | All three title claims (*event-aware*, *foundation model*, *concept drift*, *explainable*) are contradicted or unevaluated by the authors' own tables. |
| Reproducibility / transparency | **3.5** | Genuine strength. Manifests, hashes, artifact IDs, honest limitations section. Highest-value part of the submission. |
| Presentation | **2.0** | Heavy redundancy (contributions stated 3×), 21 floats, undefined symbols, degraded English throughout. |

**One-sentence assessment.** The paper is an unusually honest and well-audited report of a *null result* — event-context retrieval does not improve 24 h PM2.5 forecasting over a plain weather+calendar XGBoost model — but it is packaged and titled as a positive methodological advance it does not deliver, and the null result is not yet interpretable because the retrieval mechanism was built at a scale (191 candidate windows) at which no retrieval method could have succeeded.

---

## Part 1 — Summary of the submission (as the reviewer understands it)

The authors forecast hourly PM2.5 for Los Angeles County (L = 168 h lookback, H = 24 h horizon, 2019–2024) using a county-median target derived from EPA AQS parameter 88101, NOAA ISD weather from one station, calendar features, and NOAA Storm Events records. Three named modules are proposed:

- **SACB** — provenance-audited alignment of the four sources into an 85-dim origin vector (23 PM2.5 + 14 weather + 9 calendar + 39 event).
- **LSER** — a 191-window training-history knowledge base with a strict candidate-target embargo (`u_i + H < t − L + 1`) and a fixed four-channel similarity score (α = 0.5 time-series, β = 0.2 weather, γ = 0.1 calendar, δ = 0.2 event), top-k = 8.
- **DFEH** — 24 independent XGBoost regressors over 151 features, a separate frozen `chronos-bolt-small` path fused at output level (ω = 0.75), a 5-component heuristic drift score, and a 7,199-row `explanations.parquet` evidence log.

**Headline results (Table VI):** M04 (XGBoost + weather + calendar) MSE **26.185**; M09 (full Event-TimeRAF) MSE **26.712**; M10 (frozen Chronos-Bolt) 28.941; M11 (Chronos + retrieval) 28.709, paired 95 % CI (−0.665, 0.152).

I want to state at the outset that the authors deserve credit for reporting this honestly rather than tuning until the proposed method won. That candour is rare and is the main reason I recommend resubmission rather than outright rejection. The problems below are about *whether the experiment can support any conclusion at all*, not about the sign of the result.

---

## Part 2 — Major concerns

Each item states the **claim**, the **evidence** (line/table references from the submission), and the **required action**.

---

### M1. The retrieval knowledge base is ~200× too small for the central hypothesis to be testable

**Claim.** The paper's central negative finding ("event-context retrieval does not help") is uninterpretable, because the retrieval pool was decimated by a design choice that is not required for leakage safety.

**Evidence.**
- §VI-B-1 and Table VIII: the knowledge base contains **191 non-overlapping windows** at a **192-hour stride** over the ~4.2-year training span.
- The embargo in Eq. (9), `u_i + H < t − L + 1`, is a *query–candidate* constraint. It is fully sufficient for leakage safety on its own. **Non-overlap among candidates is a redundancy/diversity concern, not a validity concern.**
- A 1-hour stride over the same training span would yield **≈ 36,600 admissible candidates**. The reported design therefore discards **≈ 99.5 %** of legally retrievable history for no leakage-related reason.
- Consequence, quantified from the authors' own numbers: retrieving top-8 of 191 means selecting at the 4th percentile of a pool containing only ~48 windows per season. There is essentially no probability of locating a genuine meteorological analogue.
- Table VII confirms this directly. k = 1 gives MSE **65.129** — *worse than persistence* (47.649) and statistically indistinguishable from weekly-seasonal naive (65.606). **The single nearest neighbour in the knowledge base is no more informative than blindly copying last week.**
- Figure 7 (right panel) shows retrieved top-8 candidates with time-series similarity as low as **0.15–0.3**, i.e. the ranker is forced to return near-orthogonal windows because nothing better exists in the pool.

**Why this is decisive.** The paper reports a null result for retrieval and attributes it to the *information content of events and analogues*. The far more parsimonious explanation is that the pool was too sparse to contain analogues. These two explanations are not distinguished anywhere, and the second is strongly favoured by Table VII and Figure 7.

**Required action.** Rebuild the knowledge base at stride ∈ {1, 6, 24} h (yielding 10³–10⁴ candidates) under the *identical* embargo, and re-run M05–M09. If candidate redundancy is a concern, handle it at retrieval time with MMR-style diversification or a minimum temporal separation *between selected neighbours*, not by decimating the pool. Report KB size as an explicit axis of the sensitivity study alongside k.

---

### M2. The k-sensitivity result shows retrieval is converging to climatology, not finding analogues

**Claim.** The retrieval-only baselines are, to within measurement precision, unconditional-mean predictors. The one "validated" LSER ablation is therefore a comparison against a no-skill straw man.

**Evidence (all derived from the submission's own numbers).**
- From M04: MSE 26.185 at R² = 0.379 ⇒ target variance **σ²_y ≈ 42.17** (σ_y ≈ 6.49 µg/m³). This is consistent across the table (persistence 47.649 ⇒ R² = −0.130 ✓).
- **M05 random retrieval: MSE 42.367 ⇒ R² = −0.005.** Averaging 8 randomly chosen weeks reproduces the unconditional mean *exactly*. The random baseline has literally zero skill.
- M06 cosine retrieval: 38.083 ⇒ R² = 0.097. Ten percent of variance explained.
- Table VII is **monotone decreasing in k**: 65.129 → 41.237 → 38.083 → 36.488 for k = 1, 4, 8, 16. Monotone improvement with neighbour count, converging from above toward σ²_y, is the signature of **variance shrinkage toward the mean**, not of improving retrieval quality. A retriever that found real analogues would show a U-shape with an interior optimum.
- Therefore the ablation row *"M06 − M05, LSER cosine/random ranking = −4.284, CI (−5.862, −2.759)"* in Table VII reads, honestly restated: **"ranking by cosine similarity beats a zero-skill mean predictor by R² = 0.10."** This is presented in §VII-B as evidence that the LSER component works.

**Required action.** Delete the claim that the LSER is validated by M06 − M05. Replace the random baseline with defensible nulls: (i) hour-of-day × month **climatology**, (ii) calendar-only retrieval (α = β = δ = 0, γ = 1) to isolate whether cosine adds anything beyond seasonal/diurnal matching. Report a **skill score** relative to climatology, per horizon, rather than pooled R².

---

### M3. The event-similarity channel is mathematically near-inert, and the paper's own diagnostics prove it

**Claim.** The event channel cannot change the retrieval ranking, so the "event-aware" contribution was untested rather than falsified.

**Evidence.**
- Event base rate: 117 event days over 2,192 days = **5.3 %** (Table II). Consequently `s_e(q, i) ≈ 0` for the overwhelming majority of query–candidate pairs, and `δ · s_e` acts as a near-constant additive offset that **cannot reorder the top-k**.
- Table VII confirms it numerically: *"M08 − M07, LSER event-hybrid ranking = **0.005**"* — a 0.02 % change in MSE from switching cosine → event-hybrid ranking. That is not a null result; that is an inactive code path.
- Figure 7 (left panel) is independent confirmation: the `hybrid` and `hybrid_no_event` score densities are near-coincident (modal shift ≈ 0.05 on a 0–1 scale) and overlap almost completely.
- The 39 event dimensions are 46 % of the 85-dim context vector but are derived from **204 records over six years** (Table III). Most of those 39 columns are near-constant zero.

**This yields a second, competing explanation for the paper's headline null** that the paper never considers: M09 > M08 > M07 > M04 in MSE may be **pure dimensionality-induced degradation** — 39 sparse, mostly-zero columns and a 51-dim retrieval block injecting variance into unregularised trees — rather than a finding about event information. The two are not separated anywhere.

**Required action.**
1. Sweep δ ∈ {0, 0.2, 0.5, 0.8, 1.0} and **normalise `s_e` within the candidate pool** so the channel can actually reorder candidates. A paper whose title is "Event-Aware" must include a δ sensitivity study; only k is currently studied.
2. Add **event-stratified retrieval**: when the query origin is event-flagged, restrict the pool to event-flagged candidates. This is the design that would actually test the hypothesis.
3. Separate "events carry no signal" from "sparse columns hurt XGBoost" via permutation importance, a feature-selection arm, and an L1/L2-regularised linear control on the identical 151 features.

---

### M4. The two subset results most central to the paper contradict its thesis, and are not discussed

**Claim.** Table IX shows the proposed model is *specifically worse* on exactly the two regimes it is named after. This is the most important finding in the paper and it is passed over.

**Evidence (Table IX).**

| Regime | M04 (baseline) | M09 (proposed) | Proposed model is |
|---|---|---|---|
| **Event** subset (532 origins) | MSE 9.451 | MSE **11.301** | **19.6 % worse** |
| **Drift** subset (560 origins) | MSE 27.110 | MSE **31.244** | **15.2 % worse** |
| Drift subset, Chronos (M10) | — | 28.333 | beats the drift-aware model |

The paper's only comment (§VII-C) is a defensive note about not over-reading the event subset. There is no acknowledgement anywhere — abstract, results, discussion, or conclusion — that **the event-aware model is worse on event origins and the drift-aware model is worse on drift origins.** For M04, the drift penalty is 3.8 %; for M09 it is 15.2 %. Adding drift features made the model *four times more sensitive* to drift.

**A mechanistic hypothesis the authors should test.** The drift threshold is calibrated at the *validation* Q0.90 (Eq. 18). Drift-flagged test origins therefore sit in a region of drift-feature space that is sparsely populated in training, so the trees extrapolate on a constant leaf. This is a testable, fixable defect — not an inherent property of drift features.

**Required action.** Promote both rows to headline findings. State them in the abstract. Add the mechanistic diagnostic (drift-feature distribution, train vs. val vs. test; leaf-occupancy for flagged origins).

---

### M5. Cross-subset MSE comparison is invalid; the event subset is a *low-pollution* subset

**Claim.** The event subset has ~3× lower MSE than the non-event subset (9.451 vs 27.520) because it has lower target variance, not because it is easier to model — and the reason it has lower variance inverts the paper's motivation.

**Evidence and domain argument.**
- NOAA Storm Events for Los Angeles County is dominated by **high wind, heavy rain, and flood** entries. §VI-B-2 lists exactly these. All three **ventilate or wet-deposit** particulate matter and *lower* PM2.5.
- The Introduction motivates the work with "wildfire smoke" and "stagnation." **Storm Events records neither.** Stagnation is not an event type at all, and Storm Events wildfire coverage is driven by damage/casualty/NWS-interest reporting thresholds, not by smoke transport.
- The authors themselves note (§V-D, §VII-F) that NOAA HMS smoke — the *one* source that captures the dominant driver of extreme LA PM2.5 — **was not used**.
- Net effect: the paper conditions retrieval on a set of events that are **negatively associated** with high PM2.5, then reports surprise that the event subset is easy.

**MSE is not comparable across subsets with different target variance.** Every subset comparison in Table IX is uninterpretable as stated.

**Required action.** Report per-subset target variance and a **variance-normalised skill score** (or MSE relative to within-subset climatology/persistence). Either incorporate an audited HMS smoke cache — which the authors already scoped and cited (`noaahms2026`) — or remove wildfire smoke from the motivation. As written, the motivation and the operationalisation are about different phenomena.

---

### M6. The county-median target is constructed to be insensitive to the events being retrieved

**Claim.** This is a construct-validity failure that undercuts the study design independently of any modelling choice.

**Evidence and argument.**
- §VI-A: no single monitor met the coverage gate, so the target is the **hourly median across all LA County AQS monitors**.
- LA County spans coastal Long Beach, the central basin, the San Fernando and San Gabriel valleys, and the **Antelope Valley** high desert — meteorologically distinct regimes in one aggregate.
- A median is a **rank statistic**. A wildfire plume that elevates 2 of ~15 monitors by 100 µg/m³ **does not move the median at all.** The target is engineered to suppress precisely the localised, high-impact signal the event module exists to capture. The event hypothesis cannot succeed against this target.
- The median also **removes spatial variance**, artificially reducing task difficulty. Figure 9 shows the representative case oscillating in the 4–16 µg/m³ range — an implausibly narrow band for LA hourly PM2.5, and direct evidence of the smoothing.
- **Unacknowledged non-stationarity:** the AQS monitor roster changes over 2019–2024 (sites added, retired, relocated). A median over a *time-varying* monitor set is a **time-varying definition of the target**. Some fraction of the "concept drift" the paper measures in Eq. (16)–(18) is an artifact of the target's own definition. This is not mentioned.
- **Single weather station.** Station 72297023129 (Long Beach / Daugherty Field) is coastal and marine-layer dominated. It is "10.28 km from the PM2.5 target centroid," but the centroid of a >10,000 km² county is not a location where weather is representative of anything. The stagnation proxy and wind features derived from one coastal station cannot describe basin or Antelope Valley ventilation.
- **Environmental-justice concern.** The Introduction claims public-health relevance. LA County PM2.5 burden is a canonical EJ issue with severe intra-county disparities (e.g. the Wilmington/Long Beach port corridor). A county median **erases exactly the disparity** that makes the problem health-relevant. A paper making a public-health argument should not aggregate away its own outcome of interest without discussion.

**Required action.** Move to site-level targets for the highest-coverage 3–5 monitors (accepting a shorter usable record), or at minimum add a site-level arm as the primary event-hypothesis test with the county median retained as a secondary aggregate. Justify the weather-station choice per target site, or use multi-station / gridded reanalysis (ERA5, HRRR) covariates. Add a study-area map with monitor and station locations — for an air-quality paper, its absence is itself a deficiency.

---

### M7. Two concrete mathematical defects

**M7a — The uncertainty proxy is dimensionally coherent but statistically incoherent.**

Eq. (19): `u_t^proxy = sqrt( σ̄²_{R,t} + ē²_val )`, where per §VI-D-4, `ē_val` is the **mean absolute** validation residual and `σ̄_{R,t}` is a **standard deviation** of retrieved trajectories.

You cannot add a squared MAE to a variance and call the root a scale parameter. For Gaussian residuals, MAE ≈ 0.798 σ, so `ē²_val` understates the residual variance by ≈ 36 %. If a variance decomposition is intended, use the validation **RMSE**. As written the quantity has no defined statistical meaning, and the paper nonetheless writes it into every one of 7,199 evidence records as the model's uncertainty statement.

**M7b — The drift score is degenerate by construction.**

Eq. (17): `d_{t,j} = (1/6) · clip( (r_{t,j} − med_tr(r_j)) / max(1.4826·MAD_tr(r_j), ε), 0, 6 )`.

Median-centring followed by clipping **at zero from below** means that, for every component, **approximately 50 % of origins receive exactly `d_{t,j} = 0`**. For components that are already non-negative by construction — e.g. `|μ₂₄ − μ₁₆₈|` and the weather Mahalanobis-like term — the lower half of the distribution is collapsed to a point mass. `S_t` in Eq. (18) is therefore a **zero-inflated, one-sided** score whose value is dominated by whichever single component happens to have the heaviest right tail. **Figure 8 visibly confirms the zero inflation** (dense mass pinned at 0.0). The composite never exceeds 0.52 on a [0, 1] scale.

Additionally: discarding negative deviations throws away *half the drift information*. A collapse in recent variance, or an unexpected *increase* in retrieval similarity, are both regime changes and both map to 0.

**Required action.** Fix Eq. (19) to use validation RMSE. Justify or remove the one-sided clip in Eq. (17); report the marginal distribution of each `d_{t,j}` and the correlation structure among the five components (they are likely highly collinear — the similarity-drop term and the weather-shift term measure overlapping phenomena). Benchmark the composite against established detectors (ADWIN, KSWIN, Page-Hinkley, kernel-MMD two-sample). Currently a five-term ad-hoc heuristic with no comparator is offered as a contribution.

---

### M8. The statistical inference is anticonservative, and the multiplicity is unaddressed

**Claim.** No significance statement in Table VII survives scrutiny, including the two the paper relies on.

**Evidence.**
1. **Block length is far too short.** 24-hour blocks over 7,199 origins gives ~300 blocks. But (a) consecutive origins share **167 of 168** lookback hours and **23 of 24** target hours, and (b) PM2.5 dependence is governed by **synoptic weather episodes of 3–7 days** plus multi-day stagnation. The effective number of independent episodes over an 11-month test period is on the order of **40–60**, not 300. Twenty-four-hour blocks therefore produce **CIs that are materially too narrow**. Every interval in Table VII is optimistic — including `M07 − M04 = 0.487, CI (0.015, 1.028)`, whose lower bound clears zero by 0.015.
2. **B = 500 is too few** for 95 % interval endpoints on an MSE difference, whose bootstrap distribution is tail-sensitive. 2,000–10,000 is standard.
3. **Eight comparisons, no multiplicity control.** Table VII reports 8 paired intervals with no Bonferroni/Holm adjustment and no pre-registered family. Under Holm correction, `M07 − M04` would not be declared significant.
4. **No standard forecast-comparison test.** Diebold–Mariano (with HAC variance) or Giacomini–White is the accepted instrument here and is absent.

**A consequence the authors should note in their own favour:** correcting the block length weakens the two intervals they lean on (`M04 − M03`, `M06 − M05`) but *also* removes the only nominally significant harm result (`M07 − M04`). The honest post-correction reading is "**no retrieval comparison in this study is statistically resolvable**," which is a cleaner and more defensible claim than the current mixed narrative.

**Required action.** Re-run with block length ≥ 168 h (or a stationary bootstrap with data-driven mean block length via Politis–White), B ≥ 2,000, Holm-adjusted across the pre-registered family, and add Diebold–Mariano as the primary test.

---

### M9. The baseline set is inadequate for a journal submission

**Claim.** The paper surveys ~25 forecasting methods in §II and compares against **none** of them.

**Absent baselines that are standard and cheap:**
- **Deep sequence models:** no LSTM, GRU, DLinear, PatchTST, iTransformer, TimesNet, Informer, TimeMixer — all reviewed at length in §II-A, all with public reference implementations, all runnable in well under the paper's own 1,577 s budget on the reported data volume. DLinear and PatchTST in particular are the accepted competitive references and their omission alone is grounds for major revision.
- **A regularised linear control** on the identical 151 features, to establish that gradient boosting is necessary at all.
- **LightGBM**, cited in §II-A and not run.
- **SARIMAX / ARIMAX** with the same exogenous covariates.
- **Climatology** (hour-of-day × month), the correct no-skill reference. Persistence and seasonal-naive are not adequate substitutes.

**The TSFM arm is under-powered by construction:**
- Only **one checkpoint** — `chronos-bolt-small` — the smallest in the family. `base` and `large` are inexpensive; neither is run. No fine-tuning is attempted.
- **Chronos-Bolt cannot ingest exogenous covariates.** Comparing it against M04, which uses weather and calendar, is therefore not a like-for-like comparison and cannot support any statement about foundation models versus supervised models. If the point is "TSFMs vs. local supervision," a covariate-capable TSFM (Moirai, TimesFM 2.x with covariates, TTM with exogenous channels) is required.
- **No probabilistic evaluation.** Chronos-Bolt is *quantile-native* — it emits a predictive distribution for free. The authors discard it, collapse to a point forecast, and then invent an ad-hoc "uncertainty proxy" (Eq. 19, itself defective per M7a). This is backwards. **CRPS, pinball loss, and PIT/interval coverage should be the headline uncertainty metrics**, and they would cost nothing to produce.

**TimeRAF itself is never compared against.** The paper's title mirrors TimeRAF, §II-C and Table V position the work as an adaptation of it, and §IV/§VIII concede that neither the learned dual encoder nor Channel Prompting is implemented. The single closest method in the literature is therefore absent from the experiments. A reader cannot tell whether the null result is a property of event-context retrieval or of *this particular simplified retriever*.

---

### M10. Metric choice is disconnected from the stated motivation

**Claim.** The paper motivates itself on public-health warning and then evaluates only symmetric squared/absolute error on raw concentrations.

**Evidence and argument.**
- §I: "public health warning, exposure avoidance, transportation scheduling." The operational decision is a **threshold exceedance** — the 24 h AQI breakpoints (35.4 µg/m³ "Unhealthy for Sensitive Groups", 55.4 "Unhealthy"). **No exceedance classification skill is reported anywhere:** no precision/recall, no critical success index, no ROC/PR, no lead-time-to-detection.
- PM2.5 is strongly **right-skewed with a heavy tail**. Raw-scale MSE is dominated by a handful of smoke hours, which (a) makes the metric high-variance and (b) rewards conservative amplitude damping. Log-scale or quantile-based metrics are standard in the air-quality literature and are not reported.
- **Figure 9 shows the damping directly.** In the chosen "representative" case, the observation reaches ≈ 16 µg/m³ while both M04 and M09 cap at ≈ 12 — systematic peak under-prediction. For a public-health application, **under-forecasting peaks is the one failure mode that matters**, it is visible in the authors' own showcase figure, and it is not discussed.
- **Absolute performance is never contextualised.** M04 achieves R² = 0.379 and RMSE 5.117 µg/m³ against a mean level of ~10 µg/m³ — roughly 50 % relative error. Published single-station PM2.5 models at 24 h routinely report R² in the 0.7–0.9 range. The paper never tells the reader that its best model is weak in absolute terms.
- **No gain over persistence at h = 1.** Figure 4 shows all models — persistence included — clustered at MSE ≈ 10 at h = 1; separation only emerges from h ≈ 5. This is an important and unstated characterisation of where any skill actually lives.
- Pooled R² over all 24 horizons against a single global ȳ flatters every model and is not comparable to a per-horizon skill score.

**Required action.** Add: exceedance-classification metrics at AQI breakpoints; per-horizon skill relative to climatology; log-scale or quantile error; CRPS/coverage for the probabilistic arm. Contextualise R² = 0.379 against published PM2.5 results.

---

### M11. The test window excludes the regime the paper is about

**Claim.** The evaluation period is an unusually benign year for LA PM2.5 extremes, chosen (apparently inadvertently) so as to exclude the most extreme event in the record.

**Evidence.**
- Figures 7 and 8 reveal the test period as **2024-02 to 2024-12** (not stated in the text — the reader must read it off the axes; the paper says "2019–2024" throughout).
- 2024 was a **wet, low-fire-activity year** in the LA basin following consecutive atmospheric-river winters.
- The **January 2025 Palisades and Eaton fires** — the most severe PM2.5 episode in modern LA County history, and the precise regime this framework claims to address — fall **immediately outside** the test window.
- The paper is dated 2026 (run ID `20260723T112033170131Z`; references "Accessed 2026-07-17"). **Data through 2025 was available and was not used.** No justification is given for truncating at 2024.

This is the strongest available criticism of the paper's external validity: an event-aware, drift-aware framework was evaluated on the one recent year with the fewest events and least drift, while a year containing an extraordinary event sat just past the cutoff.

**Required action.** Extend the record through 2025 and re-run. The January 2025 episode is the natural stress test and would transform the paper's evidentiary value regardless of the outcome.

---

### M12. Internal inconsistencies and unexplained data attrition

These must be resolved because the paper's principal claimed strength is auditability.

1. **~8–9 % of origins vanish without explanation.** 48,332 valid windows over ≈ 52,600 study hours = **91.9 %**. In the test period specifically, Figure 8 spans ≈ 330 days ≈ 7,920 hours but only **7,199 origins** exist (90.9 %), and Figure 7 shows **visible vertical data gaps around 2024-07/08 and 2024-11/12**. Yet Table II asserts "PM2.5 observed coverage 0.9997" and "Weather complete coverage 0.9942." **The stated coverage gates are 15× more optimistic than the actual origin attrition.** Provide a per-stage origin-attrition table (raw hours → aligned → valid windows → evaluated origins) with the reason for each drop, and reconcile it with Table II.
2. **Test-split arithmetic.** 15 % of 48,332 = 7,249.8, but 7,199 origins are reported. The 51-origin gap is unexplained.
3. **The audit gates have no pass criteria.** Table II reports five "Pass" verdicts with **no stated thresholds**. An audit table without pre-registered thresholds is not an audit.
4. **"Event overlap days = 2,192 — Pass" is vacuous.** 2,192 days ≈ the entire 6-year study period. The gate reduces to "the sources overlap in time," which is true by construction. Reporting it as a passed quality gate is misleading. The term "source-overlap days" is never defined.
5. **"Coverage 0.9997" is vacuous for a median target.** The county median exists whenever *any* monitor reports. The gate therefore says nothing about individual monitor completeness and cannot function as the data-quality check it is presented as.
6. **Event category count contradicts itself.** Table II: "Qualifying event categories 3." §VI-B-2 lists five (wildfire, high wind, heavy rain, flood, other). Which three qualified, and under what rule? Since the answer determines whether wildfire is in the feature set at all, this is substantive, not cosmetic.
7. **A00 and A01 are never defined.** They carry an ablation row each in Table VII (`M09 − A00, event context`) but appear only as "auxiliary" models in §VI-D and are absent from Table V. If A00 is "M09 minus event features," state the resulting dimensionality and how it differs from M07.
8. **M12 is mentioned only to be excluded.** §VI-D: "Post-Hoc M12 artifact is not part of the manifest-backed result set and hence is excluded." A model referenced only by its exclusion, with no statement of what it was or what it scored, reads as selective reporting. Either report it with its caveat or remove all mention.
9. **Limitation 5 contradicts the reproducibility claim.** The authors concede that M05/A01 cannot be regenerated (NumPy 2.0.2 → 2.4.1) and were **reconstructed from stored evidence rows**. Note that this is a *fixable engineering defect*, not a scientific limitation: NumPy guarantees stream stability for `RandomState`, **not** for `Generator`/`default_rng`. Switching generators — or simply persisting the selected indices — resolves it. Elevating it to a stated limitation of the science, in a paper whose central selling point is manifest-backed reproducibility, is the wrong framing.
10. **"Manifest-backed" is doing work it cannot do.** The phrase appears repeatedly as a proxy for validity. Hash-verifying an archive proves *which bytes you downloaded*. It says nothing about whether they were interpreted, aligned, or aggregated correctly. Provenance is necessary, not sufficient, and the paper repeatedly treats it as sufficient.
11. **Drift flag fires *less* in test than in validation.** The threshold is the validation Q0.90, so a stationary test period would flag ≈ 10 %. Observed: **560/7,199 = 7.8 %**. By the paper's own detector, the test period is **less** drifted than the validation period. This directly undercuts "Under Concept Drift" in the title and is not mentioned.

---

### M13. Three of the four title claims are unsupported by the authors' own results

| Title claim | Status per the submission's own tables |
|---|---|
| **Event-Aware** | Event channel changes MSE by 0.005 (Table VII); score densities near-coincident (Fig. 7); model is **19.6 % worse on the event subset** (Table IX). |
| **Retrieval-Augmented Foundation Model** | The headline model (M09) is **XGBoost — it contains no foundation model.** The Chronos arm is a side branch whose fusion CI spans zero. |
| **Under Concept Drift** | No experiment establishes drift is present; the detector fires **less** often in test than validation; no adaptation is performed (§VI-D-1 concedes drift features "do not adaptively select a forecaster"); the model is **15.2 % worse under drift**. |
| **Explainable** | See M14. |

Further, **"Drift-Aware Forecast and Evidence Head" is a misnomer.** By the authors' own statement the drift signals are diagnostic inputs and log fields; nothing is *aware* of drift in the sense of adapting to it. The module is drift-**annotated**. Names should describe mechanisms.

**Required action.** Retitle to match the evidence. A defensible title: *"Does Event-Context Retrieval Help Short-Horizon PM2.5 Forecasting? An Audited Negative Result for Los Angeles County."* That paper I would be glad to see; this one over-claims what it delivers.

---

### M14. "Explainable" is asserted, never evaluated

**Claim.** The explainability contribution reduces to "we wrote a Parquet file with well-chosen columns."

**Evidence.**
- §VII-E and Table X enumerate stored fields. There is **no faithfulness evaluation** — no deletion/insertion test, no sufficiency/comprehensiveness metric, no ground-truth recovery on synthetic data, no comparison against SHAP (cited in §II-E and not used), no human/expert study, no downstream utility test.
- The method is **underspecified**: "signed local contributions provided by the 24 direct models" (§VI-D-4) does not identify an algorithm. TreeSHAP? `pred_contribs=True`? Gain-weighted path contributions? These are not equivalent and are not interchangeable. As written the explanation layer is not reproducible.
- The proxy that quantifies uncertainty in every one of the 7,199 records is mathematically ill-defined (M7a).
- Selecting the "top 3" feature effects, windows, and event IDs (Table IV) is an arbitrary truncation with no stated justification and no sensitivity analysis.

Logging provenance is genuinely valuable and I encourage the authors to keep it. But it is **traceability**, not explainability, and the paper should say so plainly rather than claim the latter in its title.

---

### M15. Novelty, assessed component by component

| Named contribution | What it actually is | Prior art |
|---|---|---|
| SACB | Careful multi-source alignment + SHA-256 manifests | Data engineering hygiene. Commendable, not novel. |
| LSER embargo | `u_i + H < t − L + 1` | Standard purged/embargoed backtesting; textbook in financial ML and required practice in any time-series retrieval system. |
| LSER hybrid score | Fixed hand-set weights over 4 similarity channels | **Simpler** than the prior art it extends (TimeRAF's learned dual encoder). Weights are untuned and unjustified. |
| DFEH fusion, Eq. (15) | Convex combination of two point forecasts | **Bates & Granger (1969).** Not cited. |
| DFEH drift score | 5-term ad-hoc heuristic, one-sided clip | No comparison to ADWIN / KSWIN / Page-Hinkley / MMD. |
| Evidence record | Structured logging | Engineering, not method. |

**On Eq. (15) specifically — this is important.** It is a well-established result that averaging two imperfect, imperfectly-correlated forecasts almost always lowers MSE through variance reduction, *irrespective of whether the second forecast carries independent signal.* ω = 0.75 places 25 % weight on a forecast whose standalone MSE is 38.083 and yields a 0.233 MSE reduction. **This is textbook forecast combination, not evidence for retrieval.**

**A decisive and nearly free falsification test the authors must run:** replace `Ŷ^R` in Eq. (15) with (a) the climatological mean and (b) M00 persistence, re-select ω on validation, and report the MSE reduction. If either produces a comparable ~0.2 gain — which I expect — then the paper's only positive foundation-model finding is attributable to averaging, and the abstract's framing of M11 vs. M10 must be withdrawn. If retrieval *survives* that test, it becomes the paper's most interesting result and should be promoted. Either outcome is worth having; the experiment is a few lines of code.

---

## Part 3 — Mandatory experiments for any resubmission

Ordered by how much each changes the paper's conclusions per unit of effort.

1. **Rebuild the KB at stride 1/6/24 h** under the identical embargo; re-run M05–M09. *(M1 — without this, nothing about retrieval is interpretable.)*
2. **Placebo fusion test:** Eq. (15) with climatology and with persistence in place of `Ŷ^R`. *(M15 — cheapest decisive test in the paper.)*
3. **δ sweep {0, 0.2, 0.5, 0.8, 1.0} + pool-normalised `s_e` + event-stratified retrieval.** *(M3 — required for any "event-aware" claim.)*
4. **Baselines:** DLinear, PatchTST, LSTM, LightGBM, regularised linear on the same 151 features, hour×month climatology. *(M9.)*
5. **Statistics:** block length ≥ 168 h (or Politis–White stationary bootstrap), B ≥ 2,000, Holm across the pre-registered family, Diebold–Mariano primary. *(M8.)*
6. **Metrics:** AQI-breakpoint exceedance skill; per-horizon skill vs. climatology; CRPS + interval coverage from the Chronos quantiles. *(M10, M9.)*
7. **Site-level target arm** for the 3–5 highest-coverage monitors. *(M6.)*
8. **Extend through 2025**, including the January 2025 episode. *(M11.)*
9. **Fix Eq. (17) and Eq. (19);** report component distributions and inter-component correlations; benchmark the detector against ADWIN/KSWIN/MMD. *(M7.)*
10. **Origin-attrition table** reconciling Table II with the 8–9 % loss; define all audit thresholds; resolve the 3-vs-5 category contradiction; define A00/A01; report or remove M12. *(M12.)*
11. **Feature-count control** separating "events carry no signal" from "sparse columns hurt GBMs." *(M3.)*
12. **Retitle and rewrite** as a negative result. *(M13.)*

Items 1, 2, 3, 5, and 12 are the minimum for the paper to make a defensible claim. Items 4 and 6 are the minimum for journal-level rigour.

---

## Part 4 — Minor and presentation issues

**Mathematical / notational**
- Eq. (13) discussion (line 271): *"The only evaluation is done for equation (13) once set 𝓘_t is applied (which is only when 𝓘_t is eligible)."* **`𝓘_t` is undefined** — the eligible set is `𝒜_t`. The sentence is also not parseable. Rewrite.
- Eq. (14) collides with §VI-C-3. The text says windows are normalised "in both their means and standard deviations, and unit length" — z-score *then* L2? If so, cosine similarity **is** Pearson correlation and should be stated as such. Meanwhile `σ_i` in Eq. (14) must be a *pre*-normalisation scale; the same symbol is used for both. Disambiguate.
- Hard-coded equation numbers "(4)", "(9)", "(11)", "(13)", "(21)", "(22)" appear alongside `\eqref`. These will silently desynchronise on any edit — and "(21)"/"(22)" already look wrong. Use `\eqref` throughout.
- "$\epsilon$" appears in Eqs. (14), (16), (17) with no value given.
- Eq. (16): a 4-component weather term is described but the surrounding text says "five drift components"; the fifth is the event-burst ratio. The `1/4` normalisation inside the weather term is not explained.

**Verified as internally consistent** (credit where due): 23 + 14 + 9 + 39 = 85 ✓; 24 + 24 + 2 + 1 = 51 ✓; 85 + 51 + 6 + 9 = 151 ✓; `G ∈ ℝ^{B×8×24}` ✓; 5 components + mean = 6 ✓; 7,199 × 24 = 172,776 ✓; the R²/MSE/variance relations in Table VI are mutually consistent ✓. The tensor bookkeeping is careful, and the derived variance σ²_y ≈ 42.17 reproduces every reported R² to three decimals.

**Structural redundancy — cut ≈ 1.5 pages**
- The contributions are stated **three times** in near-identical language: §I bullets, §III, §IV bullets. §III ("Critical Gaps") and §IV ("Core Contributions") as separate top-level sections is unusual and here purely repetitive. Merge into §I plus one short paragraph.
- "Random splitting is NOT performed" appears three times (§V, §VI-C, and by implication in the Problem Formulation). Once, in lower case.
- Environment/version details are stated twice verbatim (§VI-E and §VII-G).
- The 2019–2024 study period and the L/H configuration are each restated 5+ times.

**Floats — 21 for a ~13-page paper is excessive**
- 12 tables + 9 figures. Candidates for merger, appendix, or inline prose: Table I (symbols), Table II (audit — merge into III), Table V (model families — merge into VI as a column), Table VII (k-sensitivity — 4 rows, inline), Table X (evidence fields), Table XI (compute).
- Figures 2, 3, and 5 (SACB / LSER / DFEH schematics) are largely subsumed by Figure 1 (pipeline overview). Three additional full-width `figure*` diagrams for modules already shown once is not a good use of the page budget. Keep Figure 1 plus, at most, the LSER embargo timeline (which conveys something Figure 1 cannot).

**Figure defects (specific)**
- **Fig. 4 / Fig. 5:** the legend box **occludes the h = 1–7 region**, hiding the early-horizon behaviour of every model — which is exactly where Figure 4 shows the surprising result that nothing beats persistence at h = 1. Move the legend outside the axes.
- **Fig. 4 / Fig. 5:** 12 series, one panel, colours reused — M00 and M11 are near-identical pinks; M02 and M03 are near-identical olives; M07/M08/M09 are indistinguishable. Not colour-blind safe. Split into 3 panels by family (naive / supervised / TSFM), or plot MSE *differences* against M04.
- **Fig. 7 (right):** x-axis tick labels **overlap into illegibility** ("2024-022024-03..."). Rotate or thin.
- **Fig. 7 (right):** the visible **vertical white gaps** (≈ 2024-07/08, 2024-11/12) are unexplained data gaps in the test period — see M12.1.
- **Fig. 7 (left):** the `random` method is a degenerate spike at `total_score = 0.0`, i.e. random selection is unscored. Plotting it on shared axes with three scored methods compresses everything informative into the right-hand third. Split the panel or drop the random series.
- **Fig. 9:** y-axis is labelled "PM2.5" with **no units** (µg/m³). Legend overlaps the history trace. The chosen case is a low-concentration non-event window — an odd choice of "representative" for an event-aware paper. Show an event case and an exceedance case.
- **No study-area map.** For a paper whose validity turns on monitor topology and one weather station 10.28 km from a county centroid (M6), a map with monitor locations, the ISD station, and county sub-regions is essential.
- No residual/calibration diagnostics, no plot of the target series itself, no event- or drift-subset figure.
- Figures 4–9 are raster PNG in a vector document. Supply PDF/SVG (the four schematics already have SVG companions).

**LaTeX / formatting**
- `\author{Author Name}` — placeholder. Neither properly anonymised nor complete. No affiliation, no `\thanks`, no `\IEEEmembership`.
- `\pagestyle{plain}` and `\thispagestyle{plain}` immediately after `\maketitle` override IEEEtran's headers; remove for a journal submission.
- `\IEEEpeerreviewmaketitle` is absent.
- Table V (`tab:timeraf_comparison`) column widths sum to 0.92\textwidth plus `\tabcolsep`; check for overfull `hbox` in `table*`.
- IEEEtran wants `\bstctlcite` / consistent `\cite` compression; several multi-citation groups would benefit.
- Abstract is **~310 words** — well over the 150–250 typical IEEE limit — and reads as a results dump with six numeric values. Compress to the question, the method, one number, and the caveat.
- Keywords are written as a sentence ending "and explainable forecasting." Use noun phrases, no conjunction.

**Language and style** — the manuscript needs a full professional edit; the following are representative, not exhaustive:
- "chronos-Bolt" (line 572) — inconsistent capitalisation of the model name throughout (`Chronos-Bolt`, `chronos-Bolt`, `Chronos`).
- Inconsistent float references: "figure~\ref", "Figure~\ref", "table~\ref", "Table~\ref", "Fig.~\ref", "equation~\eqref", "Equation~\eqref" — all appear. IEEE style: "Fig. 1", "Table I", "(1)".
- "Equations~\eqref{eq:knowledge_base} formally defines" — plural/singular disagreement.
- Contractions in formal prose: "doesn't", "don't" (§VII-F, §VIII).
- Sentences that do not parse: "These factors means that…" (line 32); "The former is a decoder-only forecasting design (TimesFM), the latter a universal forecasting Transformer (Moirai), the next one…, and the last one…" (line 54 — a five-item "former/latter" construction); "This nonmonotonic behaviour is preserved, not that this means that each named piece adds accuracy." (line 597); "it doesn't bias interpretation of event association to causation" (line 47).
- "The scoring is done very conservatively" (line 47) — "very" plus an unsupported self-assessment.
- Line 618 is a 60-word sentence with a dangling "These models should not be interpreted as…" whose referent is unclear.
- Overuse of "audited", "manifest-backed", "source-audited", "transparent" as substitutes for demonstrated validity — I count 30+ instances. Assertions of rigour are not rigour.

**Citations**
- Eq. (15) requires **Bates & Granger (1969)**, "The combination of forecasts."
- Drift detection cites only SAN (`liu2023san`) for adaptive normalisation. Needs Gama et al. (concept-drift survey), ADWIN (Bifet & Gavaldà), Page-Hinkley, and an MMD two-sample reference.
- The embargo idea needs a purged/embargoed cross-validation citation.
- `zhou2023onefitsall` and `raman2023forecast` carry DOIs `10.52202/075280-1877` and `10.52202/075280-1443`. The `10.52202` prefix (Proceedings.com) is not the canonical identifier for NeurIPS papers; verify or remove — a wrong DOI on a real paper is worse than no DOI.
- All four government-data entries are dated `year = {2026}` with "Accessed: 2026-07-17". Datasets should carry the dataset's own version/vintage; the access date already conveys retrieval time.
- `hochreiter1997lstm`: "Jurgen" → "Jürgen"; `cho2014gru`: "Merrienboer" → "Merriënboer", "Gulcehre" → "Gülçehre"; `karpukhin2020dpr`: "Oguz" → "Oğuz"; `lewis2020rag`: "Kuttler" → "Küttler", "Rocktaschel" → "Rocktäschel". Use `{\"u}` etc. — authors' names should be spelled correctly.
- 43 references is thin for a paper with a five-subsection literature review, and the air-quality domain is under-cited relative to the ML methods (two reviews plus AirFormer). Add operational forecasting practice (CMAQ/WRF-Chem, NOAA/EPA AirNow forecast verification) — the paper claims operational relevance and cites no operational literature.

---

## Part 5 — What the paper does genuinely well

I want this on the record, because it should survive into the resubmission and because it is the reason this is not a straightforward reject-and-forget.

1. **The reproducibility apparatus is better than most accepted papers in this area.** Run IDs, SHA-256 manifests, byte counts, package versions, a public repository, artifact-backed recomputation of every reported number. This should be foregrounded, not buried in §VI-B.
2. **The negative result is reported rather than buried.** §VII-A, §VII-B, the Limitations section, and the Conclusion all state that the proposed method does not beat the baseline. Many authors would have tuned until it did. This is scientifically valuable behaviour and I would like to see it rewarded.
3. **The availability caveat on Storm Events is exactly right.** Recognising that retrospective event labels are not forecast-time information, and refusing to call the results operational, is a distinction many air-quality ML papers get wrong. It should be a *contribution*, not a limitation — a stated protocol for event-availability auditing is publishable in its own right.
4. **The embargo rule is correctly specified.** Eq. (9) is strict and correct. The `191 ≈ 4.18 yr / 192 h` arithmetic checks out against the training span, and the tensor bookkeeping is consistent throughout (see Part 4). The problem is that the *pool* was decimated, not that the *rule* is wrong.
5. **The paper is unusually explicit about what it did not implement** (§IV, §VIII: no learned retriever, no Channel Prompting, no geographic transfer). That honesty is what makes a focused review possible.

---

# Part II — How this review can be defended

This section anticipates author rebuttals and states the reviewer's position and evidence for each. Every criticism above is tied either to a number printed in the submission or to a first-principles argument, deliberately, so that none of it rests on reviewer preference.

## 6.1 Evidentiary basis of the review

The defence rests on four grounds, in descending strength.

| Ground | What it means | Where it is used |
|---|---|---|
| **A. Internal contradiction** | The criticism is derived from the paper's own tables/figures. Not rebuttable without retracting the data. | M2, M4, M12.11, M13 |
| **B. Arithmetic derivation** | Computed from reported numbers; verifiable by the editor in one minute. | M1, M2, M8.1, M12.1–2 |
| **C. Mathematical defect** | A definition is wrong or degenerate on its face. | M7a, M7b |
| **D. Domain/first-principles** | Requires subject expertise but no new experiment. | M5, M6, M11, M15 |

**Nothing in Part 2 is a matter of taste.** No criticism takes the form "I would have preferred a different method."

## 6.2 Anticipated rebuttals and reviewer responses

**R1. "The 192-hour stride is required to prevent leakage between candidates."**
→ Not correct, and the paper's own Eq. (9) shows why. The embargo is a *query–candidate* constraint. Two candidates overlapping each other cannot leak query-future information into the query — leakage is defined relative to the query origin, and Eq. (9) already governs that relation entirely. Candidate–candidate overlap produces *correlated neighbours*, which is a diversity concern with standard remedies (MMR, minimum separation between *selected* items). The authors' own Table V describes TimeRAF's "non-overlapping knowledge windows" as a *leakage control*; that conflation is the origin of the error. **Falsifiable prediction:** rebuild at stride 1 h with Eq. (9) unchanged and the retrieval-only MSE will fall substantially below 36.488, with no leakage introduced. If the authors decline to run it, the null result remains unsupported by their own design.

**R2. "The paper already acknowledges these as limitations."**
→ Acknowledgement is not a substitute for validity, and the mapping is wrong in both directions. The Limitations section names five items; **not one** is M1 (KB scale), M2 (climatology convergence), M3 (inert event channel), M4 (worse on target subsets), M5 (invalid cross-subset MSE), M6 (median target), M7 (two math errors), M8 (anticonservative CIs), M9 (absent baselines), M10 (metric–motivation mismatch), M11 (benign test year), or M12 (unexplained attrition). Meanwhile Limitation 5 (NumPy RNG) is a fixable engineering bug elevated to a scientific caveat. The Limitations section as written establishes the authors' good faith, not the paper's soundness.

**R3. "Missing baselines are a scope decision; we compare against a strong tabular baseline."**
→ Two independent objections. First, **§II reviews ~25 methods and runs none of them** — a five-subsection survey establishes those methods as the relevant comparison set; the authors set that bar themselves. Second, **the paper's central claim is comparative** ("retrieval does not beat context features"). A comparative claim requires that the comparison set be adequate to support it. DLinear and PatchTST are single-file implementations that run in minutes on 48k windows — well inside the reported 1,577 s budget. Cost is not a defence here. Most pointedly: **TimeRAF, the method this paper adapts and names itself after, is not compared against.** A reader cannot distinguish "event-context retrieval doesn't work" from "this simplified retriever doesn't work."

**R4. "MSE differences of 0.5 are practically negligible; the paper says so."**
→ Then the paper must apply that standard symmetrically, and it does not. The M11 − M10 gain of **0.233** is highlighted in the abstract and conclusion; the M07 − M04 harm of **0.487** — twice as large, and the only nominally significant interval in Table VII — is described as retrieval merely "not helping." Either both are negligible (and the abstract's Chronos claim must be withdrawn) or neither is. **This asymmetry is itself a reviewable defect, and it is entirely internal to the submission.**

**R5. "The county median was necessary because no single monitor met the coverage gate."**
→ A constraint can force a design and still invalidate the hypothesis test, and both are true here. The median is a **rank statistic**: a plume elevating 2 of ~15 monitors leaves it unchanged. The event hypothesis is therefore untestable against this target *by construction*, independent of any modelling choice. The correct response to a binding data constraint is to *narrow the claim*, not to keep the claim and accept an incompatible target. Secondary point, unaffected by any rebuttal: a median over a **time-varying monitor roster** is a time-varying target definition, which contaminates the drift measurement in Eqs. (16)–(18). Neither the target's spatial semantics nor its roster stability is discussed anywhere.

**R6. "The event subset has lower MSE — that is an empirical finding we report neutrally."**
→ Reporting it neutrally is not enough when the comparison is invalid. **MSE is not comparable across subsets with different target variance**, so no interpretation of the 9.451-vs-27.520 contrast is licensed. And the *substantive* reading points the opposite way from the paper's motivation: NOAA Storm Events for LA County is dominated by high wind, heavy rain, and flood — all of which ventilate or wet-deposit PM2.5. The "event subset" is a **low-concentration, low-variance subset**. The Introduction motivates the work with wildfire smoke and stagnation; Storm Events records **neither**, and the authors state (§V-D, §VII-F) that the HMS smoke product they cite was not used. The event operationalisation does not match the motivating phenomenon, and the subset result is a symptom of that.

**R7. "The bootstrap is a paired block bootstrap, which handles autocorrelation."**
→ Paired blocking handles it only if the block length exceeds the dependence range, and 24 h does not. Two independent reasons: (i) **construction** — consecutive origins share 167/168 lookback hours and 23/24 target hours; (ii) **physics** — PM2.5 dependence is set by synoptic episodes of 3–7 days plus multi-day stagnation. The effective independent sample over an 11-month test period is on the order of **40–60 episodes**, not 300 blocks, so the intervals are materially too narrow. Note this cuts both ways and I have said so in M8: correction weakens the authors' two favoured intervals *and* removes the M07 − M04 harm finding. The resulting claim — "no retrieval comparison in this study is statistically resolvable at n = 7,199 origins over 11 months" — is cleaner and more defensible than the paper's current mixed narrative. **This is a constructive criticism, not a hostile one.**

**R8. "The uncertainty proxy is explicitly labelled a diagnostic scale, not a calibrated interval."**
→ Labelling does not repair a definition. `sqrt(σ² + MAE²)` adds a squared *mean absolute deviation* to a *variance*; for Gaussian residuals MAE ≈ 0.798σ, so the term understates residual variance by ≈ 36 %. The fix is one character-level change — use validation RMSE — and it is unambiguously correct. Meanwhile the quantity is written into all 7,199 evidence records as the model's uncertainty statement, in a paper whose contribution is the auditability of those records. **An audited artifact containing an ill-defined field undermines the contribution it is meant to demonstrate.**

**R9. "Chronos-Bolt-small was chosen for computational reasons."**
→ The manifest reports a **1,577-second total runtime**. `chronos-bolt-base` on 7,199 origins is a marginal cost against that budget. More fundamentally, this is not primarily a compute objection: **Chronos-Bolt cannot ingest exogenous covariates**, so comparing it against M04 (which uses weather and calendar) is structurally unequal and cannot support any statement about foundation models versus local supervision. And the quantile outputs are discarded in favour of a hand-built proxy — a strictly worse uncertainty estimate than the one the model already provides for free. That last point costs nothing to fix and is not a compute trade-off at all.

**R10. "The framework is the contribution; performance is secondary."**
→ Accepted in principle — frameworks can be contributions. But then the framework must be *shown to work*, i.e. shown to be capable of detecting an effect. It has not been: the retrieval pool is 0.5 % of admissible history (M1), the event channel provably cannot reorder candidates (M3), the target is insensitive to the events retrieved (M6), the test year is the least eventful available (M11), and the drift detector fires *less* in test than in validation (M12.11). **A framework that cannot detect an effect where one might exist has not been validated, and a null result from it is uninformative.** Sensitivity must be demonstrated — ideally on synthetic injected events — before a null carries weight.

**R11. "Reviewer 2 was more positive / this is a novel domain application."**
→ The domain application is welcome and I have said so (Part 5). But the specific claims in the title and abstract must be supported by the specific numbers in Tables VI–IX, and by the authors' own reporting they are not. My recommendation is not "this work is uninteresting" — it is "**this work is mis-framed, and the experiment cannot yet support any framing.**" I have specified twelve experiments (Part 3) that would resolve this, five of which are cheap, and I have committed below to what would change my recommendation.

## 6.3 Reviewer conduct — why this review is fair

A tough review is defensible only if it is also fair. Specifically:

1. **Strengths are stated substantively, not perfunctorily** (Part 5, five items, with the recommendation that they survive into the resubmission).
2. **Every criticism carries a required action.** No complaint is left as an unactionable objection.
3. **Criticisms are graded**, with major concerns separated from minor and presentation issues, so the authors can triage.
4. **The reviewer's own arithmetic is shown** (σ²_y ≈ 42.17; 191 ≈ 4.18 yr / 192 h; R²(M05) = −0.005; effective sample 40–60), so the authors and editor can check it rather than take it on authority. Part 4 explicitly confirms the dimensional bookkeeping the authors got *right*.
5. **The review corrects itself where the paper is right.** The `191`-window count initially appeared inconsistent with 70 % of 48,332 windows at a 192 h stride (≈ 176); on checking against the *training time span* (≈ 4.18 yr), 191 is correct, and no split violation is alleged. This is stated in Part 5 rather than suppressed.
6. **The strongest criticism (M15/R7) points in the authors' favour** in part: correcting the block length dissolves the M07 − M04 harm finding, and I say so. A review that only ever tightens the screws is a review with a thesis rather than an analysis.
7. **The recommendation is resubmission, not rejection-with-prejudice**, and the reframing offered (audited negative result) is a genuine publication route with a proposed title.

## 6.4 What would change the recommendation

Stated in advance, so the authors know the bar and the editor can hold the reviewer to it.

- **Accept-track (major revision):** Part 3 items 1, 2, 3, 5, 12 completed, with the paper retitled as a negative/protocol result and M4's subset reversals promoted to headline findings — **even if every retrieval result remains null.** A well-powered, well-instrumented null on event-context retrieval for urban PM2.5, with a public audit protocol, is a genuine contribution and I would advocate for it.
- **Strong accept:** the above, plus item 1 showing that retrieval gain *appears* at realistic KB scale (making the original null a scale artifact and the paper a positive result), **or** item 2 showing the M11 − M10 gain survives the climatology/persistence placebo (making retrieval the operative mechanism rather than forecast averaging).
- **Reject-track maintained:** cosmetic revision, the title and abstract unchanged, the subset reversals still undiscussed, or the KB left at 191 windows.

---

## Appendix A — Numbers the reviewer derived from the submission

Recorded so that every quantitative claim above is independently checkable.

| Derived quantity | Value | Derivation |
|---|---|---|
| Target variance σ²_y | 42.17 | 26.185 / (1 − 0.379), from M04 |
| Target σ_y | 6.49 µg/m³ | √42.17 |
| Consistency check, persistence | R² = −0.130 ✓ | 1 − 47.649/42.17 = −0.130 |
| **R²(M05 random retrieval)** | **−0.005** | 1 − 42.367/42.17 → exactly a mean predictor |
| R²(M06 cosine, k=8) | 0.097 ✓ | 1 − 38.083/42.17, matches Table VI |
| R²(M06 cosine, k=16) | 0.135 | 1 − 36.488/42.17 |
| M04 skill vs. persistence | 45.0 % | 1 − 26.185/47.649 |
| M09 vs. M04, all origins | **+2.01 % worse** | (26.712 − 26.185)/26.185 |
| M09 vs. M04, **event** subset | **+19.6 % worse** | (11.301 − 9.451)/9.451 |
| M09 vs. M04, **drift** subset | **+15.2 % worse** | (31.244 − 27.110)/27.110 |
| M04 drift penalty | +3.8 % | (27.110 − 26.107)/26.107 |
| M09 drift penalty | +18.7 % | (31.244 − 26.330)/26.330 |
| M11 vs. M10 | −0.80 % | (28.709 − 28.941)/28.941 ✓ matches text |
| Drift flag rate, test | 7.78 % | 560/7,199 vs. 10 % nominal at val Q0.90 |
| Event origin rate, test | 7.39 % | 532/7,199 |
| KB windows vs. admissible | **0.52 %** | 191 / (4.18 yr × 8,766 h) ≈ 191/36,640 |
| KB span consistency | ✓ | 191 × 192 h = 36,672 h = 4.18 yr ≈ training span |
| Window yield | 91.9 % | 48,332 / ≈52,600 study hours |
| Test-period origin yield | 90.9 % | 7,199 / ≈7,920 h (Fig. 8 span ≈ 330 d) |
| Test split arithmetic gap | 51 origins | 0.15 × 48,332 = 7,249.8 vs. 7,199 |
| Effective independent test sample | ≈ 40–60 | 11 months / (3–7 d synoptic episode) |
| Event base rate | 5.3 % | 117 event days / 2,192 days |
| Event feature density | 39 dims from 204 records | Table III, Eq. (7) |
| Test period (from figure axes) | 2024-02 → 2024-12 | Figs. 7, 8 — not stated in text |
| Dimensional checks | all ✓ | 23+14+9+39=85; 24+24+2+1=51; 85+51+6+9=151 |

## Appendix B — Confidential comments to the editor

The submission is a competently executed, unusually transparent engineering study reporting a null result, currently presented as a positive methodological advance. Three of four title claims are contradicted by the authors' own tables.

My primary concern is not the null result — it is that **the experiment lacks demonstrated sensitivity**: the retrieval pool is 0.5 % of admissible history, the event similarity channel provably cannot reorder candidates at a 5 % event base rate, the county-median target is a rank statistic insensitive to the localised events being retrieved, and the test year is the least eventful in the available record while the most extreme episode in modern LA history sits one month past the cutoff. A null from that configuration is uninformative in either direction.

Against that, the reproducibility apparatus is stronger than in most accepted papers in this area, and the authors' willingness to publish a negative result should be encouraged rather than penalised. I would support a **reject with explicit invitation to resubmit** as an audited negative-result / evaluation-protocol paper. Five of my twelve required experiments are cheap (items 1, 2, 3, 5, 12), and I have stated in §6.4 that I would advocate for acceptance on that basis **even if every retrieval result remains null**.

Two items warrant editorial attention independent of the technical review: the author block is the placeholder `\author{Author Name}` (neither anonymised nor complete), and §VI-D excludes a model "M12" without stating what it was or what it scored, which should be resolved as a reporting-completeness matter.
