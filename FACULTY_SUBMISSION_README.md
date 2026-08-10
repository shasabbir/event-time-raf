# Event-TimeRAF Faculty Submission

## Deliverables

- Executed notebook: `notebooks/03_paper_claim_verification.ipynb`
- Corrected paper: `paper/main_humanized.pdf`
- Overleaf source: `event_timeraf_verified_overleaf.zip`
- AI verification log: `paper/verification_log.pdf`
- Presentation notes: `presentation/presenter_notes.md`
- Immutable evidence: `event_timeraf_final_run_20260810T103436161252Z.zip`
  (kept as a separate 159 MB file)

## Live notebook run

Place `event_timeraf_final_run_20260810T103436161252Z.zip` in the repository root, open
`notebooks/03_paper_claim_verification.ipynb`, restart the kernel, and run all
cells. The verified local runtime is approximately 81 seconds on the development
machine. The final cell
must print:

```text
FINAL VERIFICATION STATUS: PASS WITH EXPLICIT LIMITATIONS
```

The notebook checks all 54 manifest entries, reconstructs all model metrics from
172,776 test points, reruns the primary 2,000-resample comparisons, displays all
sensitivity tables, regenerates all nine figures, and applies source-level claim
gates. The full training experiment remains in
`notebooks/01_event_timeraf_kaggle_pipeline.ipynb`.

## Kaggle

Upload the repository/project ZIP and the timestamped final-run ZIP as private
Kaggle datasets. Open the verification notebook and run all cells with Internet
disabled. If automatic discovery fails, set these variables in the setup cell:

```python
import os
os.environ["PROJECT_ROOT_OVERRIDE"] = "/kaggle/working/event_timeraf"
os.environ["FINAL_RUN_ZIP_OVERRIDE"] = "/kaggle/input/<dataset>/event_timeraf_final_run_20260810T103436161252Z.zip"
```

## Overleaf

Upload `event_timeraf_verified_overleaf.zip` as a new project. The ZIP has
`main.tex` at its root and uses forward-slash archive paths. Compile with
pdfLaTeX/BibTeX. Confirm the author affiliation and add the required
institutional email before creating and submitting the Overleaf share link.

## Presentation position

The existing PPTX/PDF predates the final Aug. 10 run and is deliberately not
included in the faculty package. Regenerate and visually verify the deck from
the synchronized source and notes before submission; do not present the older
deck as final-run evidence.

The strongest verified result is M04 context XGBoost, not full M09:

- M04: MSE 26.185, MAE 3.125, RMSE 5.117, R-squared 0.379
- M09: MSE 26.734
- M10 frozen Chronos-Bolt: MSE 28.941
- M11 Chronos + retrieval: MSE 28.733, with a paired interval crossing zero
- C10 Chronos + climatology: MSE 27.450, significantly better than M11

Do not claim that retrieval or the event channel improves the strongest model.
Dense retrieval helps only the retrieval-only path, and the event-weight
sensitivity worsens as event influence increases. Event findings are
retrospective because NOAA Storm Events does not expose machine-readable
publication timestamps.
