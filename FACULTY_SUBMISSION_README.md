# Event-TimeRAF Faculty Submission

## Deliverables

- Executed notebook: `notebooks/03_paper_claim_verification.ipynb`
- Corrected paper: `paper/main.pdf`
- Overleaf source: `event_timeraf_verified_overleaf.zip`
- AI verification log: `paper/verification_log.pdf`
- Presentation: `presentation/Event-TimeRAF_Verified_Presentation.pptx`
- Presentation notes: `presentation/presenter_notes.md`
- Immutable evidence: `event_timeraf_final_run.zip` (kept as a separate file
  because it is approximately 492 MB)

## Live notebook run

Place `event_timeraf_final_run.zip` in the repository root, open
`notebooks/03_paper_claim_verification.ipynb`, restart the kernel, and run all
cells. The verified local runtime is approximately 142 seconds. The final cell
must print:

```text
FINAL VERIFICATION STATUS: PASS WITH EXPLICIT LIMITATIONS
```

The notebook checks all 43 manifest hashes, reconstructs the saved dataset and
retrieval evidence, reruns M00--M11, recomputes every manuscript metric and
ablation, regenerates all nine figures, and applies source-level claim gates.
It uses archived trained models so that a live run is short. The full original
training experiment remains in `notebooks/01_event_timeraf_kaggle_pipeline.ipynb`.

## Kaggle

Upload the repository/project ZIP and `event_timeraf_final_run.zip` as private
Kaggle datasets. Open the verification notebook and run all cells with Internet
disabled. If automatic discovery fails, set these variables in the setup cell:

```python
import os
os.environ["PROJECT_ROOT_OVERRIDE"] = "/kaggle/working/event_timeraf"
os.environ["FINAL_RUN_ZIP_OVERRIDE"] = "/kaggle/input/<dataset>/event_timeraf_final_run.zip"
```

## Overleaf

Upload `event_timeraf_verified_overleaf.zip` as a new project. The ZIP has
`main.tex` at its root and uses forward-slash archive paths. Compile with
pdfLaTeX/BibTeX. Confirm the author affiliation and add the required
institutional email before creating and submitting the Overleaf share link.

## Presentation position

The strongest verified result is M04 context XGBoost, not full M09:

- M04: MSE 26.185, MAE 3.125, RMSE 5.117, R-squared 0.379
- M09: MSE 26.712
- M10 frozen Chronos-Bolt: MSE 28.941
- M11 Chronos + retrieval: MSE 28.709, with a paired interval crossing zero

Do not report M12 as a final result. It was created after the immutable run and
is excluded from the corrected paper. Event findings are retrospective because
NOAA Storm Events does not expose machine-readable publication timestamps.
