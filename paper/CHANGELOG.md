# Verified Manuscript Changelog

## Claim verification

- Added `notebooks/03_paper_claim_verification.ipynb`, an executed top-to-bottom
  companion that verifies all 69 manifest hashes, reconstructs the test data and
  retrieval knowledge base, reruns M00--M11, recomputes metrics and ablations,
  regenerates all nine manuscript figures, and ends with a claim gate.
- Corrected the SACB origin vector from 86 to 85 features and the M09
  per-horizon input from 152 to 151. The saved composition is 23 PM2.5, 14
  weather, nine calendar, and 39 event features; M09 adds 51 retrieval, six
  drift, and nine future-calendar fields.
- Removed M12 from the abstract, results, tables, discussion, and conclusion.
  M12 is absent from the immutable manifest and appears only in post-manifest
  edited artifacts.
- Replaced the generic drift formulation with the implemented five raw
  statistics, training median/MAD scaling, clipping, and validation 0.90
  quantile threshold.
- Added the cross-version random-retrieval limitation. M05 and A01 are
  reconstructed from checksum-verified evidence because NumPy 2.4.1 does not
  reproduce the NumPy 2.0.2 selection stream.

## Manuscript

- Synchronized the abstract, contributions, methodology, experimental setup,
  results, limitations, and conclusion with the verified model ladder.
- Retained the verified headline: M04 MSE 26.185, MAE 3.125, RMSE 5.117, and
  R-squared 0.379. M09 records MSE 26.712. M11 changes Chronos-Bolt MSE from
  28.941 to 28.709, with a paired interval that crosses zero.
- Added the direct M09-versus-A00 event-context ablation and kept all reported
  confidence intervals tied to real prediction-level runs.
- Regenerated the five empirical PNG figures from the verification notebook and
  the four methodology figures from the shared vector builder.
- Compiled `main.pdf` with IEEE journal formatting and no undefined references
  or citations.

## Submission support

- Added `verification_log.tex` and the one-page `verification_log.pdf`.
- Added `presentation/Event-TimeRAF_Verified_Presentation.pptx`, its PDF export,
  source builder, and presenter Q&A notes.
- Added a faculty-submission README and Overleaf-ready source ZIP. An actual
  Overleaf share link must be created from the user's account after upload.

## Claim scope

No learned TimeRAF dual encoder, internal Channel Prompting, strict real-time
event availability, external geographic validation, or hardware-normalized
efficiency is claimed. SACB, LSER, and DFEH name behavior already present in the
repository; the names and tensor presentation are manuscript enrichments, not
unimplemented model modules.
