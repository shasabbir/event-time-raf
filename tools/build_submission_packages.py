from __future__ import annotations

from pathlib import Path, PurePosixPath
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
RUN_ID = "20260827T043457543402Z"


def write_zip(destination: Path, entries: list[tuple[Path, str]]) -> None:
    destination.unlink(missing_ok=True)
    with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
        for source, archive_name in entries:
            if not source.exists():
                raise FileNotFoundError(source)
            bundle.write(source, PurePosixPath(archive_name).as_posix())
    with zipfile.ZipFile(destination) as bundle:
        invalid = [name for name in bundle.namelist() if "\\" in name]
        if invalid:
            raise AssertionError(f"Backslash archive paths found: {invalid[:3]}")


overleaf_entries = [
    (PAPER / "main.tex", "main.tex"),
    (PAPER / "references.bib", "references.bib"),
]
ieee_access_assets = [
    "ieeeaccess.cls",
    "spotcolor.sty",
    "IEEEtran.bst",
    "logo.png",
    "notaglinelogo.png",
    "bullet.png",
]
ieee_access_assets.extend(path.name for path in sorted(PAPER.glob("t1-*")))
ieee_access_assets.extend(path.name for path in sorted(PAPER.glob("t1*.fd")))
overleaf_entries.extend((PAPER / name, name) for name in ieee_access_assets)
referenced_figures = [
    "event_timeraf_pipeline_overview.pdf",
    "source_audited_context_builder.pdf",
    "leakage_safe_event_context_retriever.pdf",
    "drift_aware_forecast_evidence_head.pdf",
    "mse_by_horizon.pdf",
    "mae_by_horizon.pdf",
    "drift_scores.png",
    "forecast_case.png",
    "forecast_event_case.png",
]
overleaf_entries.extend(
    (PAPER / "figures" / name, f"figures/{name}") for name in referenced_figures
)
write_zip(ROOT / "event_timeraf_verified_overleaf.zip", overleaf_entries)

overleaf_zip = ROOT / "event_timeraf_verified_overleaf.zip"
print(f"{overleaf_zip.name}: {overleaf_zip.stat().st_size / 1_000_000:.2f} MB")

presentation = ROOT / "presentation" / "TRACE-RAF_Verified_Presentation.pptx"
presentation_pdf = ROOT / "presentation" / "TRACE-RAF_Verified_Presentation.pdf"
faculty_zip = ROOT / "event_timeraf_faculty_submission.zip"
faculty_zip.unlink(missing_ok=True)

if presentation.exists() and presentation_pdf.exists():
    faculty_entries = [
        (ROOT / "FACULTY_SUBMISSION_README.md", "FACULTY_SUBMISSION_README.md"),
        (ROOT / "notebooks" / "01_event_timeraf_kaggle_pipeline.ipynb", "notebooks/01_event_timeraf_kaggle_pipeline.ipynb"),
        (ROOT / "notebooks" / "02_results_and_figures.ipynb", "notebooks/02_results_and_figures.ipynb"),
        (ROOT / "notebooks" / "03_paper_claim_verification.ipynb", "notebooks/03_paper_claim_verification.ipynb"),
        (PAPER / "main.pdf", "paper/TRACE-RAF_corrected.pdf"),
        (PAPER / "verification_log.pdf", "paper/AI_verification_log.pdf"),
        (PAPER / "CHANGELOG.md", "paper/CHANGELOG.md"),
        (PAPER / "submission_audit.md", "paper/submission_audit.md"),
        (PAPER / "ACTION_REQUIRED_FROM_ME.md", "paper/ACTION_REQUIRED_FROM_ME.md"),
        (presentation, "presentation/TRACE-RAF_Verified_Presentation.pptx"),
        (presentation_pdf, "presentation/TRACE-RAF_Verified_Presentation.pdf"),
        (ROOT / "presentation" / "presenter_notes.md", "presentation/presenter_notes.md"),
        (overleaf_zip, "event_timeraf_verified_overleaf.zip"),
    ]
    write_zip(faculty_zip, faculty_entries)
    print(f"{faculty_zip.name}: {faculty_zip.stat().st_size / 1_000_000:.2f} MB")
else:
    print(
        "Faculty package not built: synchronized TRACE-RAF PPTX/PDF are missing. "
        f"Rebuild the presentation for run {RUN_ID}; the legacy Event-TimeRAF deck is rejected."
    )
