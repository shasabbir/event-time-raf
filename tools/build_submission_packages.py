from __future__ import annotations

from pathlib import Path, PurePosixPath
import zipfile


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


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
    # The humanized manuscript is the submission version; Overleaf still expects
    # the project entry point to be named main.tex.
    (PAPER / "main_humanized.tex", "main.tex"),
    (PAPER / "references.bib", "references.bib"),
]
overleaf_entries.extend(
    (path, f"figures/{path.name}")
    for path in sorted((PAPER / "figures").iterdir())
    if path.suffix.lower() in {".pdf", ".png", ".svg"}
)
write_zip(ROOT / "event_timeraf_verified_overleaf.zip", overleaf_entries)

faculty_entries = [
    (ROOT / "FACULTY_SUBMISSION_README.md", "FACULTY_SUBMISSION_README.md"),
    (ROOT / "notebooks" / "03_paper_claim_verification.ipynb", "notebooks/03_paper_claim_verification.ipynb"),
    (PAPER / "main_humanized.pdf", "paper/Event-TimeRAF_corrected.pdf"),
    (PAPER / "main_humanized.tex", "paper/Event-TimeRAF_corrected.tex"),
    (PAPER / "verification_log.pdf", "paper/AI_verification_log.pdf"),
    (PAPER / "CHANGELOG.md", "paper/CHANGELOG.md"),
    (PAPER / "submission_audit.md", "paper/submission_audit.md"),
    (PAPER / "ACTION_REQUIRED_FROM_ME.md", "paper/ACTION_REQUIRED_FROM_ME.md"),
    (ROOT / "presentation" / "presenter_notes.md", "presentation/presenter_notes.md"),
    (ROOT / "event_timeraf_verified_overleaf.zip", "event_timeraf_verified_overleaf.zip"),
]
write_zip(ROOT / "event_timeraf_faculty_submission.zip", faculty_entries)

for destination in (
    ROOT / "event_timeraf_verified_overleaf.zip",
    ROOT / "event_timeraf_faculty_submission.zip",
):
    print(f"{destination.name}: {destination.stat().st_size / 1_000_000:.2f} MB")
