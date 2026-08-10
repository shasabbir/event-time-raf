from __future__ import annotations

from pathlib import Path, PurePosixPath
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "event_timeraf_kaggle_upload.zip"

entries = [
    ROOT / "configs" / "default.yaml",
    ROOT / "requirements.txt",
    ROOT / "requirements-optional.txt",
    ROOT / "README.md",
    ROOT / "notebooks" / "00_prepare_official_noaa_storm_cache.ipynb",
    ROOT / "notebooks" / "01_event_timeraf_kaggle_pipeline.ipynb",
    ROOT / "notebooks" / "02_results_and_figures.ipynb",
]
entries.extend(sorted((ROOT / "src" / "event_timeraf").glob("*.py")))
entries.extend(sorted((ROOT / "tests").glob("*.py")))

DESTINATION.unlink(missing_ok=True)
with zipfile.ZipFile(DESTINATION, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as bundle:
    for source in entries:
        if not source.exists():
            raise FileNotFoundError(source)
        archive_name = PurePosixPath(source.relative_to(ROOT).as_posix()).as_posix()
        bundle.write(source, archive_name)

with zipfile.ZipFile(DESTINATION) as bundle:
    names = bundle.namelist()
    invalid = [name for name in names if "\\" in name or name.startswith("/") or "../" in name]
    if invalid:
        raise AssertionError(f"Invalid archive member names: {invalid[:3]}")

print(f"{DESTINATION.name}: {len(entries)} files, {DESTINATION.stat().st_size / 1_000_000:.2f} MB")
