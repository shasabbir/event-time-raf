from __future__ import annotations

from pathlib import Path, PurePosixPath
import zipfile


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "event_timeraf_kaggle_upload.zip"
DIRECTORIES = ("configs", "src", "tests")
NOTEBOOK_FILES = (
    "00_prepare_official_noaa_storm_cache.ipynb",
    "01_event_timeraf_kaggle_pipeline.ipynb",
    "02_results_and_figures.ipynb",
)
ROOT_FILES = (
    "README.md",
    "requirements.txt",
    "requirements-optional.txt",
    "structured_plan.md",
    "plan.md",
)


def include(path: Path) -> bool:
    return path.is_file() and not any(
        part in {"__pycache__", ".ipynb_checkpoints"} for part in path.parts
    ) and path.suffix.lower() not in {".pyc", ".pyo"}


DESTINATION.unlink(missing_ok=True)
with zipfile.ZipFile(
    DESTINATION,
    "w",
    compression=zipfile.ZIP_DEFLATED,
    compresslevel=9,
) as bundle:
    for directory in DIRECTORIES:
        for path in sorted((ROOT / directory).rglob("*")):
            if include(path):
                archive_name = PurePosixPath(path.relative_to(ROOT).as_posix()).as_posix()
                bundle.write(path, archive_name)
    for name in NOTEBOOK_FILES:
        path = ROOT / "notebooks" / name
        bundle.write(path, (PurePosixPath("notebooks") / name).as_posix())
    for name in ROOT_FILES:
        path = ROOT / name
        bundle.write(path, PurePosixPath(name).as_posix())

with zipfile.ZipFile(DESTINATION) as bundle:
    names = bundle.namelist()
    required = {"configs/default.yaml", "src/event_timeraf/config.py",
                "notebooks/01_event_timeraf_kaggle_pipeline.ipynb"}
    missing = required - set(names)
    invalid = [name for name in names if "\\" in name or name.startswith("/") or ".." in PurePosixPath(name).parts]
    if missing:
        raise AssertionError(f"Missing required package entries: {sorted(missing)}")
    if invalid:
        raise AssertionError(f"Invalid ZIP paths: {invalid[:3]}")

print(f"{DESTINATION} ({DESTINATION.stat().st_size / 1_000_000:.2f} MB, {len(names)} files)")
