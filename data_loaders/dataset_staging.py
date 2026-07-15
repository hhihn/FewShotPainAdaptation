"""Utilities for staging predefined split datasets from local tar archives."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path


SUPPORTED_ARCHIVE_DATASETS = ("biovid_part_a", "senseemotion")


def _normalize_dataset_source(dataset_source: str) -> str:
    normalized = str(dataset_source).strip().lower()
    if normalized not in SUPPORTED_ARCHIVE_DATASETS:
        raise ValueError(
            "dataset_source must be one of: " + ", ".join(SUPPORTED_ARCHIVE_DATASETS)
        )
    return normalized


def archive_candidate_paths(
    dataset_source: str,
    drive_data_dir: Path,
    *,
    mydrive_root: Path | None = None,
) -> tuple[Path, ...]:
    """Return likely archive paths for one dataset under Google Drive."""
    dataset_source = _normalize_dataset_source(dataset_source)
    drive_data_dir = Path(drive_data_dir)
    if mydrive_root is None:
        mydrive_root = drive_data_dir.parent
    else:
        mydrive_root = Path(mydrive_root)

    if dataset_source == "biovid_part_a":
        names = (
            "BioVid.tar.gz",
            "biovid_parta.tar.gz",
            "BioVid.tgz",
            "biovid_parta.tgz",
        )
    else:
        names = (
            "sense_emotion.tar.gz",
            "SenseEmotion.tar.gz",
            "senseemotion.tar.gz",
            "sense_emotion.tgz",
            "SenseEmotion.tgz",
            "senseemotion.tgz",
        )
    return tuple(
        [drive_data_dir / name for name in names]
        + [mydrive_root / name for name in names]
        + [mydrive_root / "data" / name for name in names]
    )


def has_predefined_split_root(path: Path) -> bool:
    """Return whether a path contains Train and Test directories."""
    path = Path(path)
    return (path / "Train").is_dir() and (path / "Test").is_dir()


def resolve_predefined_dataset_root(
    dataset_source: str,
    data_dir: Path,
    *,
    staged_root: Path | None = None,
) -> Path | None:
    """Resolve an extracted predefined split dataset root if present."""
    dataset_source = _normalize_dataset_source(dataset_source)
    data_dir = Path(data_dir)
    candidates: list[Path] = []
    if staged_root is not None:
        candidates.append(Path(staged_root))
    if dataset_source == "biovid_part_a":
        candidates.extend([data_dir / "BioVid" / "PartA", data_dir / "PartA", data_dir])
    else:
        candidates.extend([data_dir / "SenseEmotion", data_dir / "senseemotion", data_dir])

    for candidate in candidates:
        if has_predefined_split_root(candidate):
            return candidate
    return None


def safe_extract_tar(tar: tarfile.TarFile, target_dir: Path) -> None:
    """Extract a tar archive after rejecting members that escape target_dir."""
    target_dir = Path(target_dir).resolve()
    for member in tar.getmembers():
        member_path = (target_dir / member.name).resolve()
        if member_path != target_dir and target_dir not in member_path.parents:
            raise RuntimeError(f"Unsafe tar member path: {member.name}")
    tar.extractall(target_dir)


def remove_archive_metadata_files(root: Path) -> int:
    """Remove common macOS archive metadata files below root."""
    removed = 0
    for pattern in ("._*", ".DS_Store"):
        for path in Path(root).rglob(pattern):
            if path.is_file():
                path.unlink()
                removed += 1
    return removed


def find_dataset_archive(
    dataset_source: str,
    drive_data_dir: Path,
    *,
    mydrive_root: Path | None = None,
    extra_candidates: tuple[Path, ...] = (),
) -> Path | None:
    """Return the first existing archive candidate for one dataset."""
    candidates = tuple(extra_candidates) + archive_candidate_paths(
        dataset_source,
        drive_data_dir,
        mydrive_root=mydrive_root,
    )
    return next((Path(path) for path in candidates if Path(path).is_file()), None)


def stage_predefined_dataset_from_archive(
    dataset_source: str,
    *,
    drive_data_dir: Path,
    local_data_dir: Path,
    local_archive_dir: Path | None = None,
    mydrive_root: Path | None = None,
    extra_archive_candidates: tuple[Path, ...] = (),
) -> Path:
    """Copy and extract a predefined split dataset archive when needed.

    Returns the local directory containing the dataset's Train/Test folders.
    """
    dataset_source = _normalize_dataset_source(dataset_source)
    local_data_dir = Path(local_data_dir)
    existing_root = resolve_predefined_dataset_root(dataset_source, local_data_dir)
    if existing_root is not None:
        return existing_root

    archive_path = find_dataset_archive(
        dataset_source,
        drive_data_dir,
        mydrive_root=mydrive_root,
        extra_candidates=extra_archive_candidates,
    )
    if archive_path is None:
        searched = ", ".join(
            str(path)
            for path in (
                tuple(extra_archive_candidates)
                + archive_candidate_paths(
                    dataset_source,
                    drive_data_dir,
                    mydrive_root=mydrive_root,
                )
            )
        )
        raise FileNotFoundError(
            f"Could not find {dataset_source} tar archive. Searched: {searched}"
        )

    local_data_dir.mkdir(parents=True, exist_ok=True)
    if local_archive_dir is None:
        local_archive_dir = local_data_dir.parent
    else:
        local_archive_dir = Path(local_archive_dir)
    local_archive_dir.mkdir(parents=True, exist_ok=True)
    local_archive = local_archive_dir / archive_path.name
    if (
        not local_archive.exists()
        or local_archive.stat().st_size != archive_path.stat().st_size
    ):
        shutil.copy2(archive_path, local_archive)

    with tarfile.open(local_archive, "r:*") as tar:
        safe_extract_tar(tar, local_data_dir)
    remove_archive_metadata_files(local_data_dir)

    dataset_root = resolve_predefined_dataset_root(dataset_source, local_data_dir)
    if dataset_root is None:
        raise FileNotFoundError(
            f"Archive {archive_path} did not produce Train/Test folders under "
            f"{local_data_dir}"
        )
    return dataset_root
