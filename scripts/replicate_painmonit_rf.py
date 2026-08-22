"""Replicate the PainMonit paper's random-forest baseline on our copy of the data.

The published numbers we are trying to reproduce come from the PainMonit dataset
paper (Gouverneur et al., *Scientific Data* 2024) and its code at
https://github.com/gouverneurp/XAIinPainResearch. For the PMED heat dataset,
leave-one-subject-out random forests on 10 s stimulus windows reach:

    task B vs. P4 (raw heater classes 0 vs. 5)
        Eda_RB  91.93 %      Eda_E4  73.48 %      Ecg  63.57 %
        Emg     52.76 %      Resp    56.98 %      late fusion  91.34 %

This script does not reimplement their feature extraction. It imports
``hcf.py`` from a clone of their repository and calls it unchanged, so the
features are theirs. Only the thin evaluation driver is reproduced here, mirroring
``scripts/classifier.py::rf`` and ``scripts/evaluation.py::loso_cross_validation``
/ ``five_loso``:

  * ``RandomForestRegressor(n_estimators, max_depth=None, min_samples_split=2)``
    fitted on one-hot targets, with the prediction taken as the argmax. Their
    ``rf`` really is a regressor, not a classifier.
  * Leave-one-subject-out over every subject, repeated ``--runs`` times, and the
    reported figure is the mean over runs of the per-run mean fold accuracy.
  * No feature normalisation, matching their headline configuration.

Deviation worth knowing: their ``hcf.py`` computes features for all six sensors
in one pass. Feature extraction is the slow part, so ``--sensors`` limits it to
the ones being evaluated. Their random forest selects feature columns by sensor
prefix anyway, so restricting extraction changes the runtime and not the result.

Usage
-----
    git clone https://github.com/gouverneurp/XAIinPainResearch.git

    python scripts/replicate_painmonit_rf.py \
        --xai-repo ../XAIinPainResearch \
        --data-dir data \
        --sensors Eda_RB \
        --features-csv results/painmonit_hcf_Eda_RB.csv

Requires ``cvxopt`` and ``tqdm`` (used by their EDA feature code):
``pip install cvxopt tqdm``.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# Raw channel order of X.npy / X_pre.npy, from the authors' PMED/config.py.
PAINMONIT_SENSORS = ("Bvp", "Eda_E4", "Resp", "Eda_RB", "Ecg", "Emg")
SAMPLING_RATE_HZ = 250

# Published LOSO accuracies for raw heater classes 0 vs. 5, for reporting only.
PUBLISHED_B_VS_P4 = {
    "Eda_RB": 91.93,
    "Eda_E4": 73.48,
    "Ecg": 63.57,
    "Emg": 52.76,
    "Resp": 56.98,
}


@contextlib.contextmanager
def working_directory(path: Path):
    """Run a block with ``path`` as the process working directory.

    Their ``hcf.py`` resolves cache paths relative to the current directory, so
    it has to be imported and called from inside its own checkout.
    """
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def load_arrays(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load the PainMonit arrays, accepting either our or the authors' naming.

    Returns:
        X shaped [n, 2500, 6, 1], one-hot heater labels, and subject ids.
    """
    # (data, labels, subjects) naming schemes: ours, the authors', and our mock set.
    naming = (
        ("X_pre.npy", "y_heater.npy", "subjects.npy"),
        ("X.npy", "y_heater.npy", "subjects.npy"),
        ("X_pre_mock.npy", "y_heater_mock.npy", "subjects_mock.npy"),
    )
    triple = next(
        (names for names in naming if all((data_dir / name).exists() for name in names)),
        None,
    )
    if triple is None:
        raise FileNotFoundError(
            f"No complete array set in {data_dir}. Expected one of: "
            + "; ".join(", ".join(names) for names in naming)
        )

    x_path = data_dir / triple[0]
    X = np.load(x_path)
    y = np.load(data_dir / triple[1])
    subjects = np.load(data_dir / triple[2])

    if X.ndim == 3:
        X = X[..., np.newaxis]
    if X.ndim != 4:
        raise ValueError(f"Expected X shaped [n, time, sensors, 1], got {X.shape}")
    if X.shape[2] != len(PAINMONIT_SENSORS):
        raise ValueError(
            f"Expected {len(PAINMONIT_SENSORS)} raw channels {PAINMONIT_SENSORS}, "
            f"got {X.shape[2]}. Channel order matters: index 3 is Eda_RB."
        )
    if len(X) != len(y) or len(X) != len(subjects):
        raise ValueError("X, y_heater and subjects disagree on sample count")

    print(f"Loaded {x_path.name}: X={X.shape}, y={y.shape}, subjects={subjects.shape}")
    print(f"  {len(np.unique(subjects))} subjects, "
          f"raw classes {sorted(set(np.argmax(y, axis=1).tolist()))}")
    return X, y, subjects


def build_features(
    X: np.ndarray,
    xai_repo: Path,
    sensors: Sequence[str],
    cache_csv: Path | None,
) -> pd.DataFrame:
    """Extract hand-crafted features using the authors' unmodified ``hcf.py``.

    Args:
        X: Raw data shaped [n, time, sensors, 1].
        xai_repo: Path to a clone of XAIinPainResearch.
        sensors: Sensor names to extract features for.
        cache_csv: Optional path to read/write the feature table.
    """
    if cache_csv is not None and cache_csv.exists():
        features = pd.read_csv(cache_csv, sep=";", decimal=",")
        print(f"Loaded cached features from {cache_csv}: {features.shape}")
        return features

    if not (xai_repo / "hcf.py").exists():
        raise FileNotFoundError(
            f"{xai_repo} does not look like an XAIinPainResearch checkout "
            "(no hcf.py). Clone it with:\n"
            "  git clone https://github.com/gouverneurp/XAIinPainResearch.git"
        )

    keep = [PAINMONIT_SENSORS.index(name) for name in sensors]
    subset = X[:, :, keep, :]

    sys.path.insert(0, str(xai_repo))
    try:
        with working_directory(xai_repo):
            hcf = importlib.import_module("hcf")
            print(f"Extracting features for {list(sensors)} on {subset.shape[0]} samples "
                  "using the authors' hcf.py (slow: cvxEDA runs per sample)...")
            features = hcf.feature_extraction(
                subset, sensor_list=list(sensors), sampling_rate=SAMPLING_RATE_HZ
            )
    finally:
        sys.path.remove(str(xai_repo))

    # Their create_hcf drops any column containing NaN before caching, and
    # prepare_data then zero-fills whatever remains. Mirror both steps.
    nan_columns = features.columns[features.isna().any()].tolist()
    if nan_columns:
        print(f"Dropping {len(nan_columns)} column(s) containing NaN: {nan_columns}")
        features = features.drop(columns=nan_columns)
    features = features.fillna(0)
    print(f"Extracted features: {features.shape}")

    if cache_csv is not None:
        cache_csv.parent.mkdir(parents=True, exist_ok=True)
        features.to_csv(cache_csv, sep=";", decimal=",", index=False)
        print(f"Cached features to {cache_csv}")
    return features


def select_classes(
    features: pd.DataFrame,
    y_onehot: np.ndarray,
    subjects: np.ndarray,
    classes: Sequence[int],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Keep two raw heater classes and re-encode them as a one-hot pair.

    Mirrors ``scripts/data_handling.py::pick_classes`` with
    ``input_is_categorical=True``: labels come back one-hot, which is what their
    random-forest regressor is fitted against.
    """
    raw = np.argmax(y_onehot, axis=1)
    mapping = {int(raw_class): idx for idx, raw_class in enumerate(classes)}
    mask = np.isin(raw, list(mapping))

    binary = np.array([mapping[int(value)] for value in raw[mask]])
    onehot = np.zeros((binary.size, len(classes)), dtype=np.float32)
    onehot[np.arange(binary.size), binary] = 1.0

    kept = features.loc[mask].reset_index(drop=True)
    print(f"Selected raw classes {list(classes)}: {mask.sum()} of {mask.size} samples "
          f"(class balance {np.bincount(binary).tolist()})")
    return kept, onehot, subjects[mask]


def loso_random_forest(
    features: pd.DataFrame,
    y_onehot: np.ndarray,
    subjects: np.ndarray,
    n_estimators: int,
    seed: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one leave-one-subject-out pass, returning per-fold accuracy and macro F1."""
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.metrics import f1_score

    accuracies, f1_scores = [], []
    for subject in np.unique(subjects):
        test_mask = subjects == subject
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=None,
            min_samples_split=2,
            random_state=seed,
            n_jobs=-1,
        )
        model.fit(features[~test_mask], y_onehot[~test_mask])

        predicted = np.argmax(model.predict(features[test_mask]), axis=1)
        actual = np.argmax(y_onehot[test_mask], axis=1)
        accuracies.append(float(np.mean(predicted == actual)))
        f1_scores.append(float(f1_score(actual, predicted, average="macro", zero_division=0)))

    return np.asarray(accuracies), np.asarray(f1_scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--xai-repo", type=Path, required=True,
                        help="Path to a clone of gouverneurp/XAIinPainResearch")
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        help="Directory holding X_pre.npy (or X.npy), y_heater.npy, subjects.npy")
    parser.add_argument("--sensors", type=str, default="Eda_RB",
                        help="Comma-separated sensor names from " + ",".join(PAINMONIT_SENSORS))
    parser.add_argument("--classes", type=str, default="0,5",
                        help="Two raw heater classes; 0,5 is the paper's B vs. P4")
    parser.add_argument("--n-estimators", type=int, default=100)
    parser.add_argument("--runs", type=int, default=5,
                        help="LOSO repetitions, matching their five_loso")
    parser.add_argument("--seed", type=int, default=None,
                        help="Base seed; run i uses seed+i. Omit for their unseeded behaviour")
    parser.add_argument("--features-csv", type=Path, default=None,
                        help="Cache path for the extracted feature table")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Debug only: keep the first N samples to smoke-test the pipeline")
    parser.add_argument("--cut", type=str, default=None,
                        help=(
                            "Crop each 10 s window to START,END in seconds before extracting "
                            "features, e.g. '1,5' for the 4 s crop our episodic model sees at "
                            "evaluation. Diagnostic: measures what the crop costs the RF, "
                            "holding features and classifier fixed. Default is the full window."
                        ))
    args = parser.parse_args()

    sensors = [name.strip() for name in args.sensors.split(",") if name.strip()]
    unknown = [name for name in sensors if name not in PAINMONIT_SENSORS]
    if unknown:
        raise ValueError(f"Unknown sensor(s) {unknown}; expected any of {PAINMONIT_SENSORS}")
    classes = [int(value) for value in args.classes.split(",")]
    if len(classes) != 2:
        raise ValueError("--classes expects exactly two raw heater classes, e.g. 0,5")

    X, y, subjects = load_arrays(args.data_dir)
    if args.max_samples is not None:
        X, y, subjects = X[: args.max_samples], y[: args.max_samples], subjects[: args.max_samples]
        print(f"DEBUG: truncated to {len(X)} samples")

    features_csv = args.features_csv
    if args.cut is not None:
        bounds = [float(value) for value in args.cut.split(",")]
        if len(bounds) != 2 or bounds[0] >= bounds[1]:
            raise ValueError("--cut expects START,END in seconds with START < END, e.g. 1,5")
        start = int(round(bounds[0] * SAMPLING_RATE_HZ))
        end = int(round(bounds[1] * SAMPLING_RATE_HZ))
        if start < 0 or end > X.shape[1]:
            raise ValueError(
                f"--cut {args.cut} is outside the {X.shape[1] / SAMPLING_RATE_HZ:g} s window"
            )
        X = X[:, start:end, :, :]
        print(f"Cropped to {bounds[0]:g}-{bounds[1]:g} s: {X.shape[1]} samples per window")
        # Keep cropped features in their own cache so they cannot collide with
        # full-window features, which have the same row count.
        if features_csv is not None:
            features_csv = features_csv.with_name(
                f"{features_csv.stem}_cut{bounds[0]:g}-{bounds[1]:g}{features_csv.suffix}"
            )

    features = build_features(X, args.xai_repo, sensors, features_csv)
    if len(features) != len(y):
        raise ValueError(
            f"Feature rows ({len(features)}) do not match samples ({len(y)}). "
            "A cached --features-csv from a different subset is the usual cause."
        )

    features, y_binary, subjects = select_classes(features, y, subjects, classes)

    print(f"\nRunning {args.runs} x LOSO over {len(np.unique(subjects))} subjects "
          f"with {args.n_estimators} trees on {features.shape[1]} features...")
    run_means, run_f1_means = [], []
    for run in range(args.runs):
        seed = None if args.seed is None else args.seed + run
        accuracies, f1_scores = loso_random_forest(
            features, y_binary, subjects, args.n_estimators, seed
        )
        run_means.append(float(np.mean(accuracies)))
        run_f1_means.append(float(np.mean(f1_scores)))
        print(f"  run {run + 1}/{args.runs}: accuracy {run_means[-1] * 100:.2f} % "
              f"(fold std {np.std(accuracies) * 100:.2f}), macro F1 {run_f1_means[-1] * 100:.2f} %")

    accuracy = float(np.mean(run_means)) * 100
    print(f"\n{'=' * 62}")
    print(f"  sensors            : {', '.join(sensors)}")
    print(f"  window             : {args.cut.replace(',', '-') + ' s' if args.cut else 'full 10 s'}")
    print(f"  raw classes        : {classes[0]} vs {classes[1]}")
    print(f"  accuracy           : {accuracy:.2f} % "
          f"(spread across runs {np.std(run_means) * 100:.2f})")
    print(f"  macro F1           : {float(np.mean(run_f1_means)) * 100:.2f} %")

    # The published figures are full-window, so only compare like with like.
    if (args.cut is None and len(sensors) == 1 and classes == [0, 5]
            and sensors[0] in PUBLISHED_B_VS_P4):
        published = PUBLISHED_B_VS_P4[sensors[0]]
        print(f"  published (B vs P4): {published:.2f} %")
        print(f"  difference         : {accuracy - published:+.2f} points")
    print("=" * 62)


if __name__ == "__main__":
    main()
