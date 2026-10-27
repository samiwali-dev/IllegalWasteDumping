"""Train and evaluate a video-level optical-flow/SVM baseline on MIVIA.

All models use the same 24 ordered RGB frames and the same official MIVIA split.
Hyperparameters are selected on an 80-video validation subset of the official
training split. The selected model remains fitted on the other 320 videos; the
100 official test videos are loaded only after selection is complete.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import load_clip, load_manifest, split_records


DEFAULT_MODEL = Path(__file__).resolve().parent / "model.joblib"
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "optical_flow_svm"
LABEL_NAMES = {0: "no_dumping", 1: "dumping"}
SEED = 42
FRAME_COUNT = 24
IMAGE_SIZE = 112
GRID_SIZE = 4
TIME_BINS = 4
CELL_FEATURES = ("mean_dx", "mean_dy", "mean_magnitude", "std_magnitude", "moving_fraction")
FLOW_LIMIT = 20.0
MOTION_THRESHOLD = 1.5
FEATURE_DIM = TIME_BINS * GRID_SIZE * GRID_SIZE * len(CELL_FEATURES)
PREDICTION_FIELDS = ("video_id", "label", "prediction", "decision_score", "inference_seconds")


def make_train_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, add_help=add_help)
    parser.add_argument("--report", type=Path, default=None, help="MIVIA frame extraction report")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODEL.parent)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS,
                        help="Exact output directory for SVM reports and prediction CSVs")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument(
        "--smoke-check",
        action="store_true",
        help="Check the split and flow features on four training/validation clips; write no artifacts",
    )
    modes.add_argument(
        "--refresh-inference",
        action="store_true",
        help="Re-time held-out test inference from the saved model after verifying unchanged predictions",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Replace this script's existing model and result files"
    )
    return parser


def parse_args(argv=None) -> argparse.Namespace:
    return make_train_parser().parse_args(argv)


def flow_features(clip: np.ndarray) -> np.ndarray:
    """Summarize Farneback flow in a 4x4 spatial grid and four temporal bins.

    The five cell statistics capture direction, motion strength and activity.
    Everything is computed from the clip itself, without labels or learned
    preprocessing. Input is the shared 24x112x112 RGB letterboxed clip.
    """
    expected_shape = (FRAME_COUNT, IMAGE_SIZE, IMAGE_SIZE, 3)
    if not isinstance(clip, np.ndarray) or clip.shape != expected_shape or clip.dtype != np.uint8:
        raise ValueError(f"Expected uint8 RGB clip of shape {expected_shape}; got {getattr(clip, 'shape', None)}")

    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in clip]
    edges = np.linspace(0, IMAGE_SIZE, GRID_SIZE + 1, dtype=int)
    per_pair = np.empty((FRAME_COUNT - 1, GRID_SIZE * GRID_SIZE * len(CELL_FEATURES)), dtype=np.float32)

    for pair_index, (previous, current) in enumerate(zip(gray[:-1], gray[1:])):
        flow = cv2.calcOpticalFlowFarneback(
            previous,
            current,
            None,
            pyr_scale=0.5,
            levels=2,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        np.clip(flow, -FLOW_LIMIT, FLOW_LIMIT, out=flow)
        magnitude = cv2.magnitude(flow[..., 0], flow[..., 1])
        position = 0
        for row in range(GRID_SIZE):
            y0, y1 = edges[row : row + 2]
            for column in range(GRID_SIZE):
                x0, x1 = edges[column : column + 2]
                cell_flow = flow[y0:y1, x0:x1]
                cell_magnitude = magnitude[y0:y1, x0:x1]
                per_pair[pair_index, position : position + len(CELL_FEATURES)] = (
                    cell_flow[..., 0].mean(),
                    cell_flow[..., 1].mean(),
                    cell_magnitude.mean(),
                    cell_magnitude.std(),
                    np.count_nonzero(cell_magnitude > MOTION_THRESHOLD) / cell_magnitude.size,
                )
                position += len(CELL_FEATURES)

    features = np.concatenate(
        [per_pair[indices].mean(axis=0) for indices in np.array_split(np.arange(FRAME_COUNT - 1), TIME_BINS)]
    )
    if features.shape != (FEATURE_DIM,) or not np.isfinite(features).all():
        raise RuntimeError("Flow extraction produced invalid features")
    return features


def feature_matrix(records: list[dict], phase: str) -> tuple[np.ndarray, np.ndarray, list[float]]:
    features = np.empty((len(records), FEATURE_DIM), dtype=np.float32)
    labels = np.empty(len(records), dtype=np.int64)
    seconds: list[float] = []
    for index, record in enumerate(records):
        started = time.perf_counter()
        features[index] = flow_features(load_clip(record, size=IMAGE_SIZE))
        seconds.append(time.perf_counter() - started)
        labels[index] = int(record["label"])
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            print(f"{phase}: extracted {index + 1}/{len(records)} videos", flush=True)
    return features, labels, seconds


def candidate_parameters() -> list[dict]:
    # Fixed, compact search space; no test-set feedback enters this choice.
    return [
        *({"kernel": "linear", "C": value} for value in (0.01, 0.1, 1.0)),
        *({"kernel": "rbf", "C": value, "gamma": "scale"} for value in (0.1, 1.0, 10.0)),
    ]


def make_model(parameters: dict) -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("svm", SVC(**parameters)),
        ]
    )


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "positive_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "false_positives": int(matrix[0, 1]),
        "confusion_matrix_labels_0_1": matrix.tolist(),
    }


def check_splits(train: list[dict], validation: list[dict], test: list[dict]) -> None:
    if (len(train), len(validation), len(test)) != (320, 80, 100):
        raise ValueError("Expected 320/80/100 video-level fit/validation/test split")
    identities = [{record["video_id"] for record in part} for part in (train, validation, test)]
    if any(len(part) != len(ids) for part, ids in zip((train, validation, test), identities)):
        raise ValueError("Duplicate video ID in a partition")
    if any(identities[left] & identities[right] for left, right in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Video ID crosses fit/validation/test partitions")
    if any(int(record["label"]) not in (0, 1) for part in (train, validation, test) for record in part):
        raise ValueError("All labels must be 0 or 1")
    if any(record["split"] != "train" for record in train + validation):
        raise ValueError("Fit and validation videos must come from the official training split")
    if any(record["split"] != "test" for record in test):
        raise ValueError("Test videos must come from the official test split")


def rows_for_records(
    records: list[dict], labels: np.ndarray, predictions: np.ndarray, scores: np.ndarray, seconds: list[float]
) -> list[dict]:
    return [
        {
            "video_id": record["video_id"],
            "label": int(label),
            "prediction": int(prediction),
            "decision_score": float(score),
            "inference_seconds": round(float(duration), 6),
        }
        for record, label, prediction, score, duration in zip(
            records, labels, predictions, scores, seconds, strict=True
        )
    ]


def check_binary_svm(model: Pipeline) -> None:
    if not isinstance(model, Pipeline) or "svm" not in model.named_steps:
        raise ValueError("Expected a saved StandardScaler/SVM pipeline")
    classes = getattr(model.named_steps["svm"], "classes_", None)
    if classes is None or not np.array_equal(classes, np.array([0, 1])):
        raise ValueError(f"Expected fitted SVM classes [0, 1], found {classes}")


def prediction_from_score(score: float) -> int:
    """Binary SVC decision scores are positive for classes_[1] == dumping."""
    if not math.isfinite(score):
        raise ValueError("SVM returned a non-finite decision score")
    return int(score > 0.0)


def infer_one_by_one(model: Pipeline, records: list[dict]) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    check_binary_svm(model)
    labels = np.empty(len(records), dtype=np.int64)
    predictions = np.empty(len(records), dtype=np.int64)
    scores = np.empty(len(records), dtype=np.float64)
    durations: list[float] = []
    for index, record in enumerate(records):
        started = time.perf_counter()
        feature = flow_features(load_clip(record, size=IMAGE_SIZE))[None, :]
        scores[index] = float(model.decision_function(feature)[0])
        predictions[index] = prediction_from_score(scores[index])
        durations.append(time.perf_counter() - started)
        labels[index] = int(record["label"])
        if (index + 1) % 25 == 0 or index + 1 == len(records):
            print(f"test: inferred {index + 1}/{len(records)} videos", flush=True)
    return labels, predictions, rows_for_records(records, labels, predictions, scores, durations)


def write_csv(path: Path, rows: list[dict]) -> None:
    with tempfile.NamedTemporaryFile("w", newline="", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def write_json(path: Path, data: dict) -> None:
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(data, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    os.replace(temporary, path)


def write_model(path: Path, model: Pipeline) -> None:
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as handle:
        temporary = Path(handle.name)
    try:
        joblib.dump(
            {
                "model": model,
                "feature_extractor": "farneback_spatiotemporal_v1",
                "label_mapping": {0: "no_dumping", 1: "dumping"},
                "frame_count": FRAME_COUNT,
                "image_size": IMAGE_SIZE,
            },
            temporary,
            compress=3,
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_backup(source: Path, destination: Path) -> None:
    with source.open("rb") as original, destination.open("xb") as backup:
        shutil.copyfileobj(original, backup)
    shutil.copystat(source, destination)


def refresh_inference(model_path: Path, test_path: Path, run_path: Path, test: list[dict]) -> None:
    """Re-measure test inference without changing training or model selection.

    Existing labels and decision scores are checked before any output changes.
    Stable backup names prevent accidentally overwriting a previous snapshot.
    """
    from evaluate import _resolve_model_path

    for path in (model_path, test_path, run_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required existing artifact is missing: {path}")
    backup_test = test_path.with_name(test_path.name + ".before_inference_refresh.bak")
    backup_run = run_path.with_name(run_path.name + ".before_inference_refresh.bak")
    for path in (backup_test, backup_run):
        if path.exists():
            raise FileExistsError(f"Refusing to replace an existing inference-refresh backup: {path}")

    with run_path.open("r", encoding="utf-8") as handle:
        run = json.load(handle)
    expected_counts = {"fit": 320, "validation": 80, "test": 100}
    if (
        run.get("method") != "optical_flow_svm"
        or run.get("seed") != SEED
        or run.get("split_counts") != expected_counts
        or run.get("frame_count") != FRAME_COUNT
        or run.get("image_size") != IMAGE_SIZE
        or run.get("feature_extractor") != "farneback_spatiotemporal_v1"
        or run.get("model_size_bytes") != model_path.stat().st_size
        or not isinstance(run.get("model_path"), str)
        or _resolve_model_path(run["model_path"], run_path).resolve() != model_path.resolve()
        or run.get("test_predictions_path") != str(test_path)
    ):
        raise ValueError("Saved run metadata does not match the requested MIVIA model and test artifacts")

    with test_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(PREDICTION_FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"Test prediction CSV lacks required columns: {test_path}")
        old_rows = list(reader)
    if len(old_rows) != len(test):
        raise ValueError(f"Expected {len(test)} saved test predictions, found {len(old_rows)}")
    for record, old in zip(test, old_rows, strict=True):
        if old["video_id"] != record["video_id"] or int(old["label"]) != int(record["label"]):
            raise ValueError(f"Saved test row does not match the official test video: {record['video_id']}")

    bundle = joblib.load(model_path)
    if not isinstance(bundle, dict) or bundle.get("feature_extractor") != "farneback_spatiotemporal_v1":
        raise ValueError("Saved model uses an unexpected feature extractor")
    if bundle.get("frame_count") != FRAME_COUNT or bundle.get("image_size") != IMAGE_SIZE:
        raise ValueError("Saved model uses unexpected clip dimensions")
    model = bundle.get("model")
    check_binary_svm(model)

    print("Refreshing test inference timing from the saved model", flush=True)
    started = time.perf_counter()
    _, _, new_rows = infer_one_by_one(model, test)
    test_seconds = time.perf_counter() - started
    maximum_score_difference = 0.0
    for old, new in zip(old_rows, new_rows, strict=True):
        if int(old["prediction"]) != new["prediction"]:
            raise ValueError(f"Prediction changed for {new['video_id']}; no artifacts were replaced")
        old_score = float(old["decision_score"])
        new_score = float(new["decision_score"])
        if not math.isfinite(old_score) or not math.isclose(old_score, new_score, rel_tol=1e-6, abs_tol=1e-5):
            raise ValueError(f"Decision score changed for {new['video_id']}; no artifacts were replaced")
        maximum_score_difference = max(maximum_score_difference, abs(old_score - new_score))

    updated_run = dict(run)
    updated_run["test_inference_seconds"] = round(test_seconds, 6)
    updated_run["mean_test_inference_seconds_per_video"] = round(test_seconds / len(test), 6)
    updated_run["inference_implementation"] = "one_decision_function_call_per_video"
    updated_run["refresh_verification"] = {
        "matched_test_predictions": len(test),
        "maximum_absolute_decision_score_difference": maximum_score_difference,
        "previous_test_predictions_backup": str(backup_test),
        "previous_run_metadata_backup": str(backup_run),
    }

    write_backup(test_path, backup_test)
    write_backup(run_path, backup_run)
    try:
        write_csv(test_path, new_rows)
        write_json(run_path, updated_run)
    except BaseException:
        shutil.copy2(backup_test, test_path)
        shutil.copy2(backup_run, run_path)
        raise
    print(json.dumps({"refreshed_test_predictions": str(test_path), "run_path": str(run_path), "matched_predictions": len(test), "maximum_absolute_decision_score_difference": maximum_score_difference}, indent=2))


def train(argv=None) -> None:
    args = parse_args(argv)
    cv2.setNumThreads(1)
    records = load_manifest(report_path=args.report)
    train, validation, test = split_records(records, seed=SEED)
    check_splits(train, validation, test)

    if args.smoke_check:
        checked = []
        for record in train[:2] + validation[:2]:
            values = flow_features(load_clip(record, size=IMAGE_SIZE))
            checked.append({"video_id": record["video_id"], "label": record["label"], "feature_dimension": len(values), "feature_norm": float(np.linalg.norm(values))})
        print(json.dumps({"split_counts": {"fit": 320, "validation": 80, "test": 100}, "checked": checked}, indent=2))
        return

    model_path = args.models_dir.expanduser().resolve() / "model.joblib"
    validation_path = args.results_dir.resolve() / "optical_flow_svm_validation_predictions.csv"
    test_path = args.results_dir.resolve() / "optical_flow_svm_test_predictions.csv"
    run_path = args.results_dir.resolve() / "optical_flow_svm_run.json"
    if args.refresh_inference:
        if args.overwrite:
            raise ValueError("--refresh-inference cannot be combined with --overwrite")
        refresh_inference(model_path, test_path, run_path, test)
        return
    outputs = (model_path, validation_path, test_path, run_path)
    if not args.overwrite and any(path.exists() for path in outputs):
        existing = [str(path) for path in outputs if path.exists()]
        raise FileExistsError(f"Existing output(s): {existing}. Use --overwrite to replace them.")
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    training_started = time.perf_counter()
    fit_features, fit_labels, _ = feature_matrix(train, "fit")
    validation_features, validation_labels, validation_extract_seconds = feature_matrix(validation, "validation")
    candidate_results = []
    selected_model = None
    selected_parameters = None
    selected_predictions = None
    selected_rank = (-1.0, -1.0, -1.0, float("-inf"))
    candidates = candidate_parameters()
    for candidate_index, parameters in enumerate(candidates, 1):
        model = make_model(parameters)
        model.fit(fit_features, fit_labels)
        predictions = model.predict(validation_features)
        result = metrics(validation_labels, predictions)
        candidate_results.append({"hyperparameters": parameters, **result})
        print(
            f"validation candidate {candidate_index}/{len(candidates)}: "
            f"{parameters} macro_f1={result['macro_f1']:.4f} "
            f"precision={result['positive_precision']:.4f} "
            f"positive_f1={result['positive_f1']:.4f} "
            f"false_positives={result['false_positives']}",
            flush=True,
        )
        rank = (
            result["macro_f1"],
            result["positive_precision"],
            result["positive_f1"],
            -result["false_positives"],
        )
        if rank > selected_rank:
            selected_rank = rank
            selected_model = model
            selected_parameters = parameters
            selected_predictions = predictions
    if selected_model is None or selected_parameters is None or selected_predictions is None:
        raise RuntimeError("No SVM candidate was fitted")
    training_seconds = time.perf_counter() - training_started

    check_binary_svm(selected_model)
    validation_scores = np.empty(len(validation), dtype=np.float64)
    validation_predict_seconds = []
    for index, feature in enumerate(validation_features):
        started = time.perf_counter()
        validation_scores[index] = float(selected_model.decision_function(feature[None, :])[0])
        validation_predict_seconds.append(time.perf_counter() - started)
        if prediction_from_score(validation_scores[index]) != int(selected_predictions[index]):
            raise RuntimeError(f"SVM score disagrees with validation prediction for {validation[index]['video_id']}")
    validation_rows = rows_for_records(
        validation,
        validation_labels,
        selected_predictions,
        validation_scores,
        [extract + predict for extract, predict in zip(validation_extract_seconds, validation_predict_seconds, strict=True)],
    )

    # The official test data is accessed only after selecting the 320-video model.
    test_started = time.perf_counter()
    _, _, test_rows = infer_one_by_one(selected_model, test)
    test_seconds = time.perf_counter() - test_started

    write_model(model_path, selected_model)
    write_csv(validation_path, validation_rows)
    write_csv(test_path, test_rows)
    run = {
        "method": "optical_flow_svm",
        "protocol_version": "macro_f1_v2_exploratory",
        "dataset": "MIVIA-IWDD 500",
        "seed": SEED,
        "split_counts": {"fit": len(train), "validation": len(validation), "test": len(test)},
        "label_mapping": {"0": "no_dumping", "1": "dumping"},
        "frame_count": FRAME_COUNT,
        "image_size": IMAGE_SIZE,
        "feature_extractor": "farneback_spatiotemporal_v1",
        "feature_dimension": FEATURE_DIM,
        "feature_specification": {
            "spatial_grid": [GRID_SIZE, GRID_SIZE],
            "temporal_bins": TIME_BINS,
            "cell_statistics": CELL_FEATURES,
            "flow_clip_pixels": FLOW_LIMIT,
            "moving_threshold_pixels": MOTION_THRESHOLD,
            "farneback": {"pyr_scale": 0.5, "levels": 2, "winsize": 15, "iterations": 3, "poly_n": 5, "poly_sigma": 1.2},
        },
        "selection_metric": "validation_macro_f1_then_dumping_precision_then_dumping_f1_then_fewer_false_positives",
        "candidate_results": candidate_results,
        "hyperparameters": selected_parameters,
        "validation_metrics": metrics(validation_labels, selected_predictions),
        "training_seconds": round(training_seconds, 6),
        "test_inference_seconds": round(test_seconds, 6),
        "mean_test_inference_seconds_per_video": round(test_seconds / len(test), 6),
        "inference_implementation": "one_decision_function_call_per_video",
        "model_path": str(model_path),
        "model_size_bytes": model_path.stat().st_size,
        "validation_predictions_path": str(validation_path),
        "test_predictions_path": str(test_path),
    }
    write_json(run_path, run)
    print(json.dumps({"run_path": str(run_path), "test_predictions_path": str(test_path), "validation_macro_f1": run["validation_metrics"]["macro_f1"]}, indent=2))


def predict_clip(clip: np.ndarray, checkpoint: str | Path | None = None) -> tuple[int, float]:
    """Classify the shared 24-frame RGB clip; return label and raw SVM score."""
    checkpoint = Path(checkpoint).expanduser().resolve() if checkpoint is not None else DEFAULT_MODEL
    if not checkpoint.is_file():
        raise FileNotFoundError(f"SVM model not found: {checkpoint}")
    saved = joblib.load(checkpoint)
    if (
        not isinstance(saved, dict)
        or saved.get("feature_extractor") != "farneback_spatiotemporal_v1"
        or saved.get("label_mapping") != LABEL_NAMES
        or saved.get("frame_count") != FRAME_COUNT
        or saved.get("image_size") != IMAGE_SIZE
    ):
        raise ValueError(f"Incompatible optical-flow checkpoint: {checkpoint}")
    model = saved.get("model")
    check_binary_svm(model)
    score = float(model.decision_function(flow_features(clip)[None, :])[0])
    if not math.isfinite(score):
        raise ValueError(f"Invalid SVM decision score from checkpoint: {checkpoint}")
    return int(score >= 0.0), score


def main(argv=None) -> int:
    """Train or classify a video without depending on legacy source scripts."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("train", parents=[make_train_parser(add_help=False)],
                        help="Train the optical-flow/SVM model")
    prediction = commands.add_parser("predict", help="Classify one MP4 video")
    prediction.add_argument("video", type=Path)
    prediction.add_argument("--device", choices=("auto", "cpu", "mps"), default="auto")
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    if args.command == "train":
        train(argv[1:])
    elif args.command == "predict":
        from common import analyze_video

        rows = analyze_video(args.video, ["optical_flow_svm"], device=args.device)
        print(json.dumps(rows, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

