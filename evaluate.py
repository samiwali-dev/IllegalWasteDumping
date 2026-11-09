"""Report all active algorithms without mixing different evaluation tasks.

Each prediction CSV needs ``video_id,label,prediction`` and may include
``inference_seconds``, ``decision_score``/``score``, or ``probability``.
``decision_score`` and ``score`` remain raw model outputs; they are never
interpreted as probabilities. Each model can also supply a metadata JSON with
``training_seconds``, ``model_path`` or ``model_size_bytes``, ``split_counts``,
``hyperparameters``, and ``protocol_version``. The CLI saves a concise CSV
comparison beside its detailed JSON report. Saved YOLO detection metrics are
included in the same report with their actual dataset and evaluation split.
Detection mAP is not video-classification accuracy and is not ranked against it.
This command does not train models or run inference again.

Example:
    python evaluate.py \
        --prediction movinet_a0=results/movinet/movinet_a0_test_predictions.csv \
        --metadata movinet_a0=results/movinet/movinet_a0_run.json \
        --yolo-report results/yolo/yolov8n_run.json \
        --output results/comparison/model_comparison.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable

from data import PROJECT_ROOT, SEED, load_manifest, split_records


DEFAULT_OUTPUT = PROJECT_ROOT / "results" / "comparison" / "model_comparison.json"
DEFAULT_YOLO_REPORT = PROJECT_ROOT / "results" / "yolo" / "yolov8n_run.json"
ACTIVE_VIDEO_MODELS = {"movinet_a0"}

# Resolve the historical location of the saved MoViNet checkpoint.
LEGACY_MODEL_ALIASES = {
    "movinet_a0_best.pt": PROJECT_ROOT / "src" / "movinet" / "model.pt",
}
COMPARISON_COLUMNS = [
    "model", "task", "dataset", "evaluation_split", "protocol_version",
    "accuracy", "precision", "recall", "f1", "map50", "map50_95",
    "TN", "FP", "FN", "TP", "training_seconds",
    "inference_mean_ms_per_video", "model_size_mib",
]


def evaluate_predictions(y_true: Iterable[int], y_pred: Iterable[int]) -> dict[str, Any]:
    """Compute binary video-level scores for positive label 1 (dumping)."""
    true = list(y_true)
    pred = list(y_pred)
    if not true or len(true) != len(pred):
        raise ValueError("True and predicted labels must have equal nonzero lengths")
    if any(type(value) is not int or value not in {0, 1} for value in true + pred):
        raise ValueError("Labels and predictions must be binary integers 0 or 1")
    tn = sum(actual == 0 and guessed == 0 for actual, guessed in zip(true, pred))
    fp = sum(actual == 0 and guessed == 1 for actual, guessed in zip(true, pred))
    fn = sum(actual == 1 and guessed == 0 for actual, guessed in zip(true, pred))
    tp = sum(actual == 1 and guessed == 1 for actual, guessed in zip(true, pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "accuracy": (tp + tn) / len(true),
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


def _nonnegative_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonnegative number") from exc
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative number")
    return result


def _named_paths(values: list[str] | None, option: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for value in values or []:
        name, separator, raw_path = value.partition("=")
        if not separator or not name.strip() or not raw_path.strip():
            raise ValueError(f"{option} must use NAME=PATH: {value!r}")
        name = name.strip()
        if name in paths:
            raise ValueError(f"Duplicate model name for {option}: {name}")
        paths[name] = Path(raw_path.strip()).expanduser()
    return paths


def _read_predictions(
    path: Path, test_by_id: dict[str, dict[str, Any]]
) -> tuple[dict[str, Any], dict[str, Any]]:
    rows: dict[str, dict[str, str]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames or []
        if not {"video_id", "label", "prediction"}.issubset(fields):
            raise ValueError(f"{path} needs video_id,label,prediction columns")
        for row in reader:
            video_id = (row.get("video_id") or "").strip()
            if video_id not in test_by_id:
                raise ValueError(f"{path}: unknown or non-test video ID {video_id!r}")
            if video_id in rows:
                raise ValueError(f"{path}: duplicate video ID {video_id}")
            label = (row.get("label") or "").strip()
            prediction = (row.get("prediction") or "").strip()
            if label not in {"0", "1"} or prediction not in {"0", "1"}:
                raise ValueError(f"{path}: invalid binary label/prediction for {video_id}")
            if int(label) != test_by_id[video_id]["label"]:
                raise ValueError(f"{path}: label does not match manifest for {video_id}")
            rows[video_id] = row

    missing = set(test_by_id) - set(rows)
    if missing:
        raise ValueError(f"{path}: missing {len(missing)} official test videos, e.g. {sorted(missing)[0]}")
    video_ids = sorted(test_by_id)
    metrics = evaluate_predictions(
        (test_by_id[video_id]["label"] for video_id in video_ids),
        (int(rows[video_id]["prediction"]) for video_id in video_ids),
    )

    inference_values: list[float] = []
    if "inference_seconds" in fields:
        for video_id in video_ids:
            raw = (rows[video_id].get("inference_seconds") or "").strip()
            if raw:
                inference_values.append(_nonnegative_float(raw, f"inference_seconds for {video_id}"))
        if inference_values and len(inference_values) != len(video_ids):
            raise ValueError(f"{path}: inference_seconds must be recorded for every test video")

    if "probability" in fields:
        for video_id in video_ids:
            raw = (rows[video_id].get("probability") or "").strip()
            if raw:
                probability = _nonnegative_float(raw, f"probability for {video_id}")
                if probability > 1:
                    raise ValueError(f"{path}: probability outside [0,1] for {video_id}")
    # Raw decision scores may have any finite sign; they are not probabilities.
    score_column = next((name for name in ("decision_score", "score") if name in fields), None)
    if score_column:
        for video_id in video_ids:
            raw = (rows[video_id].get(score_column) or "").strip()
            if raw and not math.isfinite(float(raw)):
                raise ValueError(f"{path}: nonfinite {score_column} for {video_id}")

    timing = {
        "inference_total_seconds": sum(inference_values) if inference_values else None,
        "inference_mean_ms_per_video": (
            1000 * sum(inference_values) / len(video_ids) if inference_values else None
        ),
    }
    details = {
        "prediction_csv": str(path.resolve()),
        "test_videos": len(video_ids),
        "raw_score_column": score_column,
        "probability_column": "probability" if "probability" in fields else None,
    }
    return metrics, {**timing, **details}


def _resolve_model_path(raw_path: str, metadata_path: Path) -> Path:
    """Resolve stored paths, including primary checkpoints moved into src."""
    candidate = Path(raw_path).expanduser()
    possibilities = (
        [candidate] if candidate.is_absolute()
        else [PROJECT_ROOT / candidate, metadata_path.parent / candidate]
    )
    for possibility in possibilities:
        if possibility.exists():
            return possibility

    # Match exactly models/<primary filename>, not any file with that basename.
    # An archived experiment must never silently resolve to a current model.
    if candidate.parent.name == "models":
        relocated = LEGACY_MODEL_ALIASES.get(candidate.name)
        if relocated is not None and relocated.is_file():
            return relocated
    return possibilities[-1]


def _file_size(path: Path) -> int:
    if path.is_file():
        return path.stat().st_size
    if path.is_dir():
        return sum(child.stat().st_size for child in path.rglob("*") if child.is_file())
    raise FileNotFoundError(f"Model artifact does not exist: {path}")


def _read_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"Model metadata must be a JSON object: {path}")
    training_seconds = metadata.get("training_seconds")
    if training_seconds is not None:
        training_seconds = _nonnegative_float(training_seconds, "training_seconds")
    split_counts = metadata.get("split_counts")
    if split_counts is not None:
        if not isinstance(split_counts, dict):
            raise ValueError(f"split_counts must be a JSON object: {path}")
        expected = {
            "fit": 320, "train": 320, "training": 320,
            "val": 80, "validation": 80, "test": 100,
        }
        for key, value in split_counts.items():
            if key in expected and (type(value) is not int or value != expected[key]):
                raise ValueError(f"{path}: split_counts.{key} must be {expected[key]}")
    model_paths = metadata.get("model_paths")
    if model_paths is not None and (not isinstance(model_paths, list) or
                                    not all(isinstance(value, str) for value in model_paths)):
        raise ValueError(f"model_paths must be a list of file paths: {path}")
    if metadata.get("model_path") is not None:
        if not isinstance(metadata["model_path"], str):
            raise ValueError(f"model_path must be a file path string: {path}")
        if model_paths is not None:
            raise ValueError(f"Use either model_path or model_paths in {path}")
        model_paths = [metadata["model_path"]]
    resolved_paths = [_resolve_model_path(value, path) for value in (model_paths or [])]
    if resolved_paths:
        model_size = sum(_file_size(value) for value in resolved_paths)
    elif metadata.get("model_size_bytes") is not None:
        model_size = metadata["model_size_bytes"]
        if type(model_size) is not int or model_size < 0:
            raise ValueError(f"model_size_bytes must be a nonnegative integer: {path}")
    else:
        model_size = None
    return {
        "metadata_json": str(path.resolve()),
        "protocol_version": metadata.get("protocol_version"),
        "training_seconds": training_seconds,
        "model_size_bytes": model_size,
        "model_size_mib": model_size / (1024 * 1024) if model_size is not None else None,
        "model_paths": [str(value.resolve()) for value in resolved_paths],
        "split_counts": split_counts,
        "hyperparameters": metadata.get("hyperparameters"),
    }


def _read_detection_metadata(path: Path) -> dict[str, Any]:
    """Include saved detector scores, retaining their actual dataset and split."""
    with path.open("r", encoding="utf-8") as handle:
        source = json.load(handle)
    if not isinstance(source, dict) or source.get("task") != "object_detection":
        raise ValueError(f"Expected object_detection metadata: {path}")
    if source.get("model") not in {"YOLOv8n", "yolov8n"}:
        raise ValueError(f"Expected the active YOLOv8n detector: {path}")
    if not source.get("dataset") or not source.get("evaluation_split"):
        raise ValueError(f"Detection metadata needs its dataset and evaluation_split: {path}")
    saved_metrics = source.get("metrics")
    if not isinstance(saved_metrics, dict):
        raise ValueError(f"Detection metrics must be a JSON object: {path}")
    metrics = {}
    for name in ("precision", "recall", "map50", "map50_95"):
        value = _nonnegative_float(saved_metrics.get(name), f"detection {name}")
        if value > 1:
            raise ValueError(f"Detection {name} must be in [0,1]: {path}")
        metrics[name] = value
    raw_model_path = source.get("model_path")
    if not isinstance(raw_model_path, str):
        raise ValueError(f"Detection metadata needs model_path: {path}")
    model_path = _resolve_model_path(raw_model_path, path)
    size = _file_size(model_path)
    training_seconds = source.get("training_seconds")
    if training_seconds is not None:
        training_seconds = _nonnegative_float(training_seconds, "detection training_seconds")
    validation_seconds = source.get("validation_seconds")
    if validation_seconds is not None:
        validation_seconds = _nonnegative_float(validation_seconds, "detection validation_seconds")
    return {
        "task": "object_detection",
        "dataset": source["dataset"],
        "evaluation_split": source["evaluation_split"],
        "metrics": metrics,
        "metadata_json": str(path.resolve()),
        "source_data_yaml": source.get("data_yaml"),
        "training_seconds": training_seconds,
        "validation_seconds": validation_seconds,
        "model_paths": [str(model_path.resolve())],
        "model_size_bytes": size,
        "model_size_mib": size / (1024 * 1024),
        "hyperparameters": source.get("hyperparameters"),
        "pretrained": source.get("pretrained"),
        "note": (
            "Included in the overall project evaluation. These are the saved "
            "Roboflow region-detection scores on the stated split, not MIVIA "
            "Dumping/No Dumping video scores or evidence that tracking/event "
            "detection was evaluated. No accuracy, video F1, confusion matrix, "
            "or per-video latency is invented for this detector."
        ),
    }


def evaluate_runs(
    prediction_paths: dict[str, Path],
    metadata_paths: dict[str, Path] | None = None,
    report_path: str | Path | None = None,
    yolo_report_path: str | Path | None = None,
) -> dict[str, Any]:
    """Evaluate saved MIVIA labels and optionally include saved YOLO scores."""
    if not prediction_paths:
        raise ValueError("Provide at least one model prediction CSV")
    unsupported = set(prediction_paths) - ACTIVE_VIDEO_MODELS
    if unsupported:
        raise ValueError(f"Unsupported active video models: {sorted(unsupported)}")
    metadata_paths = metadata_paths or {}
    unknown_metadata = set(metadata_paths) - set(prediction_paths)
    if unknown_metadata:
        raise ValueError(f"Metadata without predictions: {sorted(unknown_metadata)}")
    train, validation, test = split_records(load_manifest(report_path), seed=SEED)
    test_by_id = {record["video_id"]: record for record in test}
    summary: dict[str, Any] = {
        "dataset": "MIVIA-IWDD 500",
        "positive_class": "dumping",
        "evaluation_split": "official_test",
        "seed": SEED,
        "split_counts": {"fit": len(train), "validation": len(validation), "test": len(test)},
        "models": {},
        "comparison_note": (
            "MoViNet uses the MIVIA video-classification test set. YOLO is "
            "reported with its saved Roboflow detection metrics and actual "
            "evaluation split. MIVIA "
            "results remain exploratory because the official test set had "
            "already been examined before the selection protocol was revised."
        ),
    }
    for name, prediction_path in prediction_paths.items():
        metrics, details = _read_predictions(Path(prediction_path), test_by_id)
        metadata = _read_metadata(Path(metadata_paths[name])) if name in metadata_paths else {}
        summary["models"][name] = {
            "task": "video_classification",
            "dataset": summary["dataset"],
            "evaluation_split": summary["evaluation_split"],
            "metrics": metrics, **details, **metadata,
        }
    if yolo_report_path is not None:
        summary["models"]["yolov8n"] = _read_detection_metadata(Path(yolo_report_path))
    summary["included_algorithms"] = list(summary["models"])
    return summary


def write_comparison_csv(summary: dict[str, Any], path: str | Path) -> None:
    """Save every included algorithm; task-inapplicable cells stay empty."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        for name, result in summary["models"].items():
            metrics = result["metrics"]
            writer.writerow({
                "model": name,
                "task": result.get("task", "video_classification"),
                "dataset": result.get("dataset", summary.get("dataset")),
                "evaluation_split": result.get("evaluation_split", summary.get("evaluation_split")),
                "protocol_version": result.get("protocol_version"),
                "accuracy": metrics.get("accuracy"),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics.get("f1"),
                "map50": metrics.get("map50"),
                "map50_95": metrics.get("map50_95"),
                "TN": metrics.get("true_negative"),
                "FP": metrics.get("false_positive"),
                "FN": metrics.get("false_negative"),
                "TP": metrics.get("true_positive"),
                "training_seconds": result.get("training_seconds"),
                "inference_mean_ms_per_video": result.get("inference_mean_ms_per_video"),
                "model_size_mib": result.get("model_size_mib"),
            })


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prediction", action="append", required=True, metavar="NAME=CSV")
    parser.add_argument("--metadata", action="append", metavar="NAME=JSON")
    parser.add_argument(
        "--yolo-report", "--detection-metadata", type=Path, default=None,
        help="Saved YOLOv8n report to include; defaults to results/yolo/yolov8n_run.json if present",
    )
    parser.add_argument("--report", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        predictions = _named_paths(args.prediction, "--prediction")
        metadata = _named_paths(args.metadata, "--metadata")
        yolo_report = args.yolo_report
        if yolo_report is None and DEFAULT_YOLO_REPORT.is_file():
            yolo_report = DEFAULT_YOLO_REPORT
        summary = evaluate_runs(predictions, metadata, args.report, yolo_report)
        if yolo_report is None:
            summary["warnings"] = [
                f"YOLOv8n saved evaluation is unavailable at {DEFAULT_YOLO_REPORT}; "
                "its scores were not fabricated or added to this report."
            ]
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    write_comparison_csv(summary, args.output.with_suffix(".csv"))
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

