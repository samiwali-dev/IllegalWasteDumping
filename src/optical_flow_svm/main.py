"""Optical Flow + SVM model for MIVIA video classification."""

import argparse
import csv
import json
from pathlib import Path
import sys
import time

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

SEED = 42
FRAME_COUNT = 24
IMAGE_SIZE = 112
GRID_SIZE = 4
TIME_BINS = 4
FLOW_LIMIT = 20.0
MOTION_THRESHOLD = 1.5
FEATURE_DIM = TIME_BINS * GRID_SIZE * GRID_SIZE * 5


def flow_features(clip):
    """Extract 320 optical flow features from a 24x112x112 RGB video clip."""
    gray = [cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in clip]
    edges = np.linspace(0, IMAGE_SIZE, GRID_SIZE + 1, dtype=int)
    pair_features = []

    for i in range(len(gray) - 1):
        flow = cv2.calcOpticalFlowFarneback(
            gray[i], gray[i + 1], None,
            pyr_scale=0.5, levels=2, winsize=15,
            iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
        )
        flow = np.clip(flow, -FLOW_LIMIT, FLOW_LIMIT)
        mag = cv2.magnitude(flow[..., 0], flow[..., 1])

        row_feats = []
        for r in range(GRID_SIZE):
            y0, y1 = edges[r], edges[r + 1]
            for c in range(GRID_SIZE):
                x0, x1 = edges[c], edges[c + 1]
                cell_flow = flow[y0:y1, x0:x1]
                cell_mag = mag[y0:y1, x0:x1]
                moving_ratio = np.count_nonzero(cell_mag > MOTION_THRESHOLD) / cell_mag.size
                row_feats.extend([
                    float(cell_flow[..., 0].mean()),
                    float(cell_flow[..., 1].mean()),
                    float(cell_mag.mean()),
                    float(cell_mag.std()),
                    float(moving_ratio),
                ])
        pair_features.append(row_feats)

    pair_features = np.array(pair_features, dtype=np.float32)
    bins = np.array_split(pair_features, TIME_BINS, axis=0)
    features = np.concatenate([b.mean(axis=0) for b in bins])
    return features.astype(np.float32)


def extract_split(records):
    X, y, times = [], [], []
    for record in records:
        t0 = time.perf_counter()
        clip = load_clip(record, size=IMAGE_SIZE)
        X.append(flow_features(clip))
        times.append(time.perf_counter() - t0)
        y.append(int(record["label"]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), times


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "positive_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "false_positives": int(cm[0, 1]),
        "confusion_matrix_labels_0_1": cm.tolist(),
    }


def save_predictions_csv(path, records, y_true, y_pred, scores, seconds):
    fieldnames = ["video_id", "label", "prediction", "decision_score", "inference_seconds"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec, yt, yp, sc, sec in zip(records, y_true, y_pred, scores, seconds):
            writer.writerow({
                "video_id": rec["video_id"],
                "label": int(yt),
                "prediction": int(yp),
                "decision_score": float(sc),
                "inference_seconds": round(float(sec), 6),
            })


def train():
    cv2.setNumThreads(1)
    records = load_manifest()
    train_recs, val_recs, test_recs = split_records(records, seed=SEED)

    t_start = time.perf_counter()
    X_train, y_train, _ = extract_split(train_recs)
    X_val, y_val, val_times = extract_split(val_recs)

    candidates = [
        {"kernel": "linear", "C": 0.01},
        {"kernel": "linear", "C": 0.1},
        {"kernel": "linear", "C": 1.0},
        {"kernel": "rbf", "C": 0.1, "gamma": "scale"},
        {"kernel": "rbf", "C": 1.0, "gamma": "scale"},
        {"kernel": "rbf", "C": 10.0, "gamma": "scale"},
    ]

    best_model, best_params, best_preds, best_rank = None, None, None, None
    candidate_results = []

    for params in candidates:
        model = Pipeline([("scale", StandardScaler()), ("svm", SVC(**params))])
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        m = compute_metrics(y_val, preds)
        candidate_results.append({"hyperparameters": params, **m})
        rank = (m["macro_f1"], m["positive_precision"], m["positive_f1"], -m["false_positives"])
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_model = model
            best_params = params
            best_preds = preds

    train_seconds = time.perf_counter() - t_start
    val_scores = best_model.decision_function(X_val)

    t_test = time.perf_counter()
    X_test, y_test, test_times = extract_split(test_recs)
    test_scores = best_model.decision_function(X_test)
    test_preds = (test_scores > 0.0).astype(int)
    test_seconds = time.perf_counter() - t_test

    DEFAULT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_RESULTS.mkdir(parents=True, exist_ok=True)

    joblib.dump({
        "model": best_model,
        "feature_extractor": "farneback_spatiotemporal_v1",
        "label_mapping": {0: "no_dumping", 1: "dumping"},
        "frame_count": FRAME_COUNT,
        "image_size": IMAGE_SIZE,
    }, DEFAULT_MODEL, compress=3)

    val_csv = DEFAULT_RESULTS / "optical_flow_svm_validation_predictions.csv"
    test_csv = DEFAULT_RESULTS / "optical_flow_svm_test_predictions.csv"
    run_json = DEFAULT_RESULTS / "optical_flow_svm_run.json"

    save_predictions_csv(val_csv, val_recs, y_val, best_preds, val_scores, val_times)
    save_predictions_csv(test_csv, test_recs, y_test, test_preds, test_scores, test_times)

    run_data = {
        "method": "optical_flow_svm",
        "dataset": "MIVIA-IWDD 500",
        "seed": SEED,
        "split_counts": {"fit": len(train_recs), "validation": len(val_recs), "test": len(test_recs)},
        "frame_count": FRAME_COUNT,
        "image_size": IMAGE_SIZE,
        "feature_dimension": FEATURE_DIM,
        "hyperparameters": best_params,
        "validation_metrics": compute_metrics(y_val, best_preds),
        "candidate_results": candidate_results,
        "training_seconds": round(train_seconds, 6),
        "test_inference_seconds": round(test_seconds, 6),
        "model_path": str(DEFAULT_MODEL),
        "model_size_bytes": DEFAULT_MODEL.stat().st_size,
        "validation_predictions_path": str(val_csv),
        "test_predictions_path": str(test_csv),
    }
    run_json.write_text(json.dumps(run_data, indent=2), encoding="utf-8")
    print(json.dumps(run_data["validation_metrics"], indent=2))


def predict_clip(clip, checkpoint=DEFAULT_MODEL):
    """Predict Dumping (1) or No Dumping (0) and decision score for one clip."""
    saved = joblib.load(checkpoint)
    model = saved["model"] if isinstance(saved, dict) else saved
    features = flow_features(clip).reshape(1, -1)
    score = float(model.decision_function(features)[0])
    label = 1 if score >= 0.0 else 0
    return label, score


def main():
    parser = argparse.ArgumentParser(description="Train or run Optical Flow + SVM.")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("train")
    pred = sub.add_parser("predict")
    pred.add_argument("video", type=Path)
    pred.add_argument("--device", default="auto")
    args = parser.parse_args()

    if args.command == "train":
        train()
    elif args.command == "predict":
        from common import analyze_video
        result = analyze_video(args.video, ["optical_flow_svm"], device=args.device)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
