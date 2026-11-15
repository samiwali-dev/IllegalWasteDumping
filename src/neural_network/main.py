"""Spatiotemporal Motion & Scene-Change Neural Network (MLP) for MIVIA video classification."""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import torch
from torch import nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import load_clip, load_manifest, split_records
from src.optical_flow_svm.main import GRID_SIZE, IMAGE_SIZE, FRAME_COUNT, SEED, flow_features

SLUG = "neural_network"
DEFAULT_MODEL = Path(__file__).resolve().parent / "model.pt"
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "neural_network"
FEATURE_DIM = 528


def extract_nn_features(clip: np.ndarray) -> np.ndarray:
    """Extract 528 spatiotemporal features (320 optical flow + 208 scene-change & temporal diffs)."""
    base_flow = flow_features(clip)
    f_float = clip.astype(np.float32) / 255.0
    gray = np.array([cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in clip], dtype=np.float32) / 255.0
    edges = np.linspace(0, IMAGE_SIZE, GRID_SIZE + 1, dtype=int)

    early_rgb = f_float[:4].mean(axis=0)
    mid_rgb = f_float[9:15].mean(axis=0)
    late_rgb = f_float[-4:].mean(axis=0)
    diff_late_early = np.abs(late_rgb - early_rgb).mean(axis=-1)
    diff_mid_early = np.abs(mid_rgb - early_rgb).mean(axis=-1)
    diff_late_mid = np.abs(late_rgb - mid_rgb).mean(axis=-1)

    frame_diffs = np.abs(gray[1:] - gray[:-1])
    bins = np.array_split(frame_diffs, 4, axis=0)

    extra: list[float] = []
    for r in range(GRID_SIZE):
        y0, y1 = edges[r], edges[r + 1]
        for c in range(GRID_SIZE):
            x0, x1 = edges[c], edges[c + 1]
            c_le = diff_late_early[y0:y1, x0:x1]
            c_me = diff_mid_early[y0:y1, x0:x1]
            c_lm = diff_late_mid[y0:y1, x0:x1]
            extra.extend([
                float(c_le.mean()),
                float(c_le.std()),
                float((c_le > 0.08).mean()),
                float(c_me.mean()),
                float(c_lm.mean()),
            ])
            for b in bins:
                cb = b[:, y0:y1, x0:x1]
                extra.extend([float(cb.mean()), float(cb.std())])
    return np.concatenate([base_flow, np.array(extra, dtype=np.float32)]).astype(np.float32)


def build_mlp(hidden_layers: tuple[int, ...], dropout: float, input_dim: int = FEATURE_DIM) -> nn.Module:
    layers: list[nn.Module] = []
    in_dim = input_dim
    for h in hidden_layers:
        layers.extend([
            nn.Linear(in_dim, h),
            nn.BatchNorm1d(h),
            nn.GELU(),
            nn.Dropout(dropout),
        ])
        in_dim = h
    layers.append(nn.Linear(in_dim, 1))
    return nn.Sequential(*layers)


def extract_split(records: list[dict], parallel: bool = True):
    if parallel:
        def _worker(rec):
            t0 = time.perf_counter()
            clip = load_clip(rec, size=IMAGE_SIZE)
            feat = extract_nn_features(clip)
            return feat, int(rec["label"]), time.perf_counter() - t0

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            results = list(executor.map(_worker, records))
        X = np.array([r[0] for r in results], dtype=np.float32)
        y = np.array([r[1] for r in results], dtype=np.int64)
        times = [r[2] for r in results]
        return X, y, times

    X, y, times = [], [], []
    for rec in records:
        t0 = time.perf_counter()
        clip = load_clip(rec, size=IMAGE_SIZE)
        X.append(extract_nn_features(clip))
        times.append(time.perf_counter() - t0)
        y.append(int(rec["label"]))
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64), times


def compute_metrics(y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "positive_f1": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "positive_precision": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "positive_recall": float(recall_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "false_positives": int(cm[0, 1]),
        "confusion_matrix_labels_0_1": cm.tolist(),
    }


def save_predictions_csv(path: Path, records: list[dict], y_true, y_pred, probabilities, seconds) -> None:
    fieldnames = ["video_id", "label", "prediction", "probability", "inference_seconds"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec, yt, yp, prob, sec in zip(records, y_true, y_pred, probabilities, seconds):
            writer.writerow({
                "video_id": rec["video_id"],
                "label": int(yt),
                "prediction": int(yp),
                "probability": f"{float(prob):.9f}",
                "inference_seconds": f"{float(sec):.6f}",
            })


def train() -> dict:
    cv2.setNumThreads(1)
    records = load_manifest()
    train_recs, val_recs, test_recs = split_records(records, seed=SEED)

    t_start = time.perf_counter()
    X_train, y_train, _ = extract_split(train_recs, parallel=True)
    X_val, y_val, val_extract_times = extract_split(val_recs, parallel=True)

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True) + 1e-6

    X_tr_t = torch.tensor((X_train - mean) / std, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32)
    X_va_t = torch.tensor((X_val - mean) / std, dtype=torch.float32)
    y_va_t = torch.tensor(y_val, dtype=torch.float32)

    candidates = [
        {"hidden_layers": [64, 32], "dropout": 0.55, "lr": 8e-4, "weight_decay": 0.05, "batch_size": 20, "epochs_max": 40, "patience": 10},
        {"hidden_layers": [128, 64, 32], "dropout": 0.50, "lr": 5e-4, "weight_decay": 0.01, "batch_size": 16, "epochs_max": 40, "patience": 10},
        {"hidden_layers": [80, 40], "dropout": 0.55, "lr": 5e-4, "weight_decay": 0.05, "batch_size": 16, "epochs_max": 40, "patience": 10},
    ]

    best_state = None
    best_params = None
    best_val_probs = None
    best_val_preds = None
    best_rank = None
    best_epoch = 0
    candidate_results = []

    for params in candidates:
        torch.manual_seed(SEED)
        np.random.seed(SEED)
        gen = torch.Generator().manual_seed(SEED)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_tr_t, y_tr_t),
            batch_size=params["batch_size"],
            shuffle=True,
            generator=gen,
        )
        model = build_mlp(tuple(params["hidden_layers"]), params["dropout"])
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"]
        )
        criterion = nn.BCEWithLogitsLoss()

        cand_best_rank = None
        cand_best_metrics = None
        cand_best_state = None
        cand_best_probs = None
        cand_best_preds = None
        cand_best_epoch = 0
        stale = 0

        for epoch in range(1, params["epochs_max"] + 1):
            model.train()
            for bx, by in loader:
                optimizer.zero_grad()
                loss = criterion(model(bx).squeeze(-1), by)
                loss.backward()
                optimizer.step()

            model.eval()
            with torch.no_grad():
                v_logits = model(X_va_t).squeeze(-1)
                v_loss = float(criterion(v_logits, y_va_t))
                v_probs = torch.sigmoid(v_logits).numpy()
                v_preds = (v_probs >= 0.5).astype(int)
                vm = compute_metrics(y_val, v_preds)
                rank = (
                    vm["macro_f1"],
                    vm["positive_precision"],
                    vm["positive_f1"],
                    -vm["false_positives"],
                    -v_loss,
                )
                if cand_best_rank is None or rank > cand_best_rank:
                    cand_best_rank = rank
                    cand_best_metrics = {**vm, "validation_loss": v_loss, "best_epoch": epoch}
                    cand_best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    cand_best_probs = v_probs
                    cand_best_preds = v_preds
                    cand_best_epoch = epoch
                    stale = 0
                else:
                    stale += 1
                    if stale >= params["patience"]:
                        break

        candidate_results.append({"hyperparameters": params, **cand_best_metrics})
        if best_rank is None or cand_best_rank > best_rank:
            best_rank = cand_best_rank
            best_state = cand_best_state
            best_params = params
            best_val_probs = cand_best_probs
            best_val_preds = cand_best_preds
            best_epoch = cand_best_epoch

    train_seconds = time.perf_counter() - t_start

    final_model = build_mlp(tuple(best_params["hidden_layers"]), best_params["dropout"])
    final_model.load_state_dict(best_state)
    final_model.eval()

    # Single-video test inference timing
    test_probs, test_preds, test_times, y_test_list = [], [], [], []
    with torch.no_grad():
        for rec in test_recs:
            t0 = time.perf_counter()
            clip = load_clip(rec, size=IMAGE_SIZE)
            feat = extract_nn_features(clip).reshape(1, -1)
            norm_feat = torch.tensor((feat - mean) / std, dtype=torch.float32)
            prob = float(torch.sigmoid(final_model(norm_feat).reshape(-1)[0]))
            elapsed = time.perf_counter() - t0
            test_probs.append(prob)
            test_preds.append(int(prob >= 0.5))
            test_times.append(elapsed)
            y_test_list.append(int(rec["label"]))

    y_test = np.array(y_test_list, dtype=np.int64)
    test_preds_arr = np.array(test_preds, dtype=np.int64)
    test_seconds = sum(test_times)

    DEFAULT_MODEL.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_RESULTS.mkdir(parents=True, exist_ok=True)

    torch.save({
        "state_dict": best_state,
        "hidden_layers": best_params["hidden_layers"],
        "dropout": best_params["dropout"],
        "input_dim": FEATURE_DIM,
        "mean": mean.astype(np.float32),
        "std": std.astype(np.float32),
        "feature_extractor": "spatiotemporal_flow_and_scene_change_v1",
        "label_mapping": {0: "no_dumping", 1: "dumping"},
        "frame_count": FRAME_COUNT,
        "image_size": IMAGE_SIZE,
    }, DEFAULT_MODEL)

    val_csv = DEFAULT_RESULTS / "neural_network_validation_predictions.csv"
    test_csv = DEFAULT_RESULTS / "neural_network_test_predictions.csv"
    run_json = DEFAULT_RESULTS / "neural_network_run.json"

    save_predictions_csv(val_csv, val_recs, y_val, best_val_preds, best_val_probs, val_extract_times)
    save_predictions_csv(test_csv, test_recs, y_test, test_preds_arr, test_probs, test_times)

    run_data = {
        "model": SLUG,
        "method": "spatiotemporal_mlp_neural_network",
        "protocol_version": "macro_f1_v2_exploratory",
        "dataset": "MIVIA-IWDD 500",
        "seed": SEED,
        "official_test_used_for_selection": False,
        "split_counts": {"fit": len(train_recs), "validation": len(val_recs), "test": len(test_recs)},
        "frame_count": FRAME_COUNT,
        "image_size": IMAGE_SIZE,
        "feature_dimension": FEATURE_DIM,
        "architecture": {
            "name": "Spatiotemporal MLP Neural Network",
            "input_dimension": FEATURE_DIM,
            "hidden_layers": best_params["hidden_layers"],
            "activation": "GELU",
            "normalization": "BatchNorm1d",
            "dropout": best_params["dropout"],
            "output_dimension": 1,
        },
        "hyperparameters": {
            **best_params,
            "optimizer": "AdamW",
            "loss": "BCEWithLogitsLoss",
            "best_epoch": best_epoch,
            "selection": "highest validation macro F1, then dumping precision, dumping F1, fewer false positives, then lowest loss",
            "threshold": 0.5,
        },
        "validation_metrics": compute_metrics(y_val, best_val_preds),
        "test_metrics": compute_metrics(y_test, test_preds_arr),
        "candidate_results": candidate_results,
        "training_seconds": round(train_seconds, 6),
        "test_inference_seconds": round(test_seconds, 6),
        "model_path": str(DEFAULT_MODEL),
        "model_size_bytes": DEFAULT_MODEL.stat().st_size,
        "validation_predictions_path": str(val_csv),
        "test_predictions_path": str(test_csv),
    }
    run_json.write_text(json.dumps(run_data, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "validation_metrics": run_data["validation_metrics"],
        "test_metrics": run_data["test_metrics"],
    }, indent=2))
    return run_data


def predict_clip(clip: np.ndarray, checkpoint: Path = DEFAULT_MODEL) -> tuple[int, float]:
    """Predict Dumping (1) or No Dumping (0) and sigmoid probability for one clip."""
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_mlp(
        tuple(saved["hidden_layers"]),
        float(saved["dropout"]),
        int(saved.get("input_dim", FEATURE_DIM)),
    )
    model.load_state_dict(saved["state_dict"])
    model.eval()
    features = extract_nn_features(clip).reshape(1, -1)
    mean = saved["mean"]
    std = saved["std"]
    norm_features = torch.tensor((features - mean) / std, dtype=torch.float32)
    with torch.no_grad():
        logit = model(norm_features).reshape(-1)[0]
        score = float(torch.sigmoid(logit).cpu())
    label = 1 if score >= 0.5 else 0
    return label, score


def main() -> int:
    parser = argparse.ArgumentParser(description="Train or run the Spatiotemporal Neural Network (MLP).")
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
        result = analyze_video(args.video, [SLUG], device=args.device)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

