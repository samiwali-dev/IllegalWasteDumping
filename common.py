"""Shared, reproducible training protocol for the three MIVIA PyTorch models.

All neural models use the same video split, clips, loss, optimizer, training
budget, validation selection rule, and test output format. PyTorch is imported
only when an experiment starts so ``--help`` works before installation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Callable

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
SEED = 42
CLIP_LENGTH = 24
IMAGE_SIZE = 112
CHECKPOINT_PATHS = {
    "optical_flow_svm": PROJECT_ROOT / "src/optical_flow_svm/model.joblib",
    "cnn_lstm": PROJECT_ROOT / "src/cnn_lstm/model.pt",
    "cnn_3d": PROJECT_ROOT / "src/cnn_3d/model.pt",
    "movinet_a0": PROJECT_ROOT / "src/movinet/model.pt",
}
LABEL_NAMES = {0: "no_dumping", 1: "dumping"}


def make_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--report", type=Path, default=None, help="Validated 24-frame extraction report")
    parser.add_argument("--results-dir", type=Path, default=None,
                        help="Output directory; defaults to results/<algorithm>")
    parser.add_argument("--models-dir", type=Path, default=None)
    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Optional checkpoint path; defaults to this algorithm's model.pt")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Decode clips on every epoch instead of keeping uint8 clips in RAM",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace this model's existing checkpoint and result files",
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        raise ValueError("epochs, batch-size, and patience must be positive")
    if args.learning_rate <= 0 or args.weight_decay < 0:
        raise ValueError("learning-rate must be positive and weight-decay nonnegative")


def _device(torch, request: str):
    if request == "auto":
        if torch.backends.mps.is_available():
            request = "mps"
        elif torch.cuda.is_available():
            request = "cuda"
        else:
            request = "cpu"
    if request == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; choose --device cpu")
    if request == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; choose --device cpu")
    return torch.device(request)


def _synchronize(torch, device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def _seed_everything(torch) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)


def _tensor_from_clip(torch, clip: np.ndarray):
    if clip.shape != (CLIP_LENGTH, IMAGE_SIZE, IMAGE_SIZE, 3) or clip.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 RGB clip [{CLIP_LENGTH},{IMAGE_SIZE},{IMAGE_SIZE},3]; "
            f"got {clip.shape}, {clip.dtype}"
        )
    return torch.from_numpy(np.ascontiguousarray(clip)).permute(0, 3, 1, 2).float().div_(255.0)


def _scores(labels: list[int], predictions: list[int]) -> dict:
    if len(labels) != len(predictions) or not labels:
        raise ValueError("Labels and predictions must have equal, nonzero lengths")
    tn = sum(a == 0 and b == 0 for a, b in zip(labels, predictions))
    fp = sum(a == 0 and b == 1 for a, b in zip(labels, predictions))
    fn = sum(a == 1 and b == 0 for a, b in zip(labels, predictions))
    tp = sum(a == 1 and b == 1 for a, b in zip(labels, predictions))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    no_dumping_f1 = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else 0.0
    return {
        "accuracy": (tn + tp) / len(labels),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": (f1 + no_dumping_f1) / 2,
        "confusion_matrix": [[tn, fp], [fn, tp]],
    }


def _write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", prefix=f".{path.stem}-",
        dir=path.parent, delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    temp_path.replace(path)


def _write_csv_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = ("video_id", "label", "prediction", "probability", "inference_seconds")
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="", suffix=".csv",
        prefix=f".{path.stem}-", dir=path.parent, delete=False,
    ) as handle:
        temp_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temp_path.replace(path)


def _save_state_atomic(torch, model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        suffix=".pt", prefix=f".{path.stem}-", dir=path.parent, delete=False
    ) as handle:
        temp_path = Path(handle.name)
    try:
        torch.save(model.state_dict(), temp_path)
        os.replace(temp_path, path)
    finally:
        temp_path.unlink(missing_ok=True)


def run_experiment(
    args: argparse.Namespace,
    *,
    slug: str,
    model_factory: Callable,
    architecture: dict,
) -> dict:
    """Fit on 320, select on 80, then infer exactly once on official test 100."""
    _validate_args(args)
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required. Install project requirements first.") from exc

    from data import load_clip, load_manifest, split_records

    _seed_everything(torch)
    device = _device(torch, args.device)
    records = load_manifest(report_path=args.report)
    train_records, val_records, test_records = split_records(records, seed=SEED)
    counts = {"train": len(train_records), "validation": len(val_records), "test": len(test_records)}
    if counts != {"train": 320, "validation": 80, "test": 100}:
        raise ValueError(f"Unexpected MIVIA split counts: {counts}")
    id_sets = [set(record["video_id"] for record in subset) for subset in
               (train_records, val_records, test_records)]
    if any(len(ids) != n for ids, n in zip(id_sets, counts.values())) or any(
        id_sets[i] & id_sets[j] for i, j in ((0, 1), (0, 2), (1, 2))
    ):
        raise ValueError("Training, validation, and test video IDs must be unique and disjoint")
    if any(record["split"] != "test" for record in test_records):
        raise ValueError("Test records must come from the official MIVIA test split")

    class ClipDataset(torch.utils.data.Dataset):
        def __init__(self, items: list[dict]):
            self.items = items
            self.cache: dict[int, np.ndarray] = {}

        def __len__(self) -> int:
            return len(self.items)

        def __getitem__(self, index: int):
            clip = self.cache.get(index)
            if clip is None:
                clip = load_clip(self.items[index], size=IMAGE_SIZE)
                if not args.no_cache:
                    self.cache[index] = clip
            return _tensor_from_clip(torch, clip), int(self.items[index]["label"])

    train_set = ClipDataset(train_records)
    val_set = ClipDataset(val_records)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True, generator=generator,
        num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, num_workers=0,
    )

    model = model_factory().to(device)
    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    result_folder = "movinet" if slug == "movinet_a0" else slug
    args.results_dir = (
        args.results_dir or PROJECT_ROOT / "results" / result_folder
    ).expanduser().resolve()
    args.models_dir = (args.models_dir or CHECKPOINT_PATHS[slug].parent).expanduser().resolve()
    checkpoint = (args.checkpoint or args.models_dir / "model.pt").expanduser().resolve()
    predictions_path = args.results_dir / f"{slug}_test_predictions.csv"
    run_path = args.results_dir / f"{slug}_run.json"
    existing = [str(path) for path in (checkpoint, predictions_path, run_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            f"Existing output(s): {existing}. Use --overwrite only to intentionally replace them."
        )

    best_rank = (-1.0, -1.0, -1.0, float("-inf"), float("-inf"))
    best_epoch = 0
    best_validation: dict = {}
    stale_epochs = 0
    history: list[dict] = []
    print(f"{slug}: {counts}, device={device}, parameters={sum(p.numel() for p in model.parameters())}", flush=True)
    started = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        for clips, labels in train_loader:
            clips = clips.to(device)
            labels = labels.to(device=device, dtype=torch.float32)
            optimizer.zero_grad(set_to_none=True)
            logits = model(clips).reshape(-1)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_sum += float(loss.detach().cpu()) * len(labels)

        model.eval()
        val_loss_sum = 0.0
        val_labels: list[int] = []
        val_predictions: list[int] = []
        with torch.inference_mode():
            for clips, labels in val_loader:
                clips = clips.to(device)
                target = labels.to(device=device, dtype=torch.float32)
                logits = model(clips).reshape(-1)
                val_loss_sum += float(criterion(logits, target).cpu()) * len(labels)
                val_labels.extend(int(value) for value in labels.tolist())
                val_predictions.extend(int(value) for value in
                                       (torch.sigmoid(logits) >= 0.5).to("cpu").tolist())
        train_loss = train_loss_sum / len(train_set)
        val_loss = val_loss_sum / len(val_set)
        val_metrics = _scores(val_labels, val_predictions)
        history.append({"epoch": epoch, "train_loss": train_loss, "validation_loss": val_loss,
                        "validation_metrics": val_metrics})
        rank = (
            val_metrics["macro_f1"],
            val_metrics["precision"],
            val_metrics["f1"],
            -val_metrics["confusion_matrix"][0][1],
            -val_loss,
        )
        if rank > best_rank:
            best_rank = rank
            best_epoch = epoch
            best_validation = {"loss": val_loss, **val_metrics}
            _save_state_atomic(torch, model, checkpoint)
            stale_epochs = 0
        else:
            stale_epochs += 1
        print(
            f"{slug} epoch {epoch}/{args.epochs}: train loss {train_loss:.4f}, "
            f"val loss {val_loss:.4f}, val macro F1 {val_metrics['macro_f1']:.4f}, "
            f"dumping F1 {val_metrics['f1']:.4f}, "
            f"val accuracy {val_metrics['accuracy']:.4f}"
            + (" [best]" if best_epoch == epoch else ""),
            flush=True,
        )
        if stale_epochs >= args.patience:
            print(f"{slug}: early stop after epoch {epoch}", flush=True)
            break

    _synchronize(torch, device)
    training_seconds = time.perf_counter() - started
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()

    test_rows: list[dict] = []
    with torch.inference_mode():
        for record in test_records:
            sample_started = time.perf_counter()
            clip = load_clip(record, size=IMAGE_SIZE)
            tensor = _tensor_from_clip(torch, clip).unsqueeze(0).to(device)
            probability = float(torch.sigmoid(model(tensor).reshape(-1)[0]).cpu())
            _synchronize(torch, device)
            test_rows.append({
                "video_id": record["video_id"],
                "label": int(record["label"]),
                "prediction": int(probability >= 0.5),
                "probability": f"{probability:.9f}",
                "inference_seconds": f"{time.perf_counter() - sample_started:.6f}",
            })

    _write_csv_atomic(predictions_path, test_rows)
    run = {
        "model": slug,
        "protocol_version": "macro_f1_v2_exploratory",
        "seed": SEED,
        "official_test_used_for_selection": False,
        "split_counts": counts,
        "input": {"frames": CLIP_LENGTH, "height": IMAGE_SIZE, "width": IMAGE_SIZE,
                  "channels": 3, "color": "RGB", "range": "0..1", "pretrained": False},
        "architecture": architecture,
        "hyperparameters": {
            "optimizer": "AdamW", "loss": "BCEWithLogitsLoss", "epochs_max": args.epochs,
            "batch_size": args.batch_size, "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay, "early_stopping_patience": args.patience,
            "selection": "highest validation macro F1, then dumping precision, dumping F1, fewer false positives, then lowest loss",
            "threshold": 0.5,
        },
        "device": str(device),
        "torch_version": torch.__version__,
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_epoch": best_epoch,
        "validation_metrics": best_validation,
        "training_history": history,
        "training_seconds": round(training_seconds, 6),
        "inference_seconds": round(sum(float(row["inference_seconds"]) for row in test_rows), 6),
        "model_path": str(checkpoint),
        "model_size_bytes": checkpoint.stat().st_size,
        "predictions_path": str(predictions_path),
    }
    _write_json_atomic(run_path, run)
    print(f"{slug}: saved {checkpoint}, {predictions_path}, {run_path}", flush=True)
    return run


def analyze_video(video, model_names, *, device="auto", progress=None):
    """Extract one shared clip and classify it with the selected saved models."""
    import time
    import cv2
    from data import IMAGE_SIZE, SEQUENCE_LENGTH, load_clip, extract_video

    video = Path(video).expanduser().resolve()
    if not video.is_file() or video.suffix.lower() != ".mp4":
        raise ValueError("Select an existing MP4 video.")
    model_names = list(model_names)
    if not model_names or any(name not in CHECKPOINT_PATHS for name in model_names):
        raise ValueError("Select a supported model.")
    for name in model_names:
        if not CHECKPOINT_PATHS[name].is_file():
            raise FileNotFoundError(f"Saved model missing: {name}")
    if progress:
        progress("Reading video and extracting 24 ordered frames...")
    cv2.setNumThreads(1)
    with tempfile.TemporaryDirectory(prefix="mivia-app-") as temporary:
        frames_dir = Path(temporary) / "frames"
        extract_video(video, frames_dir, SEQUENCE_LENGTH, jpeg_quality=92)
        clip = load_clip({"output_dir": str(frames_dir)}, size=IMAGE_SIZE)
    rows = []
    for name in model_names:
        if progress:
            progress(f"Analyzing with {name}...")
        started = time.perf_counter()
        checkpoint = CHECKPOINT_PATHS[name]
        if name == "optical_flow_svm":
            label, score = predict_svm(clip, checkpoint)
            kind = "SVM decision score (threshold 0)"
        else:
            label, score, _ = predict_torch(clip, checkpoint, name, device)
            kind = "Uncalibrated sigmoid (threshold 0.5)"
        rows.append({"model": name, "prediction": label, "score": score,
                     "score_kind": kind, "seconds": time.perf_counter() - started})
    return rows


def predict_svm(clip, checkpoint: Path) -> tuple[int, float]:
    from src.optical_flow_svm.main import predict_clip

    return predict_clip(clip, checkpoint)


def predict_torch(clip, checkpoint: Path, model_name: str, requested_device: str) -> tuple[int, float, str]:
    import torch

    if model_name == "cnn_lstm":
        from src.cnn_lstm.main import build_model
    elif model_name == "cnn_3d":
        from src.cnn_3d.main import build_model
    elif model_name == "movinet_a0":
        from src.movinet.main import build_model
    else:
        raise ValueError(f"Unknown PyTorch model: {model_name}")

    if requested_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; use --device cpu")
    device = torch.device(
        "mps" if requested_device == "auto" and torch.backends.mps.is_available()
        else "cpu" if requested_device == "auto" else requested_device
    )
    model = build_model().to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
    model.eval()
    tensor = _tensor_from_clip(torch, clip).unsqueeze(0).to(device)
    with torch.inference_mode():
        logits = model(tensor).reshape(-1)
        if logits.numel() != 1:
            raise RuntimeError(f"Expected one video score, got {logits.numel()}")
        score = float(torch.sigmoid(logits[0]).cpu())
    if not math.isfinite(score):
        raise ValueError(f"Invalid prediction from checkpoint: {checkpoint}")
    return int(score >= 0.5), score, str(device)

