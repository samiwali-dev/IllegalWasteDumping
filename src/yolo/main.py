"""Train YOLOv8n from scratch on the prepared Roboflow Littering dataset.

No pretrained weights are downloaded or loaded. ``yolov8n.yaml`` supplies only
the network architecture and all learnable weights start from random values.
YOLO evaluation is kept separate from the four MIVIA video-classification
models because this script performs object detection on individual images.

Prepare COCO data first if necessary:
    .venv/bin/python clean.py --dataset littering

Train:
    .venv/bin/python src/yolo/main.py train

Reports and runs are stored under results/yolo. Annotated image and video
predictions are stored in its image_predictions and video_predictions folders.
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
from pathlib import Path


import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL = Path(__file__).resolve().parent / "model.pt"
DEFAULT_VIDEO_OUTPUT = PROJECT_ROOT / "results" / "yolo" / "video_predictions"
DEFAULT_IMAGE_OUTPUT = PROJECT_ROOT / "results" / "yolo" / "image_predictions"
DEFAULT_DATA = PROJECT_ROOT / "datasets" / "Littering-YOLO-clean" / "data.yaml"
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "yolo"
DEFAULT_MODELS = DEFAULT_MODEL.parent
SEED = 42


def make_train_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, add_help=add_help)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    # These defaults are sized for the project's Apple Silicon laptop. They
    # still train on every labeled image; callers can raise them if desired.
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS,
                        help="Exact base directory for the YOLO report, training, and validation folders")
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def parse_train_args(argv=None) -> argparse.Namespace:
    parser = make_train_parser()
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.imgsz < 32 or args.batch < 1 or args.patience < 1:
        parser.error("epochs, imgsz, batch, and patience must be positive")
    args.data = args.data.expanduser().resolve()
    args.results_dir = args.results_dir.expanduser().resolve()
    args.models_dir = args.models_dir.expanduser().resolve()
    return args


def resolve_device(request: str) -> str:
    if request != "auto":
        return request
    try:
        import torch
    except ImportError:
        return "cpu"
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def finite_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def metric_value(box, attribute: str):
    return finite_or_none(getattr(box, attribute, None)) if box is not None else None


def model_size_bytes(path: Path) -> int:
    return path.stat().st_size if path.is_file() else 0


def train(argv=None) -> int:
    args = parse_train_args(argv)
    if not args.data.is_file():
        raise FileNotFoundError(
            f"YOLO data.yaml not found: {args.data}. Run clean.py --dataset littering first or pass --data."
        )
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required for YOLOv8n. Install it in the project environment with: pip install ultralytics"
        ) from exc

    device = resolve_device(args.device)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    training_root = args.results_dir / "training"
    run_name = "yolov8n_littering"
    run_dir = training_root / run_name
    saved_model = args.models_dir / "model.pt"
    metadata_path = args.results_dir / "yolov8n_run.json"

    existing = [path for path in (run_dir, saved_model, metadata_path) if path.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "YOLO output already exists: " + ", ".join(map(str, existing)) + ". Use --overwrite to replace it."
        )
    if args.overwrite:
        if run_dir.is_dir():
            shutil.rmtree(run_dir)
        saved_model.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    # Important: YAML = architecture only. This does NOT load yolov8n.pt.
    model = YOLO("yolov8n.yaml")
    started = time.perf_counter()
    train_results = model.train(
        data=str(args.data),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        device=device,
        project=str(training_root),
        name=run_name,
        exist_ok=args.overwrite,
        pretrained=False,
        seed=SEED,
        deterministic=True,
        plots=True,
        val=True,
        verbose=True,
    )
    training_seconds = time.perf_counter() - started

    trainer = getattr(model, "trainer", None)
    best_path = Path(getattr(trainer, "best", "")) if trainer is not None else Path()
    if not best_path.is_file():
        candidate = run_dir / "weights" / "best.pt"
        if candidate.is_file():
            best_path = candidate
        else:
            raise FileNotFoundError("Training completed but best.pt could not be located")
    shutil.copy2(best_path, saved_model)

    best_model = YOLO(str(saved_model))
    validation_started = time.perf_counter()
    validation_metrics = best_model.val(
        data=str(args.data),
        split="val",
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        project=str(args.results_dir / "validation"),
        name="yolov8n_littering_val",
        exist_ok=True,
        plots=True,
        verbose=True,
    )
    validation_seconds = time.perf_counter() - validation_started
    box = getattr(validation_metrics, "box", None)

    report = {
        "model": "YOLOv8n",
        "task": "object_detection",
        "dataset": "Roboflow Littering",
        "data_yaml": str(args.data),
        "pretrained": False,
        "initialization": "random",
        "architecture_source": "yolov8n.yaml",
        "device": device,
        "seed": SEED,
        "hyperparameters": {
            "epochs": args.epochs,
            "imgsz": args.imgsz,
            "batch": args.batch,
            "patience": args.patience,
            "workers": args.workers,
        },
        "training_seconds": training_seconds,
        "evaluation_split": "Roboflow validation (the supplied test images have no COCO labels)",
        "validation_seconds": validation_seconds,
        "metrics": {
            "precision": metric_value(box, "mp"),
            "recall": metric_value(box, "mr"),
            "map50": metric_value(box, "map50"),
            "map50_95": metric_value(box, "map"),
        },
        "model_path": str(saved_model),
        "model_size_bytes": model_size_bytes(saved_model),
        "training_run_dir": str(run_dir),
        "validation_run_dir": str(args.results_dir / "validation" / "yolov8n_littering_val"),
        "note": "YOLOv8n metrics are object-detection metrics and are intentionally not added to the four-model MIVIA video-classification comparison.",
    }
    metadata_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0




def sample_video_frames(video: str | Path, frame_count: int) -> tuple[list[tuple[int, float, object]], float]:
    """Read evenly spaced BGR frames as (source_index, seconds, image)."""
    video = Path(video).expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(f"Video not found: {video}")
    if video.suffix.lower() != ".mp4":
        raise ValueError("Select an MP4 video")
    if frame_count < 1:
        raise ValueError("frames must be at least 1")
    capture = cv2.VideoCapture(str(video))
    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if total < 1 or fps <= 0:
            raise ValueError(f"Could not read video frames from: {video}")
        indices = [round(i * (total - 1) / max(1, frame_count - 1)) for i in range(frame_count)]
        samples = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok or frame is None:
                raise ValueError(f"Could not decode frame {index} from: {video}")
            samples.append((index, index / fps, frame))
        return samples, fps
    finally:
        capture.release()


def detect_video(
    video: str | Path,
    *,
    model_path: str | Path = DEFAULT_MODEL,
    frames: int = 12,
    conf: float = 0.25,
    device: str = "auto",
    output_dir: str | Path = DEFAULT_VIDEO_OUTPUT,
) -> dict:
    video = Path(video).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() / video.stem
    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLOv8n model not found: {model_path}. Run src/yolo/main.py train to create it."
        )
    if not 0.0 <= conf <= 1.0:
        raise ValueError("conf must be between 0 and 1")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required. Install project requirements first.") from exc

    samples, fps = sample_video_frames(video, frames)
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_device = resolve_device(device)
    model = YOLO(str(model_path))
    results = model.predict(
        source=[frame for _, _, frame in samples],
        conf=conf,
        device=resolved_device,
        verbose=False,
        save=False,
    )
    frame_reports = []
    strongest = None
    for ordinal, ((frame_index, seconds, _), result) in enumerate(zip(samples, results), start=1):
        annotated = output_dir / f"frame_{ordinal:02d}_{seconds:.2f}s.jpg"
        if not cv2.imwrite(str(annotated), result.plot()):
            raise OSError(f"Could not save annotated frame: {annotated}")
        detections = []
        boxes = result.boxes
        names = result.names
        if boxes is not None:
            for coords, confidence, class_id in zip(
                boxes.xyxy.detach().cpu().tolist(),
                boxes.conf.detach().cpu().tolist(),
                boxes.cls.detach().cpu().tolist(),
            ):
                cid = int(class_id)
                item = {
                    "class_id": cid,
                    "class_name": str(names.get(cid, cid)) if isinstance(names, dict) else str(names[cid]),
                    "confidence": float(confidence),
                    "bbox_xyxy": [round(float(value), 2) for value in coords],
                }
                detections.append(item)
                if strongest is None or item["confidence"] > strongest["confidence"]:
                    strongest = {**item, "frame_number": ordinal, "timestamp_seconds": round(seconds, 3)}
        frame_reports.append({
            "frame_number": ordinal,
            "source_frame_index": frame_index,
            "timestamp_seconds": round(seconds, 3),
            "detections": detections,
            "annotated_frame": str(annotated),
        })
    return {
        "video": str(video),
        "model": str(model_path),
        "device": resolved_device,
        "confidence_threshold": conf,
        "frames_sampled": len(frame_reports),
        "source_fps": fps,
        "frames_with_detections": sum(bool(row["detections"]) for row in frame_reports),
        "total_detections": sum(len(row["detections"]) for row in frame_reports),
        "strongest_detection": strongest,
        "output_dir": str(output_dir),
        "frame_results": frame_reports,
    }



def detect_image(
    image: str | Path,
    *,
    model_path: str | Path = DEFAULT_MODEL,
    conf: float = 0.25,
    device: str = "auto",
    output_dir: str | Path = DEFAULT_IMAGE_OUTPUT,
) -> dict:
    image = Path(image).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(f"Image not found: {image}")
    if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        raise ValueError("Select a JPG, JPEG, PNG, BMP, or WEBP image")
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLOv8n model not found: {model_path}. Run src/yolo/main.py train first.")
    if not 0.0 <= conf <= 1.0:
        raise ValueError("conf must be between 0 and 1")
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("Ultralytics is required. Install it with: pip install ultralytics") from exc

    resolved_device = resolve_device(device)
    model = YOLO(str(model_path))
    result = model.predict(
        source=str(image),
        conf=conf,
        device=resolved_device,
        verbose=False,
        save=False,
    )[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{image.stem}_yolov8n.jpg"
    result.save(filename=str(output_path))

    detections = []
    boxes = result.boxes
    names = result.names
    if boxes is not None:
        xyxy = boxes.xyxy.detach().cpu().tolist()
        confs = boxes.conf.detach().cpu().tolist()
        classes = boxes.cls.detach().cpu().tolist()
        for coords, confidence, class_id in zip(xyxy, confs, classes):
            cid = int(class_id)
            detections.append({
                "class_id": cid,
                "class_name": str(names.get(cid, cid)) if isinstance(names, dict) else str(names[cid]),
                "confidence": float(confidence),
                "bbox_xyxy": [round(float(value), 2) for value in coords],
            })

    return {
        "image": str(image),
        "model": str(model_path),
        "device": resolved_device,
        "confidence_threshold": conf,
        "detections": detections,
        "annotated_image": str(output_path),
    }



def main(argv=None) -> int:
    """One entry point for YOLO training, video scanning, and image detection."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command")
    commands.add_parser("train", parents=[make_train_parser(add_help=False)],
                        help="Train YOLOv8n from random initialization")
    video = commands.add_parser("predict", help="Scan sampled frames from an MP4 video")
    video.add_argument("video", type=Path)
    video.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    video.add_argument("--frames", type=int, default=12)
    video.add_argument("--conf", type=float, default=0.25)
    video.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    video.add_argument("--output-dir", type=Path, default=DEFAULT_VIDEO_OUTPUT)
    image = commands.add_parser("image", help="Detect littering regions in one image")
    image.add_argument("image", type=Path)
    image.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    image.add_argument("--conf", type=float, default=0.25)
    image.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    image.add_argument("--output-dir", type=Path, default=DEFAULT_IMAGE_OUTPUT)
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    if args.command == "train":
        return train(argv[1:])
    if args.command == "predict":
        report = detect_video(args.video, model_path=args.model, frames=args.frames,
                              conf=args.conf, device=args.device, output_dir=args.output_dir)
    else:
        report = detect_image(args.image, model_path=args.model, conf=args.conf,
                              device=args.device, output_dir=args.output_dir)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

