"""Train YOLOv8n from scratch on the prepared Roboflow Littering dataset."""
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
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "yolo"
DEFAULT_VIDEO_OUTPUT = DEFAULT_RESULTS / "video_predictions"
DEFAULT_IMAGE_OUTPUT = DEFAULT_RESULTS / "image_predictions"
DEFAULT_DATA = PROJECT_ROOT / "datasets" / "Littering-YOLO-clean" / "data.yaml"
DEFAULT_MODELS = DEFAULT_MODEL.parent
SEED = 42


def make_train_parser(*, add_help: bool = True) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, add_help=add_help)
    p.add_argument("--data", type=Path, default=DEFAULT_DATA)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS,
                   help="Exact base directory for the YOLO report, training, and validation folders")
    p.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS)
    p.add_argument("--overwrite", action="store_true")
    return p


def parse_train_args(argv=None) -> argparse.Namespace:
    p = make_train_parser()
    args = p.parse_args(argv)
    if any(v < 1 for v in (args.epochs, args.batch, args.patience)) or args.imgsz < 32:
        p.error("epochs, imgsz, batch, and patience must be positive")
    args.data = args.data.expanduser().resolve()
    args.results_dir = args.results_dir.expanduser().resolve()
    args.models_dir = args.models_dir.expanduser().resolve()
    return args


def resolve_device(request: str) -> str:
    if request != "auto":
        return request
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def _metric(box, attr: str) -> float | None:
    raw = getattr(box, attr, None)
    if raw is None:
        return None
    try:
        val = float(raw)
        return val if math.isfinite(val) else None
    except (TypeError, ValueError):
        return None


def _get_yolo():
    try:
        from ultralytics import YOLO
        return YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required for YOLOv8n. Install it in the project environment with: pip install ultralytics"
        ) from exc


def _extract_detections(result) -> list[dict]:
    boxes, names = result.boxes, result.names
    if boxes is None:
        return []
    coords = boxes.xyxy.detach().cpu().tolist()
    confs = boxes.conf.detach().cpu().tolist()
    clss = boxes.cls.detach().cpu().tolist()
    return [
        {
            "class_id": int(c),
            "class_name": str(names.get(int(c), int(c))) if isinstance(names, dict) else str(names[int(c)]),
            "confidence": float(conf),
            "bbox_xyxy": [round(float(v), 2) for v in box],
        }
        for box, conf, c in zip(coords, confs, clss)
    ]


def train(argv=None) -> int:
    args = parse_train_args(argv)
    if not args.data.is_file():
        raise FileNotFoundError(
            f"YOLO data.yaml not found: {args.data}. Run clean.py --dataset littering first or pass --data."
        )
    YOLO = _get_yolo()

    device = resolve_device(args.device)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    args.models_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.results_dir / "training" / "yolov8n_littering"
    saved_model = args.models_dir / "model.pt"
    metadata_path = args.results_dir / "yolov8n_run.json"

    existing = [p for p in (run_dir, saved_model, metadata_path) if p.exists()]
    if existing and not args.overwrite:
        raise FileExistsError(
            "YOLO output already exists: " + ", ".join(map(str, existing)) + ". Use --overwrite to replace it."
        )
    if args.overwrite:
        if run_dir.is_dir():
            shutil.rmtree(run_dir)
        saved_model.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    started = time.perf_counter()
    model = YOLO("yolov8n.yaml")
    model.train(
        data=str(args.data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=args.patience, workers=args.workers, device=device,
        project=str(args.results_dir / "training"), name="yolov8n_littering",
        exist_ok=args.overwrite, pretrained=False, seed=SEED, deterministic=True,
        plots=True, val=True, verbose=True,
    )
    training_seconds = time.perf_counter() - started

    trainer = getattr(model, "trainer", None)
    best_path = Path(getattr(trainer, "best", "")) if trainer else Path()
    if not best_path.is_file():
        candidate = run_dir / "weights" / "best.pt"
        if candidate.is_file():
            best_path = candidate
        else:
            raise FileNotFoundError("Training completed but best.pt could not be located")
    shutil.copy2(best_path, saved_model)

    val_start = time.perf_counter()
    metrics = YOLO(str(saved_model)).val(
        data=str(args.data), split="val", imgsz=args.imgsz, batch=args.batch,
        device=device, project=str(args.results_dir / "validation"),
        name="yolov8n_littering_val", exist_ok=True, plots=True, verbose=True,
    )
    box = getattr(metrics, "box", None)

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
            "epochs": args.epochs, "imgsz": args.imgsz, "batch": args.batch,
            "patience": args.patience, "workers": args.workers,
        },
        "training_seconds": training_seconds,
        "evaluation_split": "Roboflow validation (the supplied test images have no COCO labels)",
        "validation_seconds": time.perf_counter() - val_start,
        "metrics": {
            "precision": _metric(box, "mp"),
            "recall": _metric(box, "mr"),
            "map50": _metric(box, "map50"),
            "map50_95": _metric(box, "map"),
        },
        "model_path": str(saved_model),
        "model_size_bytes": saved_model.stat().st_size if saved_model.is_file() else 0,
        "training_run_dir": str(run_dir),
        "validation_run_dir": str(args.results_dir / "validation" / "yolov8n_littering_val"),
        "note": "YOLOv8n participates in the overall project evaluation. Its object-detection metrics use the labelled Littering validation images and are not equivalent to MIVIA video-classification metrics.",
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
    cap = cv2.VideoCapture(str(video))
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        if total < 1 or fps <= 0:
            raise ValueError(f"Could not read video frames from: {video}")
        indices = [round(i * (total - 1) / max(1, frame_count - 1)) for i in range(frame_count)]
        samples = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                raise ValueError(f"Could not decode frame {idx} from: {video}")
            samples.append((idx, idx / fps, frame))
        return samples, fps
    finally:
        cap.release()


def detect_video(
    video: str | Path,
    *,
    model_path: str | Path = DEFAULT_MODEL,
    frames: int = 12,
    conf: float = 0.25,
    device: str = "auto",
    output_dir: str | Path = DEFAULT_VIDEO_OUTPUT,
) -> dict:
    video, model_path = Path(video).expanduser().resolve(), Path(model_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() / video.stem
    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLOv8n model not found: {model_path}. Run src/yolo/main.py train to create it."
        )
    if not 0.0 <= conf <= 1.0:
        raise ValueError("conf must be between 0 and 1")

    YOLO = _get_yolo()
    samples, fps = sample_video_frames(video, frames)
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = resolve_device(device)
    results = YOLO(str(model_path)).predict(
        source=[frame for _, _, frame in samples], conf=conf, device=dev, verbose=False, save=False,
    )

    frame_reports, strongest = [], None
    for ord_idx, ((f_idx, secs, _), res) in enumerate(zip(samples, results), start=1):
        annotated = output_dir / f"frame_{ord_idx:02d}_{secs:.2f}s.jpg"
        if not cv2.imwrite(str(annotated), res.plot()):
            raise OSError(f"Could not save annotated frame: {annotated}")
        detections = _extract_detections(res)
        for item in detections:
            if strongest is None or item["confidence"] > strongest["confidence"]:
                strongest = {**item, "frame_number": ord_idx, "timestamp_seconds": round(secs, 3)}
        frame_reports.append({
            "frame_number": ord_idx, "source_frame_index": f_idx,
            "timestamp_seconds": round(secs, 3), "detections": detections,
            "annotated_frame": str(annotated),
        })

    return {
        "video": str(video),
        "model": str(model_path),
        "device": dev,
        "confidence_threshold": conf,
        "frames_sampled": len(frame_reports),
        "source_fps": fps,
        "frames_with_detections": sum(bool(r["detections"]) for r in frame_reports),
        "total_detections": sum(len(r["detections"]) for r in frame_reports),
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
    image, model_path = Path(image).expanduser().resolve(), Path(model_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve()
    if not image.is_file():
        raise FileNotFoundError(f"Image not found: {image}")
    if image.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        raise ValueError("Select a JPG, JPEG, PNG, BMP, or WEBP image")
    if not model_path.is_file():
        raise FileNotFoundError(f"YOLOv8n model not found: {model_path}. Run src/yolo/main.py train first.")
    if not 0.0 <= conf <= 1.0:
        raise ValueError("conf must be between 0 and 1")

    YOLO = _get_yolo()
    dev = resolve_device(device)
    res = YOLO(str(model_path)).predict(
        source=str(image), conf=conf, device=dev, verbose=False, save=False,
    )[0]

    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / f"{image.stem}_yolov8n.jpg"
    res.save(filename=str(out_file))

    return {
        "image": str(image),
        "model": str(model_path),
        "device": dev,
        "confidence_threshold": conf,
        "detections": _extract_detections(res),
        "annotated_image": str(out_file),
    }


def main(argv=None) -> int:
    """One entry point for YOLO training, video scanning, and image detection."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("train", parents=[make_train_parser(add_help=False)], help="Train YOLOv8n")

    pred_base = argparse.ArgumentParser(add_help=False)
    pred_base.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    pred_base.add_argument("--conf", type=float, default=0.25)
    pred_base.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")

    vid_p = sub.add_parser("predict", parents=[pred_base], help="Scan sampled frames from an MP4 video")
    vid_p.add_argument("video", type=Path)
    vid_p.add_argument("--frames", type=int, default=12)
    vid_p.add_argument("--output-dir", type=Path, default=DEFAULT_VIDEO_OUTPUT)

    img_p = sub.add_parser("image", parents=[pred_base], help="Detect littering in one image")
    img_p.add_argument("image", type=Path)
    img_p.add_argument("--output-dir", type=Path, default=DEFAULT_IMAGE_OUTPUT)

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