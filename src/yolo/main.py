"""YOLO item inference. Training is implemented in train_clean.py."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL = Path(__file__).resolve().parent / "model.pt"
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "yolo"
DEFAULT_VIDEO_OUTPUT = DEFAULT_RESULTS / "video_predictions"
DEFAULT_IMAGE_OUTPUT = DEFAULT_RESULTS / "image_predictions"
_saved_report = DEFAULT_RESULTS / 'yolov8n_run.json'
DEFAULT_CONFIDENCE = float(json.loads(_saved_report.read_text()).get('confidence_threshold', .25)) if _saved_report.is_file() else .25


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
    conf: float = DEFAULT_CONFIDENCE,
    device: str = "auto",
    output_dir: str | Path = DEFAULT_VIDEO_OUTPUT,
) -> dict:
    video, model_path = Path(video).expanduser().resolve(), Path(model_path).expanduser().resolve()
    output_dir = Path(output_dir).expanduser().resolve() / video.stem
    if not model_path.is_file():
        raise FileNotFoundError(
            f"YOLOv8n model not found: {model_path}. Use src/yolo/train_clean.py for training."
        )
    if not 0.0 <= conf <= 1.0:
        raise ValueError("conf must be between 0 and 1")

    from src.yolo.release_pipeline import ReleaseDetector
    from src.yolo.temporal import draw_candidates

    if frames < 1:
        raise ValueError("frames must be at least 1")
    output_dir.mkdir(parents=True, exist_ok=True)
    dev = resolve_device(device)
    detector = ReleaseDetector(model_path, conf, dev)
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0 or total < 1:
        capture.release()
        raise ValueError(f"Could not read video: {video}")
    stride = max(1, round(fps / 15))
    events, frame_reports, scanned = [], [], 0
    index = -1
    try:
        while capture.grab():
            index += 1
            if index % stride:
                continue
            ok, frame = capture.retrieve()
            if not ok:
                raise ValueError(f"Could not decode frame {index}")
            scanned += 1
            found = detector.update(frame, index/fps, index)
            events.extend(found)
            if found and len(frame_reports) < frames:
                annotated = output_dir / f"release_v1_{index:06d}.jpg"
                if not cv2.imwrite(str(annotated), draw_candidates(frame, found)):
                    raise OSError(f"Could not save {annotated}")
                frame_reports.append(dict(frame_number=len(frame_reports)+1,
                                          source_frame_index=index, timestamp_seconds=round(index/fps, 3),
                                          detections=found, annotated_frame=str(annotated)))
    finally:
        capture.release()
    if not frame_reports:
        # Keep clear previews useful without drawing unrelated static objects.
        samples, _ = sample_video_frames(video, min(frames, total))
        for number, (index, seconds, frame) in enumerate(samples, 1):
            annotated = output_dir / f"clear_v1_{index:06d}.jpg"
            if not cv2.imwrite(str(annotated), frame):
                raise OSError(f"Could not save {annotated}")
            frame_reports.append(dict(frame_number=number, source_frame_index=index,
                                      timestamp_seconds=round(seconds, 3), detections=[],
                                      annotated_frame=str(annotated)))
    return dict(video=str(video), model=str(model_path), device=dev,
                task="temporal_release_candidates", algorithm_version="release_v1",
                confidence_threshold=conf, frames_sampled=len(frame_reports), frames_scanned=scanned,
                source_fps=fps, scan_fps=fps/stride, events=events,
                frames_with_detections=sum(bool(r["detections"]) for r in frame_reports),
                total_detections=len(events),
                strongest_detection=max(events, key=lambda e: e.get("confidence") or 0, default=None),
                output_dir=str(output_dir), frame_results=frame_reports,
                note="Motion + person/vehicle separation + settling. Candidates require review; not a trained action verdict. Only object-confirmed boxes get YOLO scores.")


def detect_image(
    image: str | Path,
    *,
    model_path: str | Path = DEFAULT_MODEL,
    conf: float = DEFAULT_CONFIDENCE,
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
        raise FileNotFoundError(f"YOLOv8n model not found: {model_path}. Use src/yolo/train_clean.py for training.")
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
    """Run video scanning or image detection."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command")

    pred_base = argparse.ArgumentParser(add_help=False)
    pred_base.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    pred_base.add_argument("--conf", type=float, default=DEFAULT_CONFIDENCE)
    pred_base.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default="auto")

    vid_p = sub.add_parser("predict", parents=[pred_base], help="Track release candidates throughout an MP4 video")
    vid_p.add_argument("video", type=Path)
    vid_p.add_argument("--frames", type=int, default=12)
    vid_p.add_argument("--output-dir", type=Path, default=DEFAULT_VIDEO_OUTPUT)

    img_p = sub.add_parser("image", parents=[pred_base], help="Detect litter-like items in one image")
    img_p.add_argument("image", type=Path)
    img_p.add_argument("--output-dir", type=Path, default=DEFAULT_IMAGE_OUTPUT)

    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
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
