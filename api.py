"""HTTP API Server for the Expo Web & Mobile App.

Run with:
    .venv/bin/python api.py --port 8000

Provides CORS-enabled endpoints for:
- GET  /api/health       -> Server and model readiness
- GET  /api/evaluation   -> Saved 4-algorithm comparison (model_comparison.json)
- GET  /api/samples      -> Official MIVIA test videos for quick 1-tap testing
- POST /api/analyze      -> Analyze an uploaded MP4 video or sample video with all 4 models
- GET  /api/annotated/*  -> Serve YOLOv8n annotated frame images
"""

from __future__ import annotations

import argparse
import base64
from email.parser import BytesParser
from email.policy import default as default_email_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import tempfile
import time
import traceback
from urllib.parse import unquote, urlparse

import cv2
import numpy as np

from common import (
    CHECKPOINT_PATHS,
    CLIP_LENGTH,
    IMAGE_SIZE,
    PROJECT_ROOT,
    analyze_video,
    predict_neural_network,
    predict_svm,
    predict_torch,
)
from data import load_manifest, split_records

EVALUATION_PATH = PROJECT_ROOT / "results" / "comparison" / "model_comparison.json"
YOLO_EVALUATION_PATH = PROJECT_ROOT / "results" / "yolo" / "yolov8n_run.json"
YOLO_MODEL = PROJECT_ROOT / "src" / "yolo" / "model.pt"
YOLO_VIDEO_OUTPUT = PROJECT_ROOT / "results" / "yolo" / "video_predictions"
YOLO_CONFIDENCE = float(json.loads(YOLO_EVALUATION_PATH.read_text()).get("confidence_threshold", 0.25))
if not 0 < YOLO_CONFIDENCE < 1:
    raise ValueError("YOLO confidence_threshold must be between 0 and 1")

CLASSIFIER_LABELS = {
    "neural_network": "Neural Network (MLP)",
    "movinet_a0": "MoViNet A0",
    "optical_flow_svm": "Optical Flow + SVM",
}
ALL_MODELS = ("neural_network", "movinet_a0", "optical_flow_svm", "yolov8n")


def _resolve_record_video_path(rec: dict) -> Path:
    """Resolve the MP4 video path for a MIVIA manifest record."""
    if "video_path" in rec:
        return Path(rec["video_path"])
    return PROJECT_ROOT / "datasets" / "MIVIA-IWDD 500" / rec["split"] / "videos" / rec["video_id"]


def get_sample_videos() -> list[dict]:
    """Return every available official MIVIA test video."""
    try:
        _, _, test_records = split_records(load_manifest())
        samples = []
        for rec in sorted(test_records, key=lambda record: record["video_id"]):
            vpath = _resolve_record_video_path(rec)
            if vpath.is_file():
                samples.append({
                    "video_id": rec["video_id"],
                    "label": int(rec["label"]),
                    "label_name": "dumping" if int(rec["label"]) == 1 else "no_dumping",
                    "filename": vpath.name,
                })
        return samples
    except Exception:
        return []


def run_analysis(video_path: Path, selected_models: list[str], host_url: str) -> dict:
    """Run selected video classifiers and/or YOLOv8n on an MP4 video."""
    classifiers = [m for m in selected_models if m in CLASSIFIER_LABELS]
    include_yolo = "yolov8n" in selected_models

    actions = []
    yolo_report = None
    errors = []

    if classifiers:
        available = []
        for name in classifiers:
            if CHECKPOINT_PATHS[name].is_file():
                available.append(name)
            else:
                errors.append(f"{CLASSIFIER_LABELS[name]}: model file missing")
        if available:
            try:
                raw_rows = analyze_video(video_path, available)
                for row in raw_rows:
                    actions.append({
                        **row,
                        "label_title": CLASSIFIER_LABELS.get(row["model"], row["model"]),
                        "verdict": "DUMPING" if row["prediction"] == 1 else "NO DUMPING",
                    })
            except Exception as exc:
                errors.append(f"Classification error: {exc}")

    if include_yolo:
        if not YOLO_MODEL.is_file():
            errors.append(f"YOLOv8n: model missing at {YOLO_MODEL}")
        else:
            try:
                from src.yolo.main import detect_video

                report = detect_video(video_path, model_path=YOLO_MODEL, frames=12, conf=YOLO_CONFIDENCE)
                stem = Path(report["output_dir"]).name
                frames_with_urls = []
                for f in report.get("frame_results", []):
                    fname = Path(f["annotated_frame"]).name
                    frames_with_urls.append({
                        **f,
                        "image_url": f"{host_url}/api/annotated/{stem}/{fname}",
                    })
                yolo_report = {
                    **report,
                    "frame_results": frames_with_urls,
                }
            except Exception as exc:
                errors.append(f"YOLOv8n: {exc}")

    return {
        "video_name": video_path.name,
        "actions": actions,
        "yolo": yolo_report,
        "errors": errors,
    }


def _letterbox_rgb(bgr_img: np.ndarray, size: int = IMAGE_SIZE) -> np.ndarray:
    h, w = bgr_img.shape[:2]
    scale = min(size / w, size / h)
    tw, th = max(1, round(w * scale)), max(1, round(h * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(bgr_img, (tw, th), interpolation=interp)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    top = (size - th) // 2
    left = (size - tw) // 2
    canvas[top : top + th, left : left + tw] = resized[:, :, ::-1]
    return canvas


def run_live_frames_analysis(raw_frames: list[str], selected_models: list[str]) -> dict:
    """Analyze base64 camera frames with the 3 video classifiers and/or YOLOv8n."""
    decoded_bgr: list[np.ndarray] = []
    for item in raw_frames:
        if not item:
            continue
        b64_part = item.split(",", 1)[-1] if "," in item else item
        raw_bytes = base64.b64decode(b64_part)
        arr = np.frombuffer(raw_bytes, dtype=np.uint8)
        bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if bgr is not None:
            decoded_bgr.append(bgr)

    if not decoded_bgr:
        raise ValueError("Could not decode any camera frames.")

    classifiers = [m for m in selected_models if m in CLASSIFIER_LABELS]
    include_yolo = "yolov8n" in selected_models
    actions = []
    yolo_live = None
    errors = []

    if classifiers:
        indices = [
            round(i * (len(decoded_bgr) - 1) / max(1, CLIP_LENGTH - 1))
            for i in range(CLIP_LENGTH)
        ]
        clip = np.stack([_letterbox_rgb(decoded_bgr[idx], IMAGE_SIZE) for idx in indices], axis=0)
        for name in classifiers:
            checkpoint = CHECKPOINT_PATHS[name]
            if not checkpoint.is_file():
                errors.append(f"{CLASSIFIER_LABELS[name]}: model missing")
                continue
            try:
                t0 = time.perf_counter()
                if name == "optical_flow_svm":
                    label, score = predict_svm(clip, checkpoint)
                    kind = "SVM decision score (threshold 0)"
                elif name == "neural_network":
                    label, score = predict_neural_network(clip, checkpoint)
                    kind = "Uncalibrated sigmoid (threshold 0.5)"
                else:
                    label, score, _ = predict_torch(clip, checkpoint, name, "auto")
                    kind = "Uncalibrated sigmoid (threshold 0.5)"
                actions.append({
                    "model": name,
                    "label_title": CLASSIFIER_LABELS[name],
                    "prediction": label,
                    "verdict": "DUMPING" if label == 1 else "NO DUMPING",
                    "score": score,
                    "score_kind": kind,
                    "seconds": time.perf_counter() - t0,
                })
            except Exception as exc:
                errors.append(f"{CLASSIFIER_LABELS[name]}: {exc}")

    if include_yolo and YOLO_MODEL.is_file():
        try:
            from src.yolo.main import _extract_detections, _get_yolo, resolve_device

            YOLO = _get_yolo()
            dev = resolve_device("auto")
            latest_bgr = decoded_bgr[-1]
            ih, iw = latest_bgr.shape[:2]
            res = YOLO(str(YOLO_MODEL)).predict(
                source=latest_bgr, imgsz=640, rect=False, conf=YOLO_CONFIDENCE, device=dev, verbose=False, save=False
            )[0]
            detections = [
                {
                    **d,
                    "norm_box": [
                        d["bbox_xyxy"][0] / max(1, iw),
                        d["bbox_xyxy"][1] / max(1, ih),
                        d["bbox_xyxy"][2] / max(1, iw),
                        d["bbox_xyxy"][3] / max(1, ih),
                    ],
                }
                for d in _extract_detections(res)
            ]
            plotted = latest_bgr.copy()
            color = (250, 165, 96)  # Same blue used by the live YOLO overlay (BGR).
            for detection in detections:
                x1, y1, x2, y2 = [int(v) for v in detection["bbox_xyxy"]]
                label = "item" if detection["class_name"] == "litter_object" else detection["class_name"]
                text = f"YOLO {label} {detection['confidence']:.0%}"
                scale = max(0.45, iw / 1200)
                (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
                label_x = max(0, min(x1, iw - tw - 8))
                label_y = max(th + 8, y1)
                cv2.rectangle(plotted, (x1, y1), (x2, y2), color, 2)
                cv2.rectangle(plotted, (label_x, label_y - th - 8), (label_x + tw + 8, label_y), color, -1)
                cv2.putText(plotted, text, (label_x + 4, label_y - 4), cv2.FONT_HERSHEY_SIMPLEX, scale, (63, 36, 16), 1, cv2.LINE_AA)
            ok, buf = cv2.imencode(".jpg", plotted, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            annotated_b64 = (
                "data:image/jpeg;base64," + base64.b64encode(buf.tobytes()).decode("ascii")
                if ok
                else None
            )
            strongest = max(detections, key=lambda d: d["confidence"], default=None)
            yolo_live = {
                "total_detections": len(detections),
                "detections": detections,
                "strongest_detection": strongest,
                "annotated_data_uri": annotated_b64,
                "frames_buffered": len(decoded_bgr),
                "confidence_threshold": YOLO_CONFIDENCE,
                "task": "object_detection",
            }
        except Exception as exc:
            errors.append(f"YOLOv8n Live: {exc}")

    return {
        "video_name": f"Live Camera Stream ({len(decoded_bgr)} frames)",
        "actions": actions,
        "yolo_live": yolo_live,
        "errors": errors,
    }


class ApiHandler(BaseHTTPRequestHandler):
    def _set_cors(self, status: int = 200, content_type: str = "application/json; charset=utf-8") -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Accept")
        self.end_headers()

    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self._set_cors(status=status)
        self.wfile.write(body)

    def do_OPTIONS(self) -> None:
        self._set_cors(204)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        route = unquote(parsed.path)

        if route == "/api/health":
            self._send_json({
                "status": "ok",
                "models": list(ALL_MODELS),
                "classifier_labels": CLASSIFIER_LABELS,
            })
            return

        if route == "/api/evaluation":
            try:
                data = json.loads(EVALUATION_PATH.read_text(encoding="utf-8"))
                self._send_json(data)
            except Exception as exc:
                self._send_json({"error": f"Could not load evaluation report: {exc}"}, status=500)
            return

        if route == "/api/samples":
            self._send_json({"samples": get_sample_videos()})
            return

        if route.startswith("/api/sample-video/"):
            vid_id = Path(route[len("/api/sample-video/"):]).name
            _, _, test_records = split_records(load_manifest())
            match = next((r for r in test_records if r["video_id"] == vid_id), None)
            if match:
                vpath = _resolve_record_video_path(match)
                if vpath.is_file():
                    data = vpath.read_bytes()
                    self._set_cors(200, "video/mp4")
                    self.wfile.write(data)
                    return
            self._send_json({"error": "Sample video not found"}, status=404)
            return

        if route.startswith("/api/annotated/"):
            rel = route[len("/api/annotated/"):]
            parts = [p for p in Path(rel).parts if p not in ("", ".", "..")]
            if len(parts) == 2:
                img_path = (YOLO_VIDEO_OUTPUT / parts[0] / parts[1]).resolve()
                if img_path.is_file() and YOLO_VIDEO_OUTPUT.resolve() in img_path.parents:
                    data = img_path.read_bytes()
                    self._set_cors(200, "image/jpeg")
                    self.wfile.write(data)
                    return
            self._send_json({"error": "Annotated image not found"}, status=404)
            return

        # Serve Expo Web production bundle from expo-app/dist if available
        dist_dir = (PROJECT_ROOT / "expo-app" / "dist").resolve()
        if dist_dir.is_dir():
            rel_route = "index.html" if route in ("", "/") else route.lstrip("/")
            target = (dist_dir / rel_route).resolve()
            if target.is_file() and dist_dir in target.parents:
                ext = target.suffix.lower()
                mime = {
                    ".html": "text/html; charset=utf-8",
                    ".js": "application/javascript; charset=utf-8",
                    ".css": "text/css; charset=utf-8",
                    ".json": "application/json; charset=utf-8",
                    ".png": "image/png",
                    ".jpg": "image/jpeg",
                    ".ico": "image/x-icon",
                }.get(ext, "application/octet-stream")
                self._set_cors(200, mime)
                self.wfile.write(target.read_bytes())
                return

        self._send_json({"error": f"Unknown endpoint: {route}"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path not in ("/api/analyze", "/api/analyze-live"):
            self._send_json({"error": "Not found"}, status=404)
            return

        if parsed.path == "/api/analyze-live":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8") if length > 0 else "{}")
                frames = body.get("frames") or []
                requested = body.get("models") or list(ALL_MODELS)
                selected_models = [m for m in requested if m in ALL_MODELS] or list(ALL_MODELS)
                result = run_live_frames_analysis(frames, selected_models)
                self._send_json(result)
            except Exception as exc:
                traceback.print_exc()
                self._send_json({"error": str(exc)}, status=500)
            return

        host_header = self.headers.get("Host", "localhost:8000")
        host_url = f"http://{host_header}"
        content_type = self.headers.get("Content-Type", "")

        try:
            if "multipart/form-data" in content_type:
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length)
                mime_Block = f"Content-Type: {content_type}\r\n\r\n".encode("utf-8") + raw_body
                msg = BytesParser(policy=default_email_policy).parsebytes(mime_Block)

                fields: dict[str, str] = {}
                video_bytes: bytes | None = None
                orig_name = "uploaded.mp4"

                for part in msg.iter_parts():
                    name = part.get_param("name", header="content-disposition")
                    filename = part.get_filename()
                    payload = part.get_payload(decode=True) or b""
                    if name == "video" or filename:
                        video_bytes = payload
                        if filename:
                            orig_name = Path(filename).name
                    elif name:
                        fields[name] = payload.decode("utf-8", errors="replace")

                raw_models = fields.get("models", "all")
                selected_models = (
                    list(ALL_MODELS)
                    if not raw_models or raw_models == "all"
                    else [m.strip() for m in raw_models.split(",") if m.strip() in ALL_MODELS]
                )
                if not selected_models:
                    selected_models = list(ALL_MODELS)

                if not video_bytes:
                    self._send_json({"error": "Missing 'video' file field in form data"}, status=400)
                    return

                if not orig_name.lower().endswith(".mp4"):
                    orig_name = f"{Path(orig_name).stem}.mp4"

                with tempfile.TemporaryDirectory(prefix="mivia-expo-") as tmpdir:
                    temp_video = Path(tmpdir) / orig_name
                    temp_video.write_bytes(video_bytes)
                    result = run_analysis(temp_video, selected_models, host_url)
                self._send_json(result)
                return

            # JSON request (for sample videos)
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length).decode("utf-8") if length > 0 else "{}")
            sample_id = body.get("sample_video_id")
            requested = body.get("models") or list(ALL_MODELS)
            selected_models = [m for m in requested if m in ALL_MODELS] or list(ALL_MODELS)

            if not sample_id:
                self._send_json({"error": "Provide a video file or sample_video_id"}, status=400)
                return

            _, _, test_records = split_records(load_manifest())
            match = next((r for r in test_records if r["video_id"] == sample_id), None)
            if not match:
                self._send_json({"error": f"Unknown sample_video_id: {sample_id}"}, status=404)
                return

            video_path = _resolve_record_video_path(match)
            result = run_analysis(video_path, selected_models, host_url)
            result["ground_truth_label"] = int(match["label"])
            result["ground_truth_verdict"] = "DUMPING" if int(match["label"]) == 1 else "NO DUMPING"
            self._send_json(result)

        except Exception as exc:
            traceback.print_exc()
            self._send_json({"error": str(exc)}, status=500)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the Illegal Waste Dumping API server for Expo.")
    parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind (default: 0.0.0.0 for LAN/Mobile access)")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    args = parser.parse_args()

    server = ThreadingHTTPServer((args.host, args.port), ApiHandler)
    print(f"API Server listening on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
