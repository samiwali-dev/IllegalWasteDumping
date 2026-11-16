"""Shared video inference for the three saved action classifiers."""
from __future__ import annotations

import math
import tempfile
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
CLIP_LENGTH = 24
IMAGE_SIZE = 112
CHECKPOINT_PATHS = {
    "optical_flow_svm": PROJECT_ROOT / "src/optical_flow_svm/model.joblib",
    "neural_network": PROJECT_ROOT / "src/neural_network/model.pt",
    "movinet_a0": PROJECT_ROOT / "src/movinet/model.pt",
}


def _tensor_from_clip(torch, clip: np.ndarray):
    if clip.shape != (CLIP_LENGTH, IMAGE_SIZE, IMAGE_SIZE, 3) or clip.dtype != np.uint8:
        raise ValueError(
            f"Expected uint8 RGB clip [{CLIP_LENGTH},{IMAGE_SIZE},{IMAGE_SIZE},3]; "
            f"got {clip.shape}, {clip.dtype}"
        )
    return torch.from_numpy(np.ascontiguousarray(clip)).permute(0, 3, 1, 2).float().div_(255.0)


def analyze_video(video, model_names, *, device="auto", progress=None):
    """Extract one shared clip and classify it with the selected saved models."""
    import time
    import cv2
    from data import IMAGE_SIZE, SEQUENCE_LENGTH, load_clip, extract_video

    video = Path(video).expanduser().resolve()
    if not video.is_file() or video.suffix.lower() != ".mp4":
        raise ValueError("Select an existing MP4 video.")
    model_names = list(model_names)
    if (not model_names or len(set(model_names)) != len(model_names)
            or any(name not in CHECKPOINT_PATHS for name in model_names)):
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
        elif name == "neural_network":
            label, score = predict_neural_network(clip, checkpoint)
            kind = "Uncalibrated sigmoid (threshold 0.5)"
        else:
            label, score, _ = predict_torch(clip, checkpoint, name, device)
            kind = "Uncalibrated sigmoid (threshold 0.5)"
        rows.append({"model": name, "prediction": label, "score": score,
                     "score_kind": kind, "seconds": time.perf_counter() - started})
    return rows


def predict_svm(clip, checkpoint: Path) -> tuple[int, float]:
    from src.optical_flow_svm.main import predict_clip

    return predict_clip(clip, checkpoint)


def predict_neural_network(clip, checkpoint: Path) -> tuple[int, float]:
    from src.neural_network.main import predict_clip

    return predict_clip(clip, checkpoint)


def predict_torch(clip, checkpoint: Path, model_name: str, requested_device: str) -> tuple[int, float, str]:
    if model_name != "movinet_a0":
        raise ValueError(f"Unknown PyTorch model: {model_name}")
    import torch
    from src.movinet.main import build_model

    if requested_device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; use --device cpu")
    device = torch.device(
        "mps" if requested_device == "auto" and torch.backends.mps.is_available()
        else "cpu" if requested_device == "auto" else requested_device
    )
    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    if 'state_dict' in saved:
        architecture = saved.get('architecture', {})
        if architecture.get('name') != 'MoViNet A0' or architecture.get('num_classes') != 1:
            raise ValueError('Expected a binary MoViNet A0 checkpoint')
        model = build_model(tf_like=bool(architecture.get('tf_like', False))).to(device)
        state = saved['state_dict']
    else:
        model = build_model().to(device)
        state = saved
    model.load_state_dict(state)
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
