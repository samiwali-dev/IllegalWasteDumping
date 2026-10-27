"""Train or test the existing 3D-CNN video classifier.

Examples:
    .venv/bin/python src/cnn_3d/main.py train --help
    .venv/bin/python src/cnn_3d/main.py predict path/to/video.mp4

The model learns from ordered 24-frame RGB clips. Its architecture is unchanged
and training starts from random initialization without pretrained weights.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALGORITHM_DIR = Path(__file__).resolve().parent
MODEL_PATH = ALGORITHM_DIR / "model.pt"
SLUG = "cnn_3d"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import make_parser, run_experiment


ARCHITECTURE = {
    "name": "3D-CNN",
    "channels": [12, 24, 48, 64],
    "kernel": [3, 3, 3],
    "strides_time_height_width": [[1, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]],
    "dropout": 0.25,
}


def build_model():
    import torch
    from torch import nn

    class CNN3D(nn.Module):
        def __init__(self):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv3d(3, 12, kernel_size=3, stride=(1, 2, 2), padding=1),
                nn.GroupNorm(3, 12), nn.ReLU(inplace=True),
                nn.Conv3d(12, 24, kernel_size=3, stride=(2, 2, 2), padding=1),
                nn.GroupNorm(4, 24), nn.ReLU(inplace=True),
                nn.Conv3d(24, 48, kernel_size=3, stride=(2, 2, 2), padding=1),
                nn.GroupNorm(6, 48), nn.ReLU(inplace=True),
                nn.Conv3d(48, 64, kernel_size=3, stride=(2, 2, 2), padding=1),
                nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool3d(1), nn.Flatten(),
            )
            self.classifier = nn.Sequential(nn.Dropout(0.25), nn.Linear(64, 1))

        def forward(self, clips: torch.Tensor) -> torch.Tensor:
            # Input [batch,time,channels,height,width]; Conv3d expects channels first.
            volume = clips.permute(0, 2, 1, 3, 4).contiguous()
            return self.classifier(self.features(volume)).squeeze(-1)

    return CNN3D()


def predict_clip(clip, checkpoint=None, device="auto"):
    """Return the saved model's label, sigmoid score, and device for one clip."""
    from common import predict_torch

    selected = MODEL_PATH if checkpoint is None else Path(checkpoint)
    return predict_torch(clip, selected, SLUG, device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser(
        "train", parents=[make_parser("Train 3D-CNN")], add_help=False,
        help="Train from scratch using the shared MIVIA protocol",
    )
    training.set_defaults(models_dir=ALGORITHM_DIR)
    prediction = commands.add_parser("predict", help="Classify an MP4 video")
    prediction.add_argument("video", type=Path)
    prediction.add_argument("--device", choices=("auto", "mps", "cpu"), default="auto")
    args = parser.parse_args()
    if args.command == "train":
        if args.checkpoint is None:
            args.checkpoint = args.models_dir / "model.pt"
        run_experiment(args, slug=SLUG, model_factory=build_model, architecture=ARCHITECTURE)
    else:
        from common import analyze_video

        print(json.dumps(analyze_video(args.video, [SLUG], device=args.device), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
