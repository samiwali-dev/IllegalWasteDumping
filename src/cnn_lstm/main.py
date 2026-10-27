"""Train or test the existing CNN-LSTM video classifier.

Examples:
    .venv/bin/python src/cnn_lstm/main.py train --help
    .venv/bin/python src/cnn_lstm/main.py predict path/to/video.mp4

Training starts from random initialization. The 24-frame architecture is
unchanged so the existing trained model.pt can be used directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALGORITHM_DIR = Path(__file__).resolve().parent
MODEL_PATH = ALGORITHM_DIR / "model.pt"
SLUG = "cnn_lstm"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import make_parser, run_experiment


ARCHITECTURE = {
    "name": "CNN-LSTM",
    "spatial_channels": [12, 24, 48, 64],
    "spatial_kernel": 3,
    "spatial_stride": 2,
    "temporal_model": "one-layer LSTM",
    "lstm_hidden_size": 96,
    "dropout": 0.25,
}


def build_model():
    import torch
    from torch import nn

    class CNNLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 12, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(3, 12), nn.ReLU(inplace=True),
                nn.Conv2d(12, 24, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(4, 24), nn.ReLU(inplace=True),
                nn.Conv2d(24, 48, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(6, 48), nn.ReLU(inplace=True),
                nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
                nn.GroupNorm(8, 64), nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
            )
            self.temporal = nn.LSTM(input_size=64, hidden_size=96, batch_first=True)
            self.classifier = nn.Sequential(
                nn.Dropout(0.25), nn.Linear(96, 32), nn.ReLU(inplace=True),
                nn.Linear(32, 1),
            )

        def forward(self, clips: torch.Tensor) -> torch.Tensor:
            # clips: [batch, time=24, channel=3, height=112, width=112]
            batch, time, channels, height, width = clips.shape
            features = self.encoder(clips.reshape(batch * time, channels, height, width))
            features = features.reshape(batch, time, -1)
            _, (hidden, _) = self.temporal(features)
            return self.classifier(hidden[-1]).squeeze(-1)

    return CNNLSTM()


def predict_clip(clip, checkpoint=None, device="auto"):
    """Return the saved model's label, sigmoid score, and device for one clip."""
    from common import predict_torch

    selected = MODEL_PATH if checkpoint is None else Path(checkpoint)
    return predict_torch(clip, selected, SLUG, device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser(
        "train", parents=[make_parser("Train CNN-LSTM")], add_help=False,
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
