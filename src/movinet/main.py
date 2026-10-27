"""Train or test the existing MoViNet A0 video classifier.

Examples:
    .venv/bin/python src/movinet/main.py train --help
    .venv/bin/python src/movinet/main.py predict path/to/video.mp4

This uses the unofficial MoViNet-pytorch implementation at commit
c2d1edf48fc6c5259707f9d833f22171b4f63493. Its A0 configuration and existing
checkpoint remain unchanged. New training uses random initialization.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ALGORITHM_DIR = Path(__file__).resolve().parent
MODEL_PATH = ALGORITHM_DIR / "model.pt"
SLUG = "movinet_a0"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import make_parser, run_experiment


ARCHITECTURE = {
    "name": "MoViNet A0",
    "implementation": "unofficial PyTorch MoViNet-pytorch",
    "source": "https://github.com/Atze00/MoViNet-pytorch",
    "source_commit": "c2d1edf48fc6c5259707f9d833f22171b4f63493",
    "configuration": "MoViNetA0",
    "causal": False,
    "pretrained": False,
    "num_classes": 1,
    "conv_type": "3d",
    "tf_like": False,
    "initialization": "random",
}


def build_model():
    try:
        from movinets import MoViNet
        from movinets.config import _C
    except ImportError as exc:
        raise RuntimeError(
            "The unofficial PyTorch 'movinets' package is required for MoViNet A0. "
            "Install MoViNet-pytorch at commit "
            "c2d1edf48fc6c5259707f9d833f22171b4f63493."
        ) from exc

    from torch import nn

    class ActionMoViNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = MoViNet(
                _C.MODEL.MoViNetA0,
                causal=False,
                pretrained=False,
                num_classes=1,
                conv_type="3d",
                tf_like=False,
            )

        def forward(self, clips):
            # Shared loader emits [batch, time, RGB, height, width].
            # MoViNet expects [batch, RGB, time, height, width].
            return self.network(clips.permute(0, 2, 1, 3, 4).contiguous()).squeeze(-1)

    return ActionMoViNet()


def predict_clip(clip, checkpoint=None, device="auto"):
    """Return the saved model's label, sigmoid score, and device for one clip."""
    from common import predict_torch

    selected = MODEL_PATH if checkpoint is None else Path(checkpoint)
    return predict_torch(clip, selected, SLUG, device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser(
        "train", parents=[make_parser("Train MoViNet A0")], add_help=False,
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
