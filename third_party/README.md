MoViNet A0 architecture code is bundled as a small local Python wheel so the
project does not fetch pretrained weights when dependencies are installed.

Source: https://github.com/Atze00/MoViNet-pytorch
Pinned source commit: c2d1edf48fc6c5259707f9d833f22171b4f63493
License: MIT (included inside the wheel).

The wheel contains only `movinets` Python files and package metadata. It has no
model checkpoint or pretrained weights. The training script always constructs
MoViNet with `pretrained=False` and `num_classes=1`.
