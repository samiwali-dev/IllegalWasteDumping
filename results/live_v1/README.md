# Short-window live candidates

These checkpoints were trained for 1.4-second, 24-frame, RGB 112×112 letterboxed windows. They are separate from the original whole-video classifiers in `src/*/model.*`. Their validation metrics must not be substituted for official whole-video test metrics.

## Data boundary

Only the official training partition was used for fitting and model selection. Fitting has 480 samples: 320 real windows (160 positive, 160 negative) and 160 frozen-image negative sequences. Validation has 78 real windows (38 positive, 40 negative) plus 80 frozen-image negative sequences. Two positive videos with invalid event timestamps were excluded from window construction, not relabeled. Windows use existing diagnostic event timestamps, whose timing precision is limited.

The official test partition was not decoded or used for this live training, tuning, or diagnostic. The user camera screenshot was excluded from fitting and checkpoint selection. No screenshot imagery is stored in this results directory.

## Validation results

Selected checkpoints use the existing 0.5 sigmoid threshold or zero SVM decision threshold. Selection first requires recall ≥0.5 and static false positives ≤0.1, then maximizes real-window macro F1 × (1 − static false positive rate). These are development validation results, not an independent final test.

| Candidate | Real accuracy | Macro F1 | Positive recall | Frozen-image false positives |
|---|---:|---:|---:|---:|
| MoViNet A0 (epoch 4) | 80.77% | 80.69% | 89.47% | 1/80 (1.25%) |
| Optical flow SVM | 46.15% | 46.12% | 50.00% | 0/80 (0%) |
| MLP (epoch 19) | 62.82% | 62.67% | 71.05% | 0/80 (0%) |

SVM and MLP remain weak on real windows. Frozen-image negatives do not cover moving indoor cameras, walking people, held objects, lighting changes, or other real live failure modes.

## Single screenshot diagnostic

`diagnostic.json` records the raw predictions, checkpoint hashes, and preprocessing. The camera crop was letterboxed once and repeated unchanged 24 times. It therefore contains no temporal movement. This is a post-training sanity check of one static scene, not measured indoor camera accuracy.

| Model | Original whole-video prediction / score | Live candidate prediction / score |
|---|---|---|
| MoViNet A0 | 0 / 0.47027156 | 0 / 8.9538e−10 |
| Optical flow SVM | 1 / +1.03743387 | 0 / −1.09970129 |
| MLP | 1 / 0.69844270 | 0 / 0.24391206 |

Label 0 means no dumping; label 1 means dumping. SVM values are decision scores (threshold 0); other values are uncalibrated sigmoid scores (threshold 0.5), not calibrated confidence. All candidates return no dumping for this static input. This does **not** establish that indoor accuracy or indoor false positives are solved. Record representative held-out moving camera sessions before making that claim.

Python verification: `.venv/bin/python -m unittest discover -v` passed all 43 tests. No official test videos were decoded for this QA.
