# Litter-object detector experiment

The active experimental YOLOv8n now localizes the litter object itself, using
new training data and new weights. The historical model mainly learned actor or
action regions. This experiment does **not** retrain the MLP, MoViNet or SVM video
classifiers and does not establish that a detected object was thrown.

## Data and review

- [Official TACO](https://github.com/pedropro/TACO): 1,500 images with 4,784
  manually annotated litter instances. All 60 categories map to `litter_object`.
  Boxes come from the source COCO annotations, never from historical person boxes.
- A 24-image contact sheet was inspected locally. Other labels retain the
  official manual-annotation provenance; all 1,500 were not re-reviewed locally.
- Of 48 candidate person/clothing crops, 27 were accepted as background. Held
  objects and ambiguous crops were excluded. Decisions and source checksums are
  in `negative_review_decisions.json`.
- Twenty-eight frames from four existing MIVIA training clips were reviewed for
  new object boxes. Objects were too small or occluded for reliable labels, so
  these frames were excluded. Existing action labels were left unchanged.
- After v1, 136 model-proposed crops from existing MIVIA videos were visually
  reviewed. Sixty-three clear non-litter crops (cars, road markings, shadows,
  people, signs, maintained containers) were accepted after duplicate checks.
  Real litter and uncertain crops were excluded. A `noThrow` video was never
  assumed to contain no litter. See `hard_negative_review_decisions.json`.

TACO source batches stay together; exact/near duplicates (pHash distance <= 4)
cannot cross splits. Related actor crops stay together. Hard-negative source
videos were assigned to separate train/validation/test roles before mining.
Explicit lists define which files are used; unrelated directory files are ignored.

| Source | Train | Validation | Test |
|---|---:|---:|---:|
| TACO images | 1,002 | 212 | 286 |
| Reviewed person/clothing backgrounds | 16 | 3 | 8 |
| Reviewed mined backgrounds (v2) | 37 | 14 | 12 |
| Unique images in v2 | **1,055** | **229** | **306** |

The v2 training list repeats the 16 clothing backgrounds ten times and the 37
mined backgrounds four times: 1,310 exposures, **not** 1,310 independent images.
Test contains 1,131 objects and 20 background images. The negative stress set is
small and selected; it is not a deployment false-alarm-rate estimate.

Raw photos, crops, image license metadata and private user screenshots remain
local in ignored paths. Source URLs, checksums, annotation IDs, preprocessing and
split membership are in `dataset.json`. Prepared images preserve aspect ratio,
with maximum side 960 px. Source licenses vary and may be missing; no dataset
photos are published with this repository.

## Actual training

The user authorized pretrained initialization. `pretrained_source.json` records
the official COCO YOLOv8n checkpoint URL and SHA256.

| Run | Initialization | Epochs | Image size / batch | AdamW learning rate | Time |
|---|---|---:|---|---:|---:|
| v1 | COCO YOLOv8n | 40 | 640 / 8 | 0.001 | 1,546.95 s |
| v2 | v1 best | 15 | 640 / 8 | 0.0003 | 708.86 s |

Both use MPS, seed 42, workers 0 and validation checkpoint selection. MPS warned
that some deterministic kernels are unavailable; bitwise repeatability is not
claimed. Configuration reports and local run CSVs retain the actual settings.

## Results and selection

Both historical and new models were evaluated on the **same new object task**.
The historical actor-task mAP of 54.29% is a different benchmark and must not be
compared directly with the object-task mAP below.

On the 306-image object benchmark, at score 0.25 and matching IoU 0.5:

| Model | Precision | Recall | mAP50 | Background images with boxes / 20 |
|---|---:|---:|---:|---:|
| Historical actor detector | 2.70% | 0.09% | 0.55% | 1 |
| v1 object detector | 57.21% | 43.85% | 44.16% | 12 |
| v2 object detector | 55.27% | 42.18% | 45.00% | 7 |

V2 reduces flagged background stress images compared with v1, but slightly
reduces overall precision/recall at 0.25. It is not an unqualified improvement.

`calibrate_objects.py` selects v2 best/last and a threshold using **validation
only**: maximize F0.5 multiplied by background rejection, with recall >= 0.35,
over scores 0.10 through 0.80 at steps of 0.05. This selects a score threshold,
not calibrated probabilities. The chosen checkpoint is **v2 best, threshold
0.65**. Square 640 inference is used by selection and live API.

| Chosen operating point | Precision | Recall | TP / FP / FN | Background images with boxes |
|---|---:|---:|---|---:|
| Validation (229 images) | 83.39% | 43.50% | 251 / 50 / 326 | 0 / 17 |
| Development test (306 images) | **81.29%** | **32.27%** | **365 / 84 / 766** | **2 / 20** |

Test mAP50 is **44.995%**, mAP50:95 **30.708%** (Ultralytics validation integrates
score thresholds). Fixed-point precision/recall are separate IoU-0.5 matching
counts. **The test benchmark was inspected after v1 and reused during
development. These are development results, not a fresh blind test.**

`selected_video_checks.json` records qualitative checks on 12 existing MIVIA
clips, 12 sampled frames each. Counts are not throwing-event accuracy: static
litter can be present in a no-throw video. Visual review still found false
positives on a TV logo and a blue cover, and missed small objects. In the user's
already-annotated room screenshot the original strap is not detected at the
chosen threshold, but another chair region is falsely detected. That screenshot
and its strap crop were not used in training and are not clean camera benchmarks.

**Low recall and the indoor camera domain gap remain unresolved.** This model
cannot promise to catch every drop. More representative labeled object examples
and temporally labeled drop/hold/walk clips are needed for reliable event claims.

## Application integration

`src/yolo/model.pt` contains the chosen new weights; its SHA256 is recorded in
`results/yolo/yolov8n_run.json`. The old weights/report are preserved locally, and
the historical tracked weights can also be recovered from commit `a1309c6`.

The live API uses the saved validation-selected threshold for real YOLO output.
Blue item boxes keep the actual uncalibrated detection score. Object presence
alone no longer triggers an automatic throwing snapshot: the model box must
overlap a recent falling trajectory. Motion-only captures retain their separate
label. This association is a motion heuristic, not a newly trained action model.
The existing five-guard motion calculations are unchanged.

## Reproduce

Original MIVIA and Roboflow files are required to recreate the reviewed negative
crops. Checksums reject changed inputs. Use new experiment names to retain the
completed checkpoints; the training script refuses to overwrite an existing run.

```sh
.venv/bin/python src/yolo/prepare_objects.py
.venv/bin/python src/yolo/train_objects.py train --name objects_transfer_v1
.venv/bin/python src/yolo/mine_object_backgrounds.py --export-reviewed
.venv/bin/python src/yolo/train_objects.py train --name objects_transfer_v2 \
  --initial results/yolo/object_v1/training/objects_transfer_v1/weights/best.pt \
  --data datasets/litter_objects_v1/data_v2.yaml --epochs 15 \
  --learning-rate 0.0003 --patience 6
.venv/bin/python src/yolo/calibrate_objects.py
```

Run `train_objects.py compare --help` for same-task comparison arguments. Training
and calibration scripts do not replace the active model automatically.

## Verification

Five dataset/box tests and 16 web motion/capture tests pass; Expo web export
passes. Three existing `TwoModelGuiTests` in `test_project.py` fail because they
expect only two classifiers while the existing desktop app exposes three.
`app.py` and those GUI tests were not changed for this experiment.
