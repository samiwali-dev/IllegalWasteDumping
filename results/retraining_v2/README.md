# Fresh initialization comparison

This experiment separates **video classification**, **item localization**, and
**release tracking**. A correct video verdict does not establish that a box
marks the thrown item. Object confidence is not a throwing probability.

## Data boundary

- Action fit: 320 official-train videos. Validation: 80 other official-train
  videos, balanced 40 dumping / 40 no dumping. Seed 42, split by video ID.
- No official test video is used for this round's training, annotation, threshold
  selection, or tracking development. `training_protocol.py` enforces the boundary.
- New object fit: 1,049 distinct images: 1,002 TACO fit images, 37 reviewed MIVIA
  backgrounds, and 10 assistant-reviewed MIVIA item crops. Validation: 230 images
  (212 TACO validation, 14 backgrounds, four item crops). Repeated fit samples
  provide 1,193 training exposures; validation images are not repeated.
- The MIVIA object annotations are a **small pilot**, not exhaustive object labels
  for all 320 fit videos, and have not been independently human verified.
- TACO held-out images, all official MIVIA test sources, and legacy Roboflow
  crops with uncertain video provenance are excluded from new object training.
- Image/label SHA256 and actual YOLO input lists are verified against
  [the dataset manifest](yolo/dataset.json). MIVIA videos are disjoint, but camera
  locations can recur across videos. This is not a scene-independent benchmark.

## Action model

Both new MoViNet A0 strategies use the same 320/80 videos, 24 uniformly sampled
RGB frames at 112 square, full-image letterboxing, and a fixed 0.5 threshold.
The fresh transfer model starts from Kinetics600 weights with a new binary head;
the scratch model starts randomly. This compares the documented training budgets,
not the best accuracy achievable from scratch with unlimited data and compute.

| Initialization | Selected epoch | Validation accuracy | Validation macro F1 |
| --- | ---: | ---: | ---: |
| Fresh Kinetics600 transfer | 6 | 86.25% | 86.25% |
| Random initialization | 2 | 67.50% | 64.29% |

Transfer was selected on validation before a single report-only official-test
evaluation. Its weights and 0.5 threshold were frozen in
[frozen_selection.json](movinet/frozen_selection.json). The final 100-video report
is **81% accuracy**, precision 82.98%, recall 78%, F1 80.41%; confusion matrix
`[[42, 8], [11, 39]]`. The previous active model reported 72% on these same clips.
The report-only result did not change the selected weights or threshold.

The official test clips were exposed during **historical project development**.
Therefore this is an exploratory comparison, not a pristine independent test.
Neither validation nor official-test accuracy establishes indoor live-camera
performance. Live windows are also shorter than the training videos.

See [the initialization comparison](movinet/comparison.json),
[the report-only test](movinet/final_test.json), and the two strategy reports.
The active runtime supports the new architecture-tagged checkpoint and legacy
raw state dictionaries. A runtime prediction was checked against a saved
validation score; the live API also loaded the candidate without errors.

## Object model and tracking

`src/yolo/train_clean.py` compares a fresh COCO backbone with a random YOLOv8n
initialization. Neither starts from the project's previously fine-tuned head.
Both use the same frozen dataset, 20-epoch maximum, patience 6, seed 42, AdamW,
640 training resolution and batch 8. The existing active head is included as a
baseline, with its historical exposure recorded explicitly.

`src/yolo/select_clean.py` measures validation localization and selects a score
threshold by F0.5 multiplied by background-image rejection, requiring recall
of at least 0.35. Its 640 and 1280 results describe different inference sizes;
they are never mixed into one score. This is operating-point selection, not
probability calibration. Validation remains development data.

Release checks run across **all 80 validation videos**, without passing their
labels or annotated event times into either detector. Reports distinguish the
legacy motion tracker and experimental `object_tracks_v2`. Their binary summary
counts whether a video has any candidate. That measure can overstate success
when a box points to an unrelated object, and is not localization accuracy.

Visual review already found background association errors in the experimental
tracker, including static-pile matches and false candidates in a litter-picking
scene. Successful synthetic tests did not establish real-video reliability.
The rejected experimental tracker was **removed from the current code**, together with its runtime dispatch and synthetic tests. Its source and exact experiment runtime are retained in Git commit `a740cd469ec16bfddcd7f44adfcd95e30e48d52b`; reports also record source hashes. The development evaluator now runs only the active legacy tracker.

The completed 80-video check found legacy candidates in 1/40 positive videos
and 5/40 negative videos. The experimental variant produced candidates in 6/40
positive videos and 6/40 negative videos. Both miss most positive clips, and
some positive-video boxes still point to pre-existing objects. Full results
and visual-review decisions are in [tracking_comparison.json](tracking_comparison.json).

| YOLO head, 640 inference | Validation mAP50 | Precision at selected threshold | Recall | Background images flagged |
| --- | ---: | ---: | ---: | ---: |
| Existing head, retained | 54.05% | 83.39% | 43.13% | 0/14 |
| Fresh COCO transfer | 54.28% | 87.80% | 37.11% | 2/14 |
| Random initialization | 35.04% | 52.21% | 36.60% | 9/14 |

Both fresh YOLO runs completed 20 epochs. Transfer's slightly higher mAP50 did
not outweigh its lower recall and background rejection under the documented
selection rule. The existing weights are retained. These image metrics use the
same validation images; they are not comparable with the old 306-image test
numbers or with video classification accuracy.

YOLO training completion alone is not a reason to replace the active head.
Only measured improvements are promoted. This experiment does **not** establish
that thrown-item localization or the indoor false-positive problem is solved.

## Reproduction

Run from the repository root with the existing virtual environment and local
datasets. Training refuses to overwrite a completed run. Raw images, inactive
checkpoints and caches remain local; manifests, labels/provenance decisions,
configurations and reports are retained for review.

```sh
.venv/bin/python src/movinet/retrain.py --initialization transfer
.venv/bin/python src/movinet/retrain.py --initialization scratch
.venv/bin/python src/yolo/prepare_clean.py
.venv/bin/python src/yolo/train_clean.py --initialization transfer --epochs 20
.venv/bin/python src/yolo/train_clean.py --initialization scratch --epochs 20
.venv/bin/python src/yolo/select_clean.py --imgsz 640 --candidate legacy=src/yolo/model.pt --candidate transfer=results/retraining_v2/yolo/transfer/weights/best.pt --candidate scratch=results/retraining_v2/yolo/scratch/weights/best.pt
```

Do not run `prepare_clean.py` while a training job is using its frozen dataset.
The MoViNet final-test evaluator refuses to overwrite its report; it is not a
tool for choosing weights or repeated tuning against official test results.

## Code cleanup

The old desktop GUI, superseded object training/calibration/background-mining
runners, rejected item tracker and its dedicated tests were removed. Historical
MoViNet training was removed from `common.py`, which now contains inference only.
The old actor-region YOLO training entry point was removed from `main.py`.
Current training lives in `movinet/retrain.py` and `yolo/train_clean.py`.
The two required object-matching helpers live in `yolo/metrics.py`.
Historical reports and active model weights are preserved; old code is available
in Git history rather than as duplicate runnable files.
