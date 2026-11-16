# Temporal release candidates

The application now analyzes a continuous motion sequence for a release instead
of treating isolated YOLO object boxes as throwing evidence. This shared engine
runs for recorded videos and live camera sessions. It is a **motion heuristic
with pretrained actor verification**, not a newly trained action model.

## Processing

1. Scan recorded video at up to 15 FPS, preserving aspect ratio at a maximum
   width of 960. The previous path inspected only 12 widely spaced stills.
2. Maintain background change components. Link compact components by appearance
   and forward/backward Lucas–Kanade optical-flow continuity.
3. Require a short movement followed by settling, a genuinely changed landing
   region, and restoration of the earlier position toward its background.
4. Verify a person/vehicle at the separation point with official COCO YOLOv8n.
   A nearby parked vehicle alone is insufficient. Reject components still at a
   person or vehicle, including sleeves, feet and carried items.
5. Check the tracked object's local crop with the existing litter-object model.
   Only an overlapping prediction above the saved object-model threshold gets
   an actual YOLO score. Otherwise the amber box says `Release candidate`, with
   no percentage. Person/vehicle confidence is never used as litter confidence.

The output box surrounds the tracked component. The arrow uses its observed
trajectory. Blue static object detections are no longer returned as release
candidates. Static image detection remains an independent object-detection task.
The litter-object weights and their previously reported benchmark have not
changed in this correction. Neither have the three video-classifier weights.

## Live state

Each monitoring session has a distinct ID and increasing camera timestamps.
The API retains one bounded tracker across requests, skips duplicate frames in
overlapping windows, and resets its background after camera movement or missing
frame history. This fixes the earlier loss of temporal context at every request.
Sessions expire after 120 seconds of inactivity; at most four are retained.
Snapshots use the exact source frame corresponding to the returned candidate.

The browser captures evidence at up to 960 px width, keeping the full aspect
ratio. The existing local five-guard fallback calculations are unchanged. It
remains separate from the Python release tracker; it is not a YOLO confidence.

## Weights

`src/yolo/actors.pt` is the official COCO YOLOv8n checkpoint already downloaded
for the previous authorized transfer-learning experiment. It recognizes only
person, bicycle, car, motorcycle, bus and truck here. Its source and SHA256:

- https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt
- `f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36`

`src/yolo/model.pt` remains the experimental TACO litter-object model with SHA256
`c52231bc969212c332142569d52b08ce93485b0daead33f3bc42f0438b039472`.
Actor score threshold 0.35 and motion parameters are development heuristics, not
calibrated action probabilities or independent learned event thresholds.

## Observed checks

These clips and parameters were inspected repeatedly during development. This
is a regression check, **not an independent estimate of event accuracy**.

| Existing test clip | Expected case | Release candidates |
|---|---|---:|
| vid0012 | Thrown dark object in roadway | 1 |
| vid0005 | No throwing | 0 |
| vid0018 | No throwing | 0 |
| vid0021 | No throwing | 0 |
| vid0038 | No throwing | 0 |
| vid0040 | No throwing | 0 |
| vid0043 | No throwing | 0 |

For vid0012, the returned region is around the actual dark object at source
frame 196, approximately 3.27 s. It is not on the stationary blue cover. A replay
through the live API using 960 px JPEGs, increasing timestamps and overlapping
24-frame requests returned the same single release candidate, without duplicate
snapshots. The object head did not confirm this bag: its displayed score is
therefore absent rather than invented. See `video_checks.json`,
`vid0012_report.json` and `live_replay_check.json`.

Seven temporal/state unit tests cover a static dark item, walking, actual motion
then settling, camera pan, illumination changes, overlapping live requests and
separate camera sessions. The five dataset tests and 17 web tests pass, and Expo
web export passes. The full Python suite still has the three pre-existing
`TwoModelGuiTests` failures about the former two-classifier desktop interface.

## Limits

Candidates still require visual review. The tracker can miss tiny/occluded
objects, throws without settling, objects already present when monitoring starts,
stationary-vehicle throws without a visible moving source, and events during
camera motion or lost frame history. A dropped item is not inherently litter or
an illegal act. No claim is made that all official samples or live conditions
are solved. The object detector's old 81.3% precision / 32.3% recall are **object
benchmark metrics**, not performance measurements for this temporal pipeline.

## Reproduce

```sh
.venv/bin/python -m unittest test_release_tracking test_object_dataset
PYTORCH_ENABLE_MPS_FALLBACK=1 .venv/bin/python src/yolo/evaluate_releases.py --device mps
.venv/bin/python src/yolo/main.py predict 'datasets/MIVIA-IWDD 500/test/videos/vid0012.mp4'
```

Generated annotated frames remain local and ignored by Git. Reports contain
candidate geometry and source provenance, not uploaded dataset images.
