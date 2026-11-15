"""Train and independently compare the litter-object experiment.

Historical models and reports remain intact. A candidate is never promoted here.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import time

import cv2
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/litter_objects_v1"
RESULTS = ROOT / "results/yolo/object_v1"


def add_reviewed_negatives():
    review_path = DATA / "negative_review/review.json"
    if not review_path.exists():
        # Rebuild only the explicitly reviewed crops, never infer a new empty label.
        records = json.loads((RESULTS/"negative_review_decisions.json").read_text())
        review_path.parent.mkdir(parents=True, exist_ok=True)
        for item in records:
            source = ROOT/item["source_relative_file"]
            if hashlib.sha256(source.read_bytes()).hexdigest() != item["source_sha256"]:
                raise ValueError(f"Reviewed negative source changed: {source}")
            x1, y1, x2, y2 = item["crop_xyxy"]
            target = review_path.parent/f"{item['id']}.jpg"
            cv2.imwrite(str(target), cv2.imread(str(source))[y1:y2, x1:x2])
            item.update(file=str(target), source_file=str(source))
        review_path.write_text(json.dumps(records, indent=2)+"\n")
    reviewed = json.loads(review_path.read_text())
    # Actor/clothing groups selected before any model predictions are examined.
    test = {"N01", "N18", "N21", "N38", "N44", "N05", "N19", "N37"}
    val = {"N11", "N25", "N26"}
    counts = Counter()
    accepted = []
    files = {split: [line for line in (DATA / f"{split}.txt").read_text().splitlines() if "/negative_" not in line]
             for split in ("train", "val", "test")}
    for item in reviewed:
        if item["decision"] != "accept_background":
            continue
        split = "test" if item["id"] in test else "val" if item["id"] in val else "train"
        path = DATA / "images" / split / f"negative_{item['id']}.jpg"
        if not path.exists():
            path.hardlink_to(item["file"])
        (DATA / "labels" / split / f"negative_{item['id']}.txt").write_text("")
        files[split].append(str(path))
        accepted.append({**item, "split": split, "prepared_file": str(path)})
        counts[split] += 1
    for split, paths in files.items():
        (DATA / f"{split}.txt").write_text("\n".join(sorted(set(paths)))+"\n")
    report = {"counts": dict(counts), "records": accepted,
              "scope": "Reviewed clothing/person crops. Related source environments; this is not an independent indoor deployment benchmark."}
    (RESULTS / "negative_dataset.json").write_text(json.dumps(report, indent=2)+"\n")
    print("Reviewed negative images:", dict(counts), flush=True)


def split_paths(data_file, split):
    configuration = yaml.safe_load(Path(data_file).read_text())
    base = Path(configuration.get("path", DATA))
    return [Path(line) for line in (base/configuration[split]).read_text().splitlines()]


def verify_dataset(data_file=DATA/"data.yaml"):
    by_split = {}
    for split in ("train", "val", "test"):
        files = split_paths(data_file, split)
        by_split[split] = set()
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(path)
            label = DATA / "labels" / split / f"{path.stem}.txt"
            if not label.is_file():
                raise ValueError(f"Missing explicit label: {path}")
            by_split[split].add(hashlib.sha256(path.read_bytes()).hexdigest())
            for line in label.read_text().splitlines():
                category, cx, cy, width, height = map(float, line.split())
                if category != 0 or not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < width <= 1 and 0 < height <= 1):
                    raise ValueError(f"Invalid object label in {label}")
        print(split, len(files), "images", flush=True)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        if by_split[left] & by_split[right]:
            raise ValueError(f"Image duplicate between {left} and {right}")


def train(args):
    if (RESULTS/"training"/args.name).exists():
        raise FileExistsError("Experiment already exists; choose a new --name to preserve its checkpoint and configuration")
    if args.data == DATA/"data.yaml":
        add_reviewed_negatives()
    verify_dataset(args.data)
    initial = args.initial
    default_weights = DATA/"source/yolov8n.pt"
    if Path(initial) == default_weights:
        if not default_weights.exists():
            default_weights.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "90",
                            "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt",
                            "--output", str(default_weights)], check=True)
        if hashlib.sha256(default_weights.read_bytes()).hexdigest() != "f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36":
            raise ValueError("The official pretrained checkpoint checksum does not match the recorded source")
    from_weights = initial.endswith(".pt")
    configuration = {
        "task": "litter object localization; not action recognition",
        "initial": initial, "uses_pretrained_weights": from_weights,
        "data": str(args.data), "epochs": args.epochs,
        "imgsz": 640, "batch": args.batch, "seed": 42, "device": args.device,
        "learning_rate": args.learning_rate, "patience": args.patience,
        "selection": "Best validation checkpoint; test split is not used for model selection",
    }
    (RESULTS / f"{args.name}_configuration.json").write_text(json.dumps(configuration, indent=2)+"\n")
    model = YOLO(initial)
    started = time.perf_counter()
    model.train(data=str(args.data), epochs=args.epochs, imgsz=640,
                batch=args.batch, workers=0, cache=False, device=args.device,
                project=str(RESULTS / "training"), name=args.name, exist_ok=False,
                pretrained=from_weights, seed=42, deterministic=True,
                optimizer="AdamW", lr0=args.learning_rate, patience=args.patience,
                amp=False, plots=True, close_mosaic=5, verbose=True)
    configuration["training_seconds"] = time.perf_counter() - started
    configuration["best_checkpoint"] = str(model.trainer.best)
    (RESULTS / f"{args.name}_configuration.json").write_text(json.dumps(configuration, indent=2)+"\n")
    print(json.dumps(configuration, indent=2), flush=True)


def iou(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = max(0, a[2]-a[0])*max(0, a[3]-a[1]) + max(0, b[2]-b[0])*max(0, b[3]-b[1]) - intersection
    return intersection/union if union else 0


def operating_metrics(model, paths, device):
    counts = Counter()
    rows = []
    def predictions():
        # A list source may be treated as one batch by Ultralytics; bound it explicitly.
        for start in range(0, len(paths), 8):
            yield from model.predict(source=[str(path) for path in paths[start:start+8]], imgsz=640, conf=.25,
                                     device=device, batch=8, stream=True, verbose=False)
    for result in predictions():
        path = Path(result.path)
        label = DATA / "labels/test" / f"{path.stem}.txt"
        ground_truth = []
        for line in label.read_text().splitlines():
            _, x, y, w, h = map(float, line.split())
            ground_truth.append([x-w/2, y-h/2, x+w/2, y+h/2])
        predictions = result.boxes.xyxyn.detach().cpu().tolist() if result.boxes is not None else []
        scores = result.boxes.conf.detach().cpu().tolist() if result.boxes is not None else []
        matched = set()
        tp = 0
        for box in predictions:
            choices = [(iou(box, target), index) for index, target in enumerate(ground_truth) if index not in matched]
            overlap, index = max(choices, default=(0, -1))
            if overlap >= .5:
                matched.add(index)
                tp += 1
        fp, fn = len(predictions)-tp, len(ground_truth)-tp
        counts.update(tp=tp, fp=fp, fn=fn, images=1)
        if not ground_truth:
            counts.update(negative_images=1, negative_images_with_detection=int(bool(predictions)))
        rows.append({"file": path.name, "tp": tp, "fp": fp, "fn": fn,
                     "predicted_boxes": predictions, "scores": scores, "ground_truth": ground_truth})
    report = dict(counts)
    report.update(precision=counts["tp"]/max(1, counts["tp"]+counts["fp"]),
                  recall=counts["tp"]/max(1, counts["tp"]+counts["fn"]),
                  confidence_threshold=.25, match_iou=.5)
    return report, rows


def compare(args):
    verify_dataset(args.data)
    paths = split_paths(args.data, "test")
    report = {"task": "New litter-object task, different from the historical actor-region task",
              "test_policy": "Held-out TACO source batches and reviewed negative crops; no live-camera accuracy claim",
              "data_yaml": str(args.data),
              "benchmark_note": "After v1, the TACO test benchmark was reused during development. Hard-negative stress cases are not a population false-alarm estimate.",
              "dataset_manifest_sha256": hashlib.sha256((RESULTS/"dataset.json").read_bytes()).hexdigest(), "models": {}}
    for name, checkpoint in ((args.baseline_label, args.baseline), ("object_candidate", args.candidate)):
        model = YOLO(str(checkpoint))
        validation = model.val(data=str(args.data), split="test", imgsz=640,
                               batch=8, device=args.device, workers=0, plots=False,
                               project=str(RESULTS/"evaluation"), name=name, exist_ok=True,
                               conf=.001, verbose=False)
        point, rows = operating_metrics(model, paths, args.device)
        predictions_file=RESULTS/f"{Path(args.output).stem}_{name}_predictions.json"
        predictions_file.write_text(json.dumps(rows, indent=2)+"\n")
        report["models"][name] = {"checkpoint": str(checkpoint), "map50": float(validation.box.map50),
                                   "map50_95": float(validation.box.map), "operating_point": point,
                                   "predictions_file": str(predictions_file)}
        print(name, json.dumps(report["models"][name]), flush=True)
    (RESULTS / args.output).write_text(json.dumps(report, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fitting = commands.add_parser("train")
    fitting.add_argument("--initial", default=str(DATA/"source/yolov8n.pt"))
    fitting.add_argument("--data", type=Path, default=DATA/"data.yaml")
    fitting.add_argument("--learning-rate", type=float, default=.001)
    fitting.add_argument("--patience", type=int, default=10)
    fitting.add_argument("--epochs", type=int, default=40)
    fitting.add_argument("--batch", type=int, default=8)
    fitting.add_argument("--name", default="objects_scratch_v1")
    fitting.add_argument("--device", default="mps")
    evaluation = commands.add_parser("compare")
    evaluation.add_argument("--candidate", type=Path, required=True)
    evaluation.add_argument("--baseline", type=Path, default=ROOT/"src/yolo/model.pt")
    evaluation.add_argument("--baseline-label", default="historical_actor_detector")
    evaluation.add_argument("--data", type=Path, default=DATA/"data.yaml")
    evaluation.add_argument("--output", default="comparison.json")
    evaluation.add_argument("--device", default="mps")
    args = parser.parse_args()
    train(args) if args.command == "train" else compare(args)


if __name__ == "__main__":
    main()
