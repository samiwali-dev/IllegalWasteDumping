"""Prepare a separate, traceable litter-object dataset from official TACO labels.

Does not modify the historical person/action-region labels or existing models.
All 60 source categories map to one object class. No label is invented.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/litter_objects_v1"
REPORTS = ROOT / "results/yolo/object_v1"


def object_labels(image, annotations):
    """Convert the existing COCO object boxes to normalized YOLO boxes."""
    width, height = image["width"], image["height"]
    rows = []
    for annotation in annotations:
        x, y, w, h = annotation["bbox"]
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(width, x + w), min(height, y + h)
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Invalid annotation {annotation['id']}")
        rows.append(f"0 {(x1+x2)/2/width:.8f} {(y1+y2)/2/height:.8f} {(x2-x1)/width:.8f} {(y2-y1)/height:.8f}")
    if not rows:
        raise ValueError("An unannotated image must not silently become a negative")
    return rows


def phash(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    low = cv2.dct(cv2.resize(gray, (32, 32)).astype(np.float32))[:8, :8].flatten()
    median = np.median(low[1:])
    return int("".join("1" if x > median else "0" for x in low), 2)


def download_image(image, directory):
    output = directory / f"taco_{image['id']:04d}.jpg"
    url = image.get("flickr_640_url") or image.get("flickr_url")
    try:
        if not output.is_file():
            response = subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error",
                                       "--max-time", "25", "--retry", "1", "--max-filesize", "20000000", url],
                                      capture_output=True, check=True)
            content = response.stdout
            if len(content) > 20_000_000:
                raise ValueError("Image exceeds the download size limit")
            decoded = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError("Image cannot be decoded")
            output.write_bytes(content)
        content = output.read_bytes()
        decoded = cv2.imdecode(np.frombuffer(content, np.uint8), cv2.IMREAD_COLOR)
        if decoded is None:
            raise ValueError("Cached image cannot be decoded")
        h, w = decoded.shape[:2]
        if abs((w / h) / (image["width"] / image["height"]) - 1) > 0.015:
            raise ValueError("Thumbnail aspect ratio disagrees with the original annotations")
        return {"id": image["id"], "source_file": image["file_name"], "source_url": url,
                "source_width": image["width"], "source_height": image["height"],
                "width": w, "height": h, "license_id": image.get("license"),
                "sha256": hashlib.sha256(content).hexdigest(), "phash": f"{phash(decoded):016x}",
                "file": str(output)}
    except Exception as error:
        return {"id": image["id"], "source_url": url, "error": str(error)}


def assign_splits(records, seed=42):
    """Keep source batches and exact/near duplicate images in the same split."""
    parent = {record["source_file"].split("/")[0]: record["source_file"].split("/")[0] for record in records}
    def find(key):
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key
    def union(a, b):
        parent[find(a)] = find(b)
    near_pairs = []
    for i, left in enumerate(records):
        for right in records[i + 1:]:
            if left["sha256"] == right["sha256"] or (int(left["phash"], 16) ^ int(right["phash"], 16)).bit_count() <= 4:
                union(left["source_file"].split("/")[0], right["source_file"].split("/")[0])
                near_pairs.append([left["id"], right["id"]])
    groups = defaultdict(list)
    for record in records:
        groups[find(record["source_file"].split("/")[0])].append(record)
    ordered = list(groups.values())
    random.Random(seed).shuffle(ordered)
    ordered.sort(key=len, reverse=True)
    targets = {"train": len(records) * .70, "val": len(records) * .15, "test": len(records) * .15}
    counts = Counter()
    for group in ordered:
        split = max(targets, key=lambda key: (targets[key] - counts[key]) / targets[key])
        for record in group:
            record["split"] = split
        counts[split] += len(group)
    if any(counts[split] == 0 for split in targets):
        raise ValueError("Not enough independent source groups to create three splits")
    return counts, near_pairs


def contact_sheet(records, labels_by_id, output, count=24):
    selected = sorted(records, key=lambda row: row["id"])[::max(1, len(records)//count)][:count]
    sheet = np.full((int(np.ceil(len(selected)/4))*220, 4*320, 3), 245, np.uint8)
    for index, record in enumerate(selected):
        image = cv2.imread(record["file"])
        h, w = image.shape[:2]
        scale = min(320/w, 190/h)
        tile = cv2.resize(image, (round(w*scale), round(h*scale)))
        th, tw = tile.shape[:2]
        y, x = (index//4)*220, (index%4)*320
        sheet[y:y+th, x:x+tw] = tile
        for line in labels_by_id[record["id"]]:
            _, cx, cy, bw, bh = map(float, line.split())
            cv2.rectangle(sheet, (x+int((cx-bw/2)*tw), y+int((cy-bh/2)*th)),
                          (x+int((cx+bw/2)*tw), y+int((cy+bh/2)*th)), (40,180,30), 1)
        cv2.putText(sheet, f"TACO {record['id']} / {record['split']}", (x+5, y+207), cv2.FONT_HERSHEY_SIMPLEX, .45, (30,30,30), 1)
    cv2.imwrite(str(output), sheet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    cache = DATA / "source/images"
    cache.mkdir(parents=True, exist_ok=True)
    source_path = DATA / "source/taco_annotations.json"
    if not source_path.exists():
        temporary = source_path.with_suffix(".download")
        subprocess.run(["curl", "--fail", "--location", "--silent", "--show-error", "--max-time", "90",
                        "https://raw.githubusercontent.com/pedropro/TACO/master/data/annotations.json",
                        "--output", str(temporary)], check=True)
        json.loads(temporary.read_text())
        temporary.replace(source_path)
    source = json.loads(source_path.read_text())
    annotations = defaultdict(list)
    for annotation in source["annotations"]:
        annotations[annotation["image_id"]].append(annotation)
    by_id = {image["id"]: image for image in source["images"]}
    labels = {key: object_labels(image, annotations[key]) for key, image in by_id.items()}
    probe = download_image(source["images"][0], cache)
    if probe.get("error"):
        raise RuntimeError(f"Dataset download preflight failed: {probe['error']}")
    records, failures = [], []
    started = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download_image, image, cache) for image in source["images"]]
        for done, future in enumerate(as_completed(futures), 1):
            result = future.result()
            (failures if result.get("error") else records).append(result)
            if done % 50 == 0 or done == len(futures):
                print(f"Downloaded/checked {done}/{len(futures)}; usable {len(records)}; failed {len(failures)}; {time.time()-started:.0f}s", flush=True)
    counts, near_pairs = assign_splits(records)
    split_files = defaultdict(list)
    for record in records:
        split = record["split"]
        images = DATA / "images" / split
        target_labels = DATA / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        target_labels.mkdir(parents=True, exist_ok=True)
        name = Path(record["file"]).name
        destination = images / name
        original = cv2.imread(record["file"])
        oh, ow = original.shape[:2]
        factor = min(1, 960 / max(oh, ow))
        prepared = cv2.resize(original, (round(ow*factor), round(oh*factor)), interpolation=cv2.INTER_AREA) if factor < 1 else original
        ok, encoded = cv2.imencode(".jpg", prepared, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise ValueError(f"Could not prepare {record['file']}")
        temporary = destination.with_suffix(".tmp")
        temporary.write_bytes(encoded.tobytes())
        temporary.replace(destination)
        record["prepared_width"], record["prepared_height"] = prepared.shape[1], prepared.shape[0]
        record["prepared_sha256"] = hashlib.sha256(encoded.tobytes()).hexdigest()
        split_files[split].append(str(destination))
        (target_labels / f"{Path(name).stem}.txt").write_text("\n".join(labels[record["id"]])+"\n")
        record["object_count"] = len(labels[record["id"]])
        record["annotation_ids"] = [annotation["id"] for annotation in annotations[record["id"]]]
    for split, files in split_files.items():
        (DATA / f"{split}.txt").write_text("\n".join(sorted(files))+"\n")
    (DATA / "data.yaml").write_text(f"path: {json.dumps(str(DATA))}\ntrain: train.txt\nval: val.txt\ntest: test.txt\nnames:\n  0: litter_object\n")
    report = {"source": "Official TACO annotations (pedropro/TACO)", "source_url": "https://github.com/pedropro/TACO",
              "annotation_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
              "task": "litter object localization, not throwing action recognition", "source_category_count": len(source["categories"]),
              "mapping": "All 60 source object categories to class 0 litter_object", "seed": 42,
              "preprocessing": "Aspect-preserving resize to at most 960 pixels; JPEG quality 90; normalized boxes unchanged",
              "split_policy": "Source batch groups; exact/near duplicates (pHash distance <=4) stay in the same split",
              "counts": dict(counts), "objects": sum(record["object_count"] for record in records),
              "near_duplicate_pairs": near_pairs, "download_failures": failures,
              "human_review": "Source annotations are manual TACO labels; local contact-sheet review is recorded separately",
              "image_rights": "Original per-image license metadata retained; no private images uploaded or published",
              "records": sorted(records, key=lambda row: row["id"])}
    (REPORTS / "dataset.json").write_text(json.dumps(report, indent=2)+"\n")
    contact_sheet(records, labels, REPORTS / "object_label_review.jpg")
    print(json.dumps({"counts": dict(counts), "objects": report["objects"], "failures": len(failures)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
