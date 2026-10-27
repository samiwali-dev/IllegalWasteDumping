"""Reproducible, non-destructive cleaning for MIVIA and Roboflow Littering.

    .venv/bin/python clean.py
    .venv/bin/python clean.py --dataset mivia
    .venv/bin/python clean.py --dataset littering --audit-only

MIVIA checks reuse existing extracted frames and never change official splits.
Littering checks decode images, validate boxes, quarantine confirmed defects,
and optionally export a separate cleaned YOLO dataset. Similarity candidates
remain available for human review. Original files and saved models are kept.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import tempfile
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from data import SEED, SEQUENCE_LENGTH, split_records


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# COCO image/category lookup, shared by the Littering checks below.
def resolve_image(split_dir: Path, file_name: str) -> Path:
    supplied = Path(file_name)
    candidates = [split_dir / supplied, split_dir / supplied.name]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    matches = [p for p in split_dir.rglob(supplied.name) if p.is_file()]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Image {file_name!r} referenced by COCO JSON was not found in {split_dir}")
    raise RuntimeError(f"Image name {supplied.name!r} is ambiguous in {split_dir}")


def choose_categories(payload: dict, class_name: str) -> tuple[set[int], str]:
    categories = payload.get("categories")
    if not isinstance(categories, list) or not categories:
        raise ValueError("COCO file has no categories")
    wanted = class_name.strip().casefold()
    exact = [c for c in categories if str(c.get("name", "")).strip().casefold() == wanted]
    if exact:
        # This Roboflow export has two metadata categories with the same name.
        # Treat all identically named IDs as the one requested detection class.
        return {int(c["id"]) for c in exact}, str(exact[0]["name"])
    if len(categories) == 1:
        category = categories[0]
        return {int(category["id"])}, str(category["name"])
    names = [str(c.get("name")) for c in categories]
    raise ValueError(f"Could not find class {class_name!r}. COCO categories are: {names}")


# MIVIA validation: raw videos, strict labels, existing frames, exact duplicates.
def parse_label_data(data: Any) -> dict[str, Any]:
    """Parse labels strictly; timestamp defects do not change the action label."""
    result: dict[str, Any] = {
        "raw_label": None, "label": None, "timestamp": None,
        "timestamp_issues": [], "errors": [], "metadata": {},
    }
    if not isinstance(data, dict):
        result["errors"].append("label_json_not_object")
        return result
    value = data.get("label")
    result["raw_label"] = value
    if isinstance(value, bool):
        result["label"] = int(value)
    elif isinstance(value, str) and value.strip().lower() in {"true", "false"}:
        result["label"] = int(value.strip().lower() == "true")
    elif type(value) is int and value in (0, 1):
        result["label"] = value
    else:
        result["errors"].append("invalid_action_label")
    metadata = data.get("metadata", {})
    if isinstance(metadata, dict):
        result["metadata"] = metadata
    else:
        result["errors"].append("metadata_not_object")
    timestamp = data.get("timestamp")
    if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)):
        if result["label"] == 1 or timestamp is not None:
            result["timestamp_issues"].append("timestamp_missing_or_not_numeric")
    elif not math.isfinite(timestamp):
        result["timestamp_issues"].append("timestamp_not_finite")
    else:
        result["timestamp"] = float(timestamp)
        if result["label"] == 1 and timestamp < 0:
            result["timestamp_issues"].append("positive_timestamp_negative")
    return result


def decode_frame(path: Path) -> dict[str, Any]:
    """Decode one existing JPEG; an existing nonempty file can still be corrupt."""
    try:
        if not path.is_file() or path.stat().st_size == 0:
            return {"path": str(path), "valid": False, "error": "missing_or_empty_frame"}
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
            return {"path": str(path), "valid": False, "error": "frame_decode_failed"}
        height, width = frame.shape[:2]
        if height < 1 or width < 1:
            return {"path": str(path), "valid": False, "error": "frame_shape_invalid"}
        return {"path": str(path), "valid": True, "height": height, "width": width}
    except (OSError, cv2.error) as exc:
        return {"path": str(path), "valid": False, "error": f"frame_read_error: {exc}"}


def duplicate_groups(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Group exact video-byte duplicates, retaining split and label provenance."""
    by_hash: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("sha256"):
            by_hash[record["sha256"]].append(record)
    groups = []
    for digest, members in sorted(by_hash.items()):
        if len(members) < 2:
            continue
        original_splits = {member["original_split"] for member in members}
        model_splits = {member.get("model_split") for member in members}
        labels = {member.get("label") for member in members}
        groups.append({
            "sha256": digest,
            "members": [{key: member.get(key) for key in
                         ("video_id", "source_video", "original_split", "model_split", "label")}
                        for member in members],
            "crosses_official_split": len(original_splits) > 1,
            "crosses_model_split": len(model_splits) > 1,
            "conflicting_labels": len(labels) > 1,
        })
    return groups


def load_hash_cache(path: Path) -> dict[str, dict[str, Any]]:
    cached: dict[str, dict[str, Any]] = {}
    if path.is_file():
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    item = json.loads(line)
                    if isinstance(item, dict) and isinstance(item.get("path"), str):
                        cached[item["path"]] = item
                except (ValueError, TypeError):
                    # An interrupted final append must not invalidate prior hashes.
                    continue
    return cached


def hash_video(path: Path, cache: dict[str, dict[str, Any]], cache_path: Path) -> tuple[str, bool]:
    """Reuse SHA256 only if the source's path, size and nanosecond mtime match."""
    path_key = str(path.resolve())
    before = path.stat()
    previous = cache.get(path_key, {})
    digest = previous.get("sha256")
    if (previous.get("size_bytes") == before.st_size
            and previous.get("mtime_ns") == before.st_mtime_ns
            and isinstance(digest, str) and len(digest) == 64):
        return digest, True
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise OSError(f"Source changed during hashing: {path}")
    item = {"path": path_key, "size_bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns, "sha256": hasher.hexdigest()}
    needs_newline = False
    if cache_path.is_file() and cache_path.stat().st_size:
        with cache_path.open("rb") as handle:
            handle.seek(-1, 2)
            needs_newline = handle.read(1) != b"\n"
    with cache_path.open("a", encoding="utf-8") as handle:
        if needs_newline:
            # Preserve and isolate an interrupted record before the next append.
            handle.write("\n")
        handle.write(json.dumps(item, ensure_ascii=False) + "\n")
        handle.flush()
    cache[path_key] = item
    return item["sha256"], False


def _finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _video_header(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            return {"errors": ["source_video_open_failed"]}
        frame_count = capture.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = capture.get(cv2.CAP_PROP_FPS)
        ok, frame = capture.read()
        errors = []
        if not ok or frame is None or frame.size == 0:
            errors.append("source_first_frame_decode_failed")
        if not _finite_number(frame_count) or frame_count < SEQUENCE_LENGTH:
            errors.append("source_frame_count_invalid")
        if not _finite_number(fps) or fps <= 0:
            errors.append("source_fps_invalid")
        return {"errors": errors, "frame_count": int(frame_count) if _finite_number(frame_count) else None,
                "fps": fps if _finite_number(fps) else None}
    finally:
        capture.release()


def _audit_cached_record(
    root: Path, item: dict[str, Any], cached: dict[str, Any] | None,
) -> list[Path]:
    errors, warnings = item["errors"], item["warnings"]
    if cached is None:
        errors.append("missing_extraction_report_record")
        return []
    if cached.get("label") != item["label"] or type(cached.get("label")) is not int:
        errors.append("cached_label_disagrees_with_raw_json")
    if cached.get("split") != item["original_split"]:
        errors.append("cached_official_split_disagrees_with_source")
    expected_class = "dumping" if item["label"] else "no_dumping"
    if item["label"] is not None and cached.get("class_name") != expected_class:
        errors.append("cached_class_name_disagrees_with_raw_label")
    count = cached.get("source_frame_count")
    indices = cached.get("sampled_frame_indices_zero_based")
    if type(count) is not int or count < SEQUENCE_LENGTH:
        errors.append("cached_source_frame_count_invalid")
    if (not isinstance(indices, list) or len(indices) != SEQUENCE_LENGTH
            or any(type(index) is not int for index in indices)
            or any(left >= right for left, right in zip(indices, indices[1:]))):
        errors.append("cached_frame_indices_not_24_ordered_integers")
    elif type(count) is int and count >= SEQUENCE_LENGTH:
        expected = [round(i * (count - 1) / (SEQUENCE_LENGTH - 1)) for i in range(SEQUENCE_LENGTH)]
        if indices != expected:
            errors.append("cached_frame_indices_not_uniform_end_to_end")
    times = cached.get("sampled_times_seconds")
    if not isinstance(times, list) or len(times) != SEQUENCE_LENGTH:
        errors.append("cached_sample_times_count_invalid")
    elif any(value is not None and not _finite_number(value) for value in times):
        errors.append("cached_sample_times_not_finite")
    else:
        available = [value for value in times if value is not None]
        if any(left > right for left, right in zip(available, available[1:])):
            errors.append("cached_sample_times_not_ordered")
    header = item["source_header"]
    if header.get("frame_count") != count:
        warnings.append("source_header_frame_count_differs_from_cached_exact_count")
    if (_finite_number(header.get("fps")) and _finite_number(cached.get("source_fps"))
            and not math.isclose(header["fps"], cached["source_fps"], abs_tol=0.001)):
        errors.append("source_fps_changed_since_extraction")
    fps = cached.get("source_fps")
    if not _finite_number(fps) or fps <= 0:
        errors.append("cached_fps_invalid")
    elif type(count) is int and count > 0:
        item["cached_duration_seconds"] = count / fps
        if item["label"] == 1 and item["timestamp"] is not None:
            if not 0 <= item["timestamp"] <= count / fps:
                item["timestamp_issues"].append("event_timestamp_outside_video")
    default_dir = root / "datasets" / "MIVIA-Frames" / str(cached.get("split")) / str(cached.get("class_name")) / Path(item["video_id"]).stem
    recorded_dir = Path(cached["output_dir"]) if isinstance(cached.get("output_dir"), str) else None
    if recorded_dir is None:
        errors.append("cached_output_directory_missing")
        return []
    if recorded_dir.resolve() != default_dir.resolve():
        warnings.append("cached_output_path_differs_from_current_project_location")
    frame_dir = recorded_dir if recorded_dir.is_dir() else default_dir
    if not frame_dir.is_dir():
        errors.append("cached_frame_directory_missing")
        return []
    expected_names = [f"frame_{index:03d}.jpg" for index in range(1, SEQUENCE_LENGTH + 1)]
    actual_names = sorted(path.name for path in frame_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg"})
    if actual_names != expected_names:
        errors.append("cached_frame_filenames_or_count_invalid")
    item["cached_frame_directory"] = str(frame_dir)
    return [frame_dir / name for name in expected_names]


def audit_mivia(project_root: Path, output_dir: Path, workers: int = 4) -> dict[str, Any]:
    started = time.monotonic()
    root = project_root.resolve()
    source_root = root / "datasets" / "MIVIA-IWDD 500"
    report_path = root / "results" / "data" / "mivia_frames_report_24.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    if source_root == output_dir or source_root in output_dir.parents:
        raise ValueError("Audit outputs must not be written inside the raw dataset")
    with report_path.open(encoding="utf-8") as handle:
        extraction = json.load(handle)
    global_errors: list[str] = []
    global_warnings: list[str] = []
    if extraction.get("status") != "validated" or extraction.get("failures"):
        global_errors.append("existing_extraction_report_not_validated")
    if extraction.get("sequence_length") != SEQUENCE_LENGTH:
        global_errors.append("existing_extraction_sequence_length_not_24")
    if extraction.get("timestamp_used_for_sampling") is not False:
        global_errors.append("existing_extraction_used_event_timestamp_for_sampling")
    if extraction.get("source_dataset") != str(source_root):
        global_warnings.append("extraction_source_path_differs_from_current_source_root")
    cached_items = extraction.get("videos", [])
    if not isinstance(cached_items, list):
        raise ValueError("Existing extraction report has no video-record list")
    cached_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for cached in cached_items:
        if not isinstance(cached, dict):
            global_errors.append("extraction_contains_nonobject_record")
            continue
        key = (cached.get("split"), cached.get("video_id"))
        if key in cached_by_key:
            global_errors.append(f"duplicate_extraction_record:{key}")
        cached_by_key[key] = cached

    items: list[dict[str, Any]] = []
    orphan_labels: list[str] = []
    for split in ("train", "test"):
        video_dir, label_dir = source_root / split / "videos", source_root / split / "labels"
        if not video_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing source videos/labels: {source_root / split}")
        videos = sorted(path for path in video_dir.iterdir() if path.suffix.lower() == ".mp4")
        labels = {path.stem: path for path in label_dir.iterdir() if path.suffix.lower() == ".json"}
        video_stems = {path.stem for path in videos}
        orphan_labels.extend(str(path) for stem, path in labels.items() if stem not in video_stems)
        for video in videos:
            label_path = labels.get(video.stem)
            label = parse_label_data({})
            if label_path is None:
                label["errors"] = ["missing_label_json"]
            else:
                try:
                    with label_path.open(encoding="utf-8-sig") as handle:
                        label = parse_label_data(json.load(handle))
                except (OSError, ValueError) as exc:
                    label["errors"] = [f"label_json_unreadable:{exc}"]
            metadata = label.pop("metadata")
            item = {"video_id": video.name, "source_video": str(video),
                    "source_label": str(label_path) if label_path else None,
                    "original_split": split, "model_split": None,
                    **label, "warnings": [], "sha256": None,
                    "cached_frame_directory": None, "decoded_frames": 0,
                    "frame_failures": [], "split_review_required": False}
            if metadata.get("video_id") not in (None, video.name):
                item["errors"].append("label_metadata_video_id_mismatch")
            items.append(item)
    if orphan_labels:
        global_errors.append("orphan_json_labels_without_video")
    source_keys = {(item["original_split"], item["video_id"]) for item in items}
    orphan_cached = sorted(str(key) for key in cached_by_key.keys() - source_keys)
    if orphan_cached:
        global_errors.append("extraction_records_without_source_video")
    minimal_records = [{"video_id": item["video_id"], "split": item["original_split"], "label": item["label"]}
                       for item in items if item["label"] is not None]
    try:
        groups = split_records(minimal_records, seed=SEED)
        assignment = {record["video_id"]: split for split, records in
                      zip(("train", "validation", "test"), groups) for record in records}
        for item in items:
            item["model_split"] = assignment.get(item["video_id"])
    except ValueError as exc:
        global_errors.append(f"cannot_reproduce_existing_split:{exc}")

    cache_path = output_dir / "mivia_hash_cache.jsonl"
    cache = load_hash_cache(cache_path)
    hashes_reused = 0
    decoded_total = 0
    print(f"MIVIA audit: {len(items)} original videos; checking existing {SEQUENCE_LENGTH}-frame sequences, no extraction/training.", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, item in enumerate(items, 1):
            video = Path(item["source_video"])
            item["source_header"] = _video_header(video)
            item["errors"].extend(item["source_header"]["errors"])
            try:
                item["sha256"], reused = hash_video(video, cache, cache_path)
                hashes_reused += int(reused)
            except OSError as exc:
                item["errors"].append(f"source_hash_failed:{exc}")
            cached = cached_by_key.get((item["original_split"], item["video_id"]))
            frame_paths = _audit_cached_record(root, item, cached)
            results = list(pool.map(decode_frame, frame_paths))
            item["decoded_frames"] = sum(frame["valid"] for frame in results)
            item["frame_failures"] = [frame for frame in results if not frame["valid"]]
            if item["frame_failures"]:
                item["errors"].append("one_or_more_cached_frames_corrupt_or_missing")
            shapes = {(frame["height"], frame["width"]) for frame in results if frame["valid"]}
            item["frame_shapes_height_width"] = sorted(shapes)
            if len(shapes) > 1:
                item["warnings"].append("cached_frame_resolution_changes_within_video")
            decoded_total += item["decoded_frames"]
            if index == 1 or index % 25 == 0 or index == len(items):
                print(f"[{index}/{len(items)}] decoded JPEGs: {decoded_total}; hash cache reused: {hashes_reused}; elapsed {time.monotonic() - started:.1f}s", flush=True)

    duplicates = duplicate_groups(items)
    duplicate_by_source = {member["source_video"]: group for group in duplicates for member in group["members"]}
    for item in items:
        duplicate = duplicate_by_source.get(item["source_video"])
        if duplicate:
            item["warnings"].append("exact_duplicate_video_bytes")
            item["split_review_required"] = (duplicate["crosses_model_split"] or duplicate["conflicting_labels"])
        item["valid"] = not item["errors"]
        item["valid_for_training"] = item["valid"] and item["model_split"] == "train"
        item["issues"] = item["errors"] + item["warnings"] + item["timestamp_issues"]
    counts = Counter((item["original_split"], item["label"]) for item in items)
    if counts != Counter({("train", 0): 200, ("train", 1): 200, ("test", 0): 50, ("test", 1): 50}):
        global_errors.append("official_source_label_counts_not_200_200_50_50")
    split_counts = Counter(item["model_split"] for item in items)
    if split_counts != Counter({"train": 320, "validation": 80, "test": 100}):
        global_errors.append("existing_model_split_not_320_80_100")
    timestamp_warnings = [{"video_id": item["video_id"], "original_split": item["original_split"],
                           "timestamp": item["timestamp"], "issues": item["timestamp_issues"]}
                          for item in items if item["timestamp_issues"]]
    needs_split_review = any(item["split_review_required"] for item in items)
    technical_pass = not global_errors and all(item["valid"] for item in items)
    result = {
        "status": "passed_with_timestamp_warnings" if technical_pass and not needs_split_review and timestamp_warnings else
                  "passed" if technical_pass and not needs_split_review else "needs_review",
        "technical_validation_pass": technical_pass,
        "split_integrity_pass": not needs_split_review,
        "clean_pass": technical_pass and not needs_split_review,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_dataset": str(source_root), "existing_extraction_report": str(report_path),
        "policy": {
            "raw_files_modified": False, "frames_reextracted": False, "models_retrained": False,
            "official_split_preserved": True, "split_function": "data.split_records", "split_seed": SEED,
            "timestamp_used_for_training": False,
            "exclusion_rule": "Only record errors invalidate data; timestamp warnings alone never exclude a video.",
            "duplicates_policy": "Report duplicate groups for review; do not silently alter official train/test or remove originals.",
            "valid_for_training_definition": "Technically valid records assigned to fitting/train (validation/test retained separately).",
        },
        "summary": {
            "original_videos": len(items), "original_train_videos": sum(item["original_split"] == "train" for item in items),
            "original_test_videos": sum(item["original_split"] == "test" for item in items),
            "model_split_source_video_counts": dict(split_counts),
            "valid_records": sum(item["valid"] for item in items),
            "invalid_records": sum(not item["valid"] for item in items),
            "cached_frames_decoded": decoded_total,
            "cached_frame_failures": sum(len(item["frame_failures"]) for item in items),
            "exact_duplicate_groups": len(duplicates),
            "duplicate_groups_across_official_splits": sum(group["crosses_official_split"] for group in duplicates),
            "duplicate_groups_across_model_splits": sum(group["crosses_model_split"] for group in duplicates),
            "timestamp_warning_videos": len(timestamp_warnings), "hashes_reused": hashes_reused,
            "source_videos_hashed": sum(bool(item["sha256"]) for item in items),
        },
        "global_errors": global_errors, "global_warnings": global_warnings,
        "orphan_labels": orphan_labels, "orphan_extraction_records": orphan_cached,
        "exact_duplicate_groups": duplicates, "timestamp_warnings": timestamp_warnings,
        "limitations": [
            "No manual review of all action labels or visual scene content was performed.",
            "SHA256 detects exact video-byte duplicates, not reencoded or visually similar videos.",
            "Raw video headers and first frames were checked; complete raw-video decode was not repeated because existing extracted frames were reused.",
            "Chronological order is checked against extraction metadata and frame naming; semantic visual order was not manually reviewed.",
            "This audit does not overwrite or retroactively change existing trained models or evaluation results.",
        ],
        "videos": items, "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    report_output = output_dir / "mivia_cleaning_report.json"
    manifest_output = output_dir / "mivia_cleaning_manifest.csv"
    with report_output.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(result), handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    fields = ("video_id", "source_video", "source_label", "raw_label", "label", "original_split", "model_split",
              "valid", "valid_for_training", "split_review_required", "sha256", "cached_frame_directory",
              "decoded_frames", "timestamp", "timestamp_issues", "errors", "warnings", "issues")
    with manifest_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in items:
            row = {key: item.get(key) for key in fields}
            for key in ("raw_label", "timestamp_issues", "errors", "warnings", "issues"):
                row[key] = json.dumps(_json_safe(row[key]), ensure_ascii=False, allow_nan=False)
            writer.writerow(row)
    print(f"Audit {result['status']}: {report_output}\nManifest: {manifest_output}", flush=True)
    return result


# Littering cleaning: image integrity, box repairs, duplicates, and clean export.
CACHE_VERSION = 1


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def sanitize_bbox(bbox, width, height, min_retained_area=0.99):
    """Return (clean box, action). Large clipping is held for review, not guessed."""
    if (not isinstance(bbox, list) or len(bbox) != 4
            or not all(finite_number(v) for v in bbox)):
        return None, 'invalid_bbox_format_or_number'
    x, y, w, h = map(float, bbox)
    if w <= 0 or h <= 0:
        return None, 'nonpositive_bbox_size'
    x0, y0 = max(0., min(float(width), x)), max(0., min(float(height), y))
    x1, y1 = max(0., min(float(width), x + w)), max(0., min(float(height), y + h))
    cw, ch = x1 - x0, y1 - y0
    if cw <= 0 or ch <= 0:
        return None, 'bbox_outside_image'
    cleaned = [x0, y0, cw, ch]
    if x < 0 or y < 0 or x + w > width or y + h > height:
        if cw * ch / (w * h) < min_retained_area:
            return None, 'bbox_large_clip_requires_review'
        return cleaned, 'clipped_small_edge_rounding'
    return cleaned, 'unchanged'


def yolo_lines(boxes, width, height):
    return [f"0 {(x+w/2)/width:.8f} {(y+h/2)/height:.8f} {w/width:.8f} {h/height:.8f}"
            for x, y, w, h in boxes]


def perceptual_hash(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    low = cv2.dct(small)[:8, :8].ravel()
    bits = low > np.median(low[1:])
    bits[0] = False
    return np.packbits(bits).tobytes().hex()


def inspect_image(path):
    """Decode all pixels and fingerprint both file bytes and decoded pixels."""
    path = Path(path)
    record = {'path': str(path), 'version': CACHE_VERSION, 'valid': False}
    try:
        before = path.stat()
        record.update(size_bytes=before.st_size, mtime_ns=before.st_mtime_ns)
        raw = path.read_bytes()
        record['sha256'] = hashlib.sha256(raw).hexdigest()
        decoded = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
        if decoded is None or decoded.size == 0 or decoded.ndim != 3:
            record['error'] = 'image_decode_failed'
            return record
        height, width = decoded.shape[:2]
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            record['error'] = 'image_changed_during_check'
            return record
        pixel_hasher = hashlib.sha256(f'{width}x{height}x3:'.encode())
        pixel_hasher.update(decoded.tobytes())
        record.update(valid=True, width=width, height=height,
                      pixel_sha256=pixel_hasher.hexdigest(), phash=perceptual_hash(decoded))
    except (OSError, cv2.error) as exc:
        record['error'] = f'image_read_error:{exc}'
    return record


def read_cache(path):
    cache = {}
    if path.is_file():
        for line in path.read_text(encoding='utf-8').splitlines():
            try:
                item = json.loads(line)
                if item.get('version') == CACHE_VERSION:
                    cache[item['path']] = item
            except (ValueError, KeyError, TypeError):
                continue
    return cache


def cache_matches(item, path):
    if not item or not item.get('valid'):
        return False
    stat = path.stat()
    return item.get('size_bytes') == stat.st_size and item.get('mtime_ns') == stat.st_mtime_ns


def group_duplicates(records, key):
    groups = defaultdict(list)
    for r in records:
        if r.get(key):
            groups[r[key]].append(r)
    return [members for _, members in sorted(groups.items()) if len(members) > 1]


def exclude_confirmed_duplicates(records):
    """Keep val over train for matching labels; quarantine annotation conflicts."""
    decisions = []
    for members in group_duplicates(records, 'pixel_sha256'):
        valid = [r for r in members if not r['errors']]
        if len(valid) < 2:
            continue
        signatures = {tuple(sorted(tuple(round(v, 4) for v in b) for b in r['boxes'])) for r in valid}
        if len(signatures) > 1:
            for r in valid:
                r['errors'].append('same_pixels_conflicting_annotations_requires_review')
            decisions.append({'action': 'hold_conflicting_annotations', 'paths': [r['path'] for r in valid]})
            continue
        ordered = sorted(valid, key=lambda r: (0 if r['source_split'] == 'valid' else 1, r['path']))
        for r in ordered[1:]:
            r['errors'].append('confirmed_exact_duplicate_excluded')
        decisions.append({'action': 'keep_one_exact_pixel_image', 'kept': ordered[0]['path'],
                          'excluded': [r['path'] for r in ordered[1:]],
                          'cross_split': len({r['source_split'] for r in ordered}) > 1})
    return decisions


def write_csv(path, rows, fields):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(row.get(k), ensure_ascii=False) if isinstance(row.get(k), (list, dict))
                             else row.get(k, '') for k in fields})


def near_duplicate_candidates(raw_records, max_candidates=300):
    """Candidate retrieval only: three nearest pHashes per non-train image.

    Static surveillance backgrounds can match despite different actions.
    Never automatically remove these candidate images or call this exhaustive.
    """
    train = [r for r in raw_records if r.get('valid') and r['source_split'] == 'train']
    queries = [r for r in raw_records if r.get('valid') and r['source_split'] in {'valid', 'test'}]
    if not train or not queries:
        return [], 0
    as_array = lambda rows: np.asarray([list(bytes.fromhex(r['phash'])) for r in rows], dtype=np.uint8)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    matches = matcher.knnMatch(as_array(queries), as_array(train), k=min(3, len(train)))
    pairs = []
    for q, found in zip(queries, matches):
        for match in found:
            t = train[match.trainIdx]
            if match.distance <= 6 and q['pixel_sha256'] != t['pixel_sha256']:
                pairs.append({'query_split': q['source_split'], 'query_image': q['path'],
                              'training_image': t['path'], 'phash_hamming': int(match.distance),
                              'decision': 'human_review_only_not_confirmed_duplicate'})
    pairs.sort(key=lambda r: (r['phash_hamming'], r['query_image'], r['training_image']))
    return pairs[:max_candidates], len(pairs)


def create_review_contact_sheet(candidates, path):
    """Side-by-side actual images, without invented annotations or labels."""
    selected = candidates[:8]
    if not selected:
        return
    canvas = np.full((len(selected) * 190 + 55, 1100, 3), 255, dtype=np.uint8)
    cv2.putText(canvas, 'Similarity candidates only - human review required', (18, 30),
                cv2.FONT_HERSHEY_SIMPLEX, .8, (0, 0, 0), 2)
    for i, row in enumerate(selected):
        top = 55 + i * 190
        for col, key in enumerate(('training_image', 'query_image')):
            image = cv2.imread(row[key])
            h, w = image.shape[:2]
            scale = min(525/w, 155/h)
            image = cv2.resize(image, (round(w*scale), round(h*scale)))
            x = 15 + col * 550
            canvas[top:top+image.shape[0], x:x+image.shape[1]] = image
            label = ('train' if col == 0 else row['query_split']) + f" | candidate {i+1} | distance {row['phash_hamming']}"
            cv2.putText(canvas, label, (x, top+176), cv2.FONT_HERSHEY_SIMPLEX, .55, (0, 0, 0), 1)
    if not cv2.imwrite(str(path), canvas):
        raise OSError(f'Could not write review sheet: {path}')


def export_yolo(records, output_root, signature):
    """A new dataset only. Existing unrelated directories are never overwritten."""
    marker = output_root / '_cleaning_export.json'
    if output_root.exists():
        if marker.is_file() and json.loads(marker.read_text()).get('signature') == signature:
            expected = [r for r in records if r['include']]
            for r in expected:
                split = 'train' if r['source_split'] == 'train' else 'val'
                name = f"{r['image_id']:08d}_{Path(r['path']).name}"
                img = output_root / 'images' / split / name
                label = output_root / 'labels' / split / f'{Path(name).stem}.txt'
                if not img.is_file() or not label.is_file() or label.read_text() != r['yolo_text']:
                    raise ValueError('Existing clean export is incomplete or changed; choose a NEW --export-root.')
            return {'status': 'reused_matching_export', 'path': str(output_root), 'images': len(expected)}
        raise FileExistsError(f'Output already exists with a different/no manifest: {output_root}. Choose a new --export-root.')
    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=output_root.name + '-partial-', dir=output_root.parent))
    count = 0
    for r in records:
        if not r['include']:
            continue
        split = 'train' if r['source_split'] == 'train' else 'val'
        name = f"{r['image_id']:08d}_{Path(r['path']).name}"
        img = stage / 'images' / split / name
        label = stage / 'labels' / split / f'{Path(name).stem}.txt'
        img.parent.mkdir(parents=True, exist_ok=True)
        label.parent.mkdir(parents=True, exist_ok=True)
        # Never rewrite/recompress image bytes. Hard links save storage; source
        # images are treated as read-only throughout the training pipeline.
        try:
            os.link(r['path'], img)
        except OSError:
            shutil.copy2(r['path'], img)
        label.write_text(r['yolo_text'], encoding='utf-8')
        count += 1
    yaml = f'path: {json.dumps(str(output_root))}\ntrain: images/train\nval: images/val\nnames:\n  0: littering\n'
    (stage / 'data.yaml').write_text(yaml, encoding='utf-8')
    (stage / '_cleaning_export.json').write_text(json.dumps({'signature': signature, 'images': count,
        'raw_files_modified': False, 'source_split_preserved': True,
        'manual_semantic_and_augmentation_review_complete': False}, indent=2) + '\n')
    stage.rename(output_root)
    return {'status': 'created', 'path': str(output_root), 'images': count,
            'data_yaml': str(output_root / 'data.yaml')}


def clean(source_root, output_dir, export_root=None, workers=4):
    started = time.monotonic()
    source_root, output_dir = source_root.resolve(), output_dir.resolve()
    if output_dir == source_root or source_root in output_dir.parents:
        raise ValueError('Reports cannot be written inside the original dataset.')
    if export_root is not None:
        export_root = export_root.resolve()
        if (export_root == source_root or source_root in export_root.parents or
                export_root == output_dir or export_root in source_root.parents):
            raise ValueError('Clean export must be a separate new directory, outside original data.')
    output_dir.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)
    cache_path = output_dir / 'littering_image_cache.jsonl'
    cache = read_cache(cache_path)
    # Isolate any interrupted trailing cache line before appending checkpoints.
    if cache_path.is_file() and cache_path.stat().st_size:
        with cache_path.open('rb') as stream:
            stream.seek(-1, 2)
            needs_newline = stream.read(1) != b'\n'
        if needs_newline:
            with cache_path.open('a') as stream:
                stream.write('\n')
    paths = [(split, p.resolve()) for split in ('train', 'valid', 'test')
             for p in sorted((source_root / split).rglob('*')) if p.suffix.lower() in IMAGE_SUFFIXES and p.is_file()]
    print(f'Littering cleaning: checking {len(paths)} raw images; originals remain unchanged.', flush=True)
    raw_records = []
    reused = 0
    def check(pair):
        split, path = pair
        item = cache.get(str(path))
        was_cached = cache_matches(item, path)
        record = dict(item) if was_cached else inspect_image(path)
        record['source_split'] = split
        return record, was_cached
    with ThreadPoolExecutor(max_workers=workers) as pool, cache_path.open('a', encoding='utf-8') as stream:
        for index, (record, was_cached) in enumerate(pool.map(check, paths), 1):
            raw_records.append(record)
            reused += int(was_cached)
            if not was_cached:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
            if index % 500 == 0 or index == len(paths):
                stream.flush()
                print(f'Images {index}/{len(paths)} | cache reused {reused} | {time.monotonic()-started:.1f}s', flush=True)
    by_path = {r['path']: r for r in raw_records}
    records, issues, changes, annotation_hashes = [], [], [], []
    for split in ('train', 'valid'):
        anno_path = source_root / split / '_annotations.coco.json'
        anno_bytes = anno_path.read_bytes()
        annotation_hashes.append(hashlib.sha256(anno_bytes).hexdigest())
        payload = json.loads(anno_bytes)
        target_ids, _ = choose_categories(payload, 'littering')
        images, annotations = payload['images'], payload['annotations']
        ids = [image['id'] for image in images]
        repeated_ids = {key for key, n in Counter(ids).items() if n > 1}
        if repeated_ids:
            issues.append({'split': split, 'error': 'duplicate_coco_image_ids', 'ids': sorted(repeated_ids)})
        known_ids = set(ids)
        by_id = defaultdict(list)
        annotation_ids = Counter(a.get('id') for a in annotations)
        for a in annotations:
            if a.get('image_id') not in known_ids:
                issues.append({'split': split, 'annotation_id': a.get('id'), 'error': 'orphan_annotation'})
            else:
                by_id[a['image_id']].append(a)
        for image in images:
            r = {'source_split': split, 'image_id': image['id'], 'source_file_name': image['file_name'],
                 'path': '', 'errors': [], 'warnings': [], 'boxes': [], 'original_boxes': len(by_id[image['id']])}
            if image['id'] in repeated_ids:
                r['errors'].append('duplicate_coco_image_id')
            try:
                path = resolve_image(source_root / split, image['file_name']).resolve()
                if source_root / split not in path.parents:
                    raise ValueError('Image path escapes its source split')
                info = by_path.get(str(path))
                r['path'] = str(path)
                if not info or not info['valid']:
                    r['errors'].append('missing_or_corrupt_image')
                else:
                    for key in ('width','height','sha256','pixel_sha256'):
                        r[key] = info[key]
                    if image.get('width') != r['width'] or image.get('height') != r['height']:
                        r['errors'].append('coco_dimensions_disagree_with_decoded_image')
                    for a in by_id[image['id']]:
                        if annotation_ids[a.get('id')] > 1:
                            r['errors'].append('duplicate_annotation_id')
                        if a.get('category_id') not in target_ids:
                            r['errors'].append('unknown_annotation_category')
                            continue
                        box, action = sanitize_bbox(a.get('bbox'), r['width'], r['height'])
                        if box is None:
                            r['errors'].append(action)
                        elif box in r['boxes']:
                            changes.append({'split': split, 'image_id': image['id'], 'annotation_id': a.get('id'),
                                            'action': 'remove_repeated_identical_box', 'before': a['bbox'], 'after': box})
                        else:
                            r['boxes'].append(box)
                            if action != 'unchanged':
                                changes.append({'split': split, 'image_id': image['id'], 'annotation_id': a.get('id'),
                                                'action': action, 'before': a['bbox'], 'after': box})
                    if not by_id[image['id']]:
                        r['warnings'].append('provided_empty_annotation_requires_visual_review')
            except (OSError, ValueError, RuntimeError) as exc:
                r['errors'].append(f'image_resolution_error:{exc}')
            records.append(r)
    duplicate_decisions = exclude_confirmed_duplicates(records)
    for r in records:
        r['include'] = not r['errors']
        r['boxes_after_cleaning'] = len(r['boxes'])
        lines = yolo_lines(r['boxes'], r['width'], r['height']) if r.get('width') else []
        r['yolo_text'] = '\n'.join(lines) + ('\n' if lines else '')
    referenced = {r['path'] for r in records if r['path']}
    for r in raw_records:
        r['referenced_by_coco'] = r['path'] in referenced
    byte_groups = group_duplicates(raw_records, 'sha256')
    pixel_groups = group_duplicates(raw_records, 'pixel_sha256')
    print('Checking perceptual similarity candidates across source splits (review only)...', flush=True)
    near, total_near = near_duplicate_candidates(raw_records)
    write_csv(output_dir / 'littering_similarity_review.csv', near,
              ['query_split','query_image','training_image','phash_hamming','decision'])
    create_review_contact_sheet(near, output_dir / 'littering_similarity_review.jpg')
    write_csv(output_dir / 'littering_cleaning_manifest.csv', records,
              ['source_split','image_id','source_file_name','path','width','height','sha256','pixel_sha256',
               'original_boxes','boxes_after_cleaning','include','errors','warnings'])
    write_csv(output_dir / 'littering_box_repairs.csv', changes,
              ['split','image_id','annotation_id','action','before','after'])
    unused = [{**r, 'reason': 'unlabelled_test_image' if r['source_split'] == 'test'
              else 'no_coco_image_record_do_not_invent_negative_label'} for r in raw_records if not r['referenced_by_coco']]
    write_csv(output_dir / 'littering_unlabelled_files.csv', unused,
              ['source_split','path','reason','valid','sha256'])
    signature_data = {'annotations': annotation_hashes, 'cache_version': CACHE_VERSION,
                      'included': [(r['source_split'],r['image_id'],r.get('sha256'),r['yolo_text'])
                                   for r in records if r['include']]}
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()
    exported = None
    if export_root is not None and not issues:
        exported = export_yolo(records, export_root, signature)
    report = {
        'status': 'technical_cleaning_completed_manual_review_pending',
        'created_at_utc': datetime.now(timezone.utc).isoformat(), 'source_root': str(source_root),
        'raw_files_modified': False, 'existing_models_or_results_replaced': False,
        'source_split_preserved': True, 'original_scene_count_verified': False,
        'summary': {
            'raw_images': len(raw_records), 'raw_split_counts': dict(Counter(r['source_split'] for r in raw_records)),
            'decoded_valid_images': sum(r['valid'] for r in raw_records),
            'corrupt_or_unreadable_images': sum(not r['valid'] for r in raw_records),
            'coco_image_records': len(records), 'included_clean_image_records': sum(r['include'] for r in records),
            'clean_split_counts': dict(Counter(r['source_split'] for r in records if r['include'])),
            'source_annotation_boxes': sum(r['original_boxes'] for r in records),
            'clean_boxes': sum(len(r['boxes']) for r in records if r['include']),
            'small_edge_boxes_repaired': sum(c['action']=='clipped_small_edge_rounding' for c in changes),
            'repeated_identical_boxes_removed': sum(c['action']=='remove_repeated_identical_box' for c in changes),
            'images_quarantined_from_clean_export': sum(not r['include'] for r in records),
            'empty_annotated_images_retained_for_review': sum(r['include'] and not r['boxes'] for r in records),
            'unreferenced_training_images': sum(r['source_split']=='train' for r in unused),
            'unlabelled_test_images': sum(r['source_split']=='test' for r in unused),
            'exact_byte_duplicate_groups_raw': len(byte_groups),
            'exact_pixel_duplicate_groups_raw': len(pixel_groups),
            'exact_byte_duplicate_groups_across_raw_splits': sum(len({r['source_split'] for r in g})>1 for g in byte_groups),
            'exact_pixel_duplicate_groups_across_raw_splits': sum(len({r['source_split'] for r in g})>1 for g in pixel_groups),
            'perceptual_review_candidates_found_in_nearest3_search': total_near,
            'perceptual_review_candidates_saved': len(near), 'cached_image_checks_reused': reused},
        'box_repair_policy': 'Clip only boxes retaining at least 99% of their area; quarantine larger/invalid changes. Never infer a missing label.',
        'filename_provenance_policy': 'Do not deduplicate using video_X_Y_jpg prefixes: these names collide across visibly different scenes.',
        'global_annotation_issues': issues, 'confirmed_duplicate_cleaning_decisions': duplicate_decisions,
        'raw_duplicate_groups': [[{'path':r['path'],'split':r['source_split'],'referenced_by_coco':r['referenced_by_coco']} for r in g] for g in byte_groups],
        'corrupt_images': [r for r in raw_records if not r['valid']], 'export': exported,
        'limitations': [
            'Technical validity and exact duplication were checked. All action/box labels were not manually reviewed.',
            'Nearest-three perceptual hashes yield candidates only, not verified duplicates or an exhaustive augmentation audit.',
            'Related scenes can have similar static backgrounds; candidate images are never automatically removed.',
            'The supplied export does not reliably map augmented variants to original sources. Source-level independence remains unverified.',
            'The clean export preserves supplied train/validation membership and has no labelled test split.',
            'Existing models and their metrics do not become results from a newly cleaned/repartitioned dataset automatically.'],
        'elapsed_seconds': round(time.monotonic()-started, 3)}
    report_path = output_dir / 'littering_cleaning_report.json'
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(report['summary'], indent=2), flush=True)
    print(f'Report: {report_path}\nClean export: {exported}', flush=True)
    return report


# One command-line entry point for both datasets.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=('all', 'mivia', 'littering'), default='all')
    parser.add_argument('--project-root', type=Path, default=PROJECT_ROOT)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--export-root', type=Path)
    parser.add_argument('--audit-only', action='store_true', help='Do not create the cleaned YOLO export.')
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    root = args.project_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve() if args.output_dir else root / 'results/data/cleaning'
    raw_dirs = [root / 'datasets/MIVIA-IWDD 500', root / 'datasets/Littering.v8-yolotiny.coco']
    if any(output == raw or raw in output.parents for raw in raw_dirs):
        parser.error('--output-dir cannot be inside an original dataset')
    output.mkdir(parents=True, exist_ok=True)
    reports = {}
    if args.dataset in ('all', 'mivia'):
        reports['mivia'] = audit_mivia(root, output, workers=args.workers)
    if args.dataset in ('all', 'littering'):
        export = args.export_root.expanduser().resolve() if args.export_root else root / 'datasets/Littering-YOLO-clean'
        reports['littering'] = clean(raw_dirs[1], output, None if args.audit_only else export, args.workers)
    summary = {'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'originals_deleted_or_modified': False, 'models_retrained': False,
               'datasets_run': list(reports),
               'statuses': {name: report['status'] for name, report in reports.items()},
               'summaries': {name: report['summary'] for name, report in reports.items()},
               'review_warning': 'Technical cleaning is automated. Similarity, ambiguous source provenance, and semantic labels still need review.'}
    (output / 'cleaning_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'\nCleaning finished. Reports: {output}', flush=True)
    print('Review candidates are not automatically deleted. Existing trained models were not changed.', flush=True)
    failed = (not reports.get('mivia', {}).get('technical_validation_pass', True)
              or bool(reports.get('littering', {}).get('global_annotation_issues')))
    return 1 if failed else 0


if __name__ == '__main__':
    raise SystemExit(main())

