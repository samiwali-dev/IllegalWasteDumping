"""MIVIA data loading, video-level splits, and checkpointed frame extraction.

Every action model uses the same 24 chronological RGB frames per source video.
Run `.venv/bin/python data.py --help` for extraction options, or
`.venv/bin/python clean.py --dataset littering` for the cleaned YOLO export.
The legacy `prepare-yolo` converter requires an explicit output directory.
Existing complete frame directories are reused; event timestamps never choose
the frames.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_REPORT = PROJECT_ROOT / "results" / "data" / "mivia_frames_report_24.json"
SEQUENCE_LENGTH = 24
IMAGE_SIZE = 112
SEED = 42


def _frame_paths(record: dict[str, Any]) -> tuple[Path, ...]:
    """Return the uniformly sampled frames in the report's temporal order."""
    video_dir = Path(record["output_dir"])
    if not video_dir.is_dir():
        # Reports contain absolute paths. This also supports moving the entire
        # project to a different machine without modifying the source report.
        video_dir = (
            PROJECT_ROOT
            / "datasets"
            / "MIVIA-Frames"
            / record["split"]
            / record["class_name"]
            / Path(record["video_id"]).stem
        )
    return tuple(video_dir / f"frame_{index:03d}.jpg" for index in range(1, SEQUENCE_LENGTH + 1))


def load_manifest(report_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load and validate the 500-video MIVIA extraction report.

    Records remain the original report dictionaries so model implementations can
    read ``video_id``, ``label``, ``split`` and ``output_dir`` directly.
    """
    path = Path(report_path) if report_path is not None else DEFAULT_REPORT
    with path.open("r", encoding="utf-8") as handle:
        report = json.load(handle)

    if report.get("status") != "validated" or report.get("failures"):
        raise ValueError(f"MIVIA extraction report is not validated: {path}")
    if report.get("sequence_length") != SEQUENCE_LENGTH:
        raise ValueError(f"Expected {SEQUENCE_LENGTH} frames per video in {path}")
    if report.get("timestamp_used_for_sampling") is not False:
        raise ValueError("Frames must be sampled uniformly, without event timestamps")

    records = report.get("videos")
    if not isinstance(records, list) or len(records) != 500:
        raise ValueError(f"Expected 500 original videos in {path}")

    counts: Counter[tuple[str, int]] = Counter()
    seen_ids: set[str] = set()
    seen_directories: set[Path] = set()
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("Each video record must be a JSON object")
        video_id = record.get("video_id")
        split = record.get("split")
        label = record.get("label")
        if not isinstance(video_id, str) or not video_id.endswith(".mp4"):
            raise ValueError(f"Invalid video ID: {video_id!r}")
        if video_id in seen_ids:
            raise ValueError(f"Duplicate video ID: {video_id}")
        seen_ids.add(video_id)
        if split not in {"train", "test"} or type(label) is not int or label not in {0, 1}:
            raise ValueError(f"Invalid split/label for {video_id}")
        expected_class = "dumping" if label else "no_dumping"
        if record.get("class_name") != expected_class:
            raise ValueError(f"Class name disagrees with label for {video_id}")
        indices = record.get("sampled_frame_indices_zero_based")
        if (
            not isinstance(indices, list)
            or len(indices) != SEQUENCE_LENGTH
            or any(type(index) is not int for index in indices)
            or any(left >= right for left, right in zip(indices, indices[1:]))
        ):
            raise ValueError(f"Invalid temporal sample indices for {video_id}")
        paths = _frame_paths(record)
        video_dir = paths[0].parent.resolve()
        if video_dir in seen_directories:
            raise ValueError(f"Frame directory reused by multiple videos: {video_dir}")
        seen_directories.add(video_dir)
        missing = [frame for frame in paths if not frame.is_file() or frame.stat().st_size == 0]
        if missing:
            raise FileNotFoundError(f"Missing or empty frame for {video_id}: {missing[0]}")
        counts[split, label] += 1

    if counts != Counter({("train", 0): 200, ("train", 1): 200,
                          ("test", 0): 50, ("test", 1): 50}):
        raise ValueError(f"Unexpected official MIVIA class/split counts: {counts}")
    return records


def split_records(
    records: list[dict[str, Any]], seed: int = SEED
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create stratified 80/20 fit/validation groups from official train videos.

    Each video goes into exactly one group. The official test videos are never
    shuffled into training or validation. For MIVIA this returns 320/80/100.
    """
    if len({record["video_id"] for record in records}) != len(records):
        raise ValueError("Video IDs must be unique before splitting")
    if any(record["split"] not in {"train", "test"} or record["label"] not in {0, 1}
           for record in records):
        raise ValueError("Invalid official split or label")
    rng = random.Random(seed)
    train: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    for label in (0, 1):
        group = sorted(
            (record for record in records if record["split"] == "train" and record["label"] == label),
            key=lambda record: record["video_id"],
        )
        rng.shuffle(group)
        validation_size = round(len(group) * 0.2)
        validation.extend(group[:validation_size])
        train.extend(group[validation_size:])
    test = [record for record in records if record["split"] == "test"]
    for group in (train, validation, test):
        group.sort(key=lambda record: record["video_id"])
    if len(train) + len(validation) + len(test) != len(records):
        raise AssertionError("A video was lost during splitting")
    return train, validation, test


def load_clip(record: dict[str, Any], size: int = IMAGE_SIZE) -> np.ndarray:
    """Read one video as ordered ``[24, size, size, 3]`` uint8 RGB frames.

    Each frame keeps its aspect ratio and is centered on a black square canvas.
    The returned array has no model-specific normalization or augmentation.
    """
    if not isinstance(size, int) or size < 1:
        raise ValueError("size must be a positive integer")
    clip = np.zeros((SEQUENCE_LENGTH, size, size, 3), dtype=np.uint8)
    for index, path in enumerate(_frame_paths(record)):
        frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame is None or frame.size == 0:
            raise OSError(f"Could not decode MIVIA frame: {path}")
        height, width = frame.shape[:2]
        scale = min(size / width, size / height)
        scaled_width = max(1, min(size, round(width * scale)))
        scaled_height = max(1, min(size, round(height * scale)))
        interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
        resized = cv2.resize(frame, (scaled_width, scaled_height), interpolation=interpolation)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        top = (size - scaled_height) // 2
        left = (size - scaled_width) // 2
        clip[index, top:top + scaled_height, left:left + scaled_width] = rgb
    return clip


# Extraction CLI: preserves the existing report and checkpoint protocol.
DEFAULT_SOURCE = PROJECT_ROOT / "datasets" / "MIVIA-IWDD 500"
DEFAULT_OUTPUT = PROJECT_ROOT / "datasets" / "MIVIA-Frames"
DEFAULT_RESULTS = PROJECT_ROOT / "results" / "data"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sequence-length", type=int, default=24)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--jpeg-quality", type=int, default=92)
    parser.add_argument("--progress-every", type=int, default=10)
    args = parser.parse_args(argv)
    if args.sequence_length < 2:
        parser.error("--sequence-length must be at least 2")
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in the range 1..100")
    if args.progress_every < 1:
        parser.error("--progress-every must be at least 1")
    args.source_root = args.source_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.report = (
        args.report.expanduser().resolve()
        if args.report
        else DEFAULT_RESULTS / f"mivia_frames_report_{args.sequence_length}.json"
    )
    if args.output_root == args.source_root or args.source_root in args.output_root.parents:
        parser.error("output directory must not be inside the raw MIVIA dataset")
    if args.report == args.source_root or args.source_root in args.report.parents:
        parser.error("report must not be written inside the raw MIVIA dataset")
    return args


def read_label(path: Path) -> tuple[int, float | None]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    value = data.get("label")
    if isinstance(value, bool):
        label = int(value)
    elif isinstance(value, str) and value.lower() in {"true", "false"}:
        label = int(value.lower() == "true")
    else:
        raise ValueError(f"Invalid label in {path}: {value!r}")
    timestamp = data.get("timestamp")
    if label == 1 and isinstance(timestamp, (int, float)) and not isinstance(timestamp, bool):
        timestamp = float(timestamp)
    else:
        timestamp = None
    return label, timestamp


def sample_indices(frame_count: int, sequence_length: int) -> list[int]:
    if frame_count < sequence_length:
        raise ValueError(
            f"Only {frame_count} frames; cannot sample {sequence_length} distinct frames"
        )
    last = frame_count - 1
    indices = [round(i * last / (sequence_length - 1)) for i in range(sequence_length)]
    if indices[0] != 0 or indices[-1] != last:
        raise AssertionError("Sampling failed to include both temporal endpoints")
    if any(left >= right for left, right in zip(indices, indices[1:])):
        raise AssertionError("Sampled indices are not strictly increasing")
    return indices


def expected_names(sequence_length: int) -> list[str]:
    return [f"frame_{index:03d}.jpg" for index in range(1, sequence_length + 1)]


def is_complete_video_dir(path: Path, sequence_length: int) -> bool:
    if not path.is_dir():
        return False
    return sorted(p.name for p in path.iterdir()) == expected_names(sequence_length) and all(
        (path / name).stat().st_size > 0 for name in expected_names(sequence_length)
    )


def count_frames_exact(video_path: Path) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    count = 0
    try:
        while capture.grab():
            count += 1
    finally:
        capture.release()
    return count


def write_sampled_frames(
    video_path: Path, destination: Path, indices: list[int], jpeg_quality: int
) -> list[float | None]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    sampled_times: list[float | None] = []
    wanted = {frame_index: output_index for output_index, frame_index in enumerate(indices, 1)}
    try:
        for frame_index in range(indices[-1] + 1):
            if not capture.grab():
                raise RuntimeError(f"Video ended before expected frame {indices[-1]}")
            output_index = wanted.get(frame_index)
            if output_index is None:
                continue
            ok, frame = capture.retrieve()
            if not ok or frame is None or frame.size == 0:
                raise RuntimeError(f"Could not decode frame {frame_index}")
            output_path = destination / f"frame_{output_index:03d}.jpg"
            if not cv2.imwrite(
                str(output_path), frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality]
            ):
                raise RuntimeError(f"Could not write {output_path}")
            position_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            sampled_times.append(
                round(position_ms / 1000.0, 6)
                if math.isfinite(position_ms) and position_ms >= 0
                else None
            )
        if capture.grab():
            raise RuntimeError("Video contains more frames than its reported frame count")
    finally:
        capture.release()
    if len(sampled_times) != len(indices):
        raise RuntimeError(f"Extracted only {len(sampled_times)} of {len(indices)} frames")
    return sampled_times


def extract_video(
    video_path: Path, output_dir: Path, sequence_length: int, jpeg_quality: int
) -> tuple[list[int], list[float | None], float, str]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {video_path}")
    try:
        reported_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if not math.isfinite(fps) or fps <= 0:
        raise RuntimeError(f"Invalid FPS for {video_path}: {fps}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and not is_complete_video_dir(output_dir, sequence_length):
        raise RuntimeError(
            f"Incomplete or different-length output exists: {output_dir}. "
            "Inspect it instead of overwriting it."
        )
    if reported_count < sequence_length:
        reported_count = count_frames_exact(video_path)
    indices = sample_indices(reported_count, sequence_length)
    status = "reused" if output_dir.exists() else "extracted"

    if status == "reused":
        # The source is still checked on resume. The existing sequence is never overwritten.
        actual_count = count_frames_exact(video_path)
        if actual_count != reported_count:
            raise RuntimeError(
                f"Source frame count changed: metadata {reported_count}, decoded {actual_count}"
            )
        return indices, [None] * len(indices), fps, status

    with tempfile.TemporaryDirectory(prefix=f".{video_path.stem}-", dir=output_dir.parent) as temp:
        temporary_dir = Path(temp)
        try:
            times = write_sampled_frames(video_path, temporary_dir, indices, jpeg_quality)
        except RuntimeError as exc:
            if not (
                "more frames than its reported frame count" in str(exc)
                or "Video ended before expected frame" in str(exc)
            ):
                raise
            # Recount if a video container under- or over-reports its frame count.
            actual_count = count_frames_exact(video_path)
            indices = sample_indices(actual_count, sequence_length)
            for path in temporary_dir.glob("frame_*.jpg"):
                path.unlink()
            times = write_sampled_frames(video_path, temporary_dir, indices, jpeg_quality)
        if not is_complete_video_dir(temporary_dir, sequence_length):
            raise RuntimeError(f"Incomplete temporary extraction for {video_path}")
        temporary_dir.rename(output_dir)
    return indices, times, fps, status


def inventory(source_root: Path) -> tuple[list[tuple[str, Path, Path]], list[dict]]:
    items: list[tuple[str, Path, Path]] = []
    failures: list[dict] = []
    stems_by_split: dict[str, set[str]] = {}
    for split in ("train", "test"):
        video_dir = source_root / split / "videos"
        label_dir = source_root / split / "labels"
        if not video_dir.is_dir() or not label_dir.is_dir():
            raise FileNotFoundError(f"Missing MIVIA input directory under {source_root / split}")
        videos = sorted(video_dir.glob("*.mp4"))
        labels = {path.stem: path for path in label_dir.glob("*.json")}
        stems = {path.stem for path in videos}
        stems_by_split[split] = stems
        for video_path in videos:
            label_path = labels.get(video_path.stem)
            if label_path is None:
                failures.append(
                    {"split": split, "video_id": video_path.name, "error": "Missing JSON label"}
                )
            else:
                items.append((split, video_path, label_path))
        for stem in sorted(labels.keys() - stems):
            failures.append(
                {"split": split, "video_id": f"{stem}.mp4", "error": "JSON without video"}
            )
    overlap = stems_by_split["train"] & stems_by_split["test"]
    if overlap:
        raise RuntimeError(f"Video IDs in both official splits: {sorted(overlap)[:10]}")
    return items, failures


def make_report(
    source_root: Path,
    output_root: Path,
    sequence_length: int,
    items: list[tuple[str, Path, Path]],
    records: list[dict],
    failures: list[dict],
    status: str,
) -> dict:
    input_counts = Counter(split for split, _, _ in items)
    success_counts = Counter(record["split"] for record in records)
    label_counts = Counter(record["label"] for record in records)
    bad_timestamps = [
        record["video_id"]
        for record in records
        if record.get("event_timestamp_in_video") is False
    ]
    return {
        "status": status,
        "source_dataset": str(source_root),
        "output_dataset": str(output_root),
        "sequence_length": sequence_length,
        "sampling_rule": "round(i * (decoded_frame_count - 1) / (sequence_length - 1))",
        "timestamp_used_for_sampling": False,
        "summary": {
            "original_train_videos": input_counts["train"],
            "original_test_videos": input_counts["test"],
            "total_original_videos": len(items),
            "train_extracted_frames": success_counts["train"] * sequence_length,
            "test_extracted_frames": success_counts["test"] * sequence_length,
            "total_extracted_frames": len(records) * sequence_length,
            "dumping_videos": label_counts[1],
            "no_dumping_videos": label_counts[0],
            "failed_videos": len(failures),
            "reused_videos": sum(record["status"] == "reused" for record in records),
            "event_timestamps_outside_video": bad_timestamps,
        },
        "failures": failures,
        "videos": records,
    }


def save_report(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", suffix=".json", prefix=".mivia-report-",
        dir=path.parent, delete=False
    ) as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
        temporary_path = Path(handle.name)
    temporary_path.replace(path)


def extract_main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    items, failures = inventory(args.source_root)
    records: list[dict] = []
    started = time.monotonic()
    print(
        f"MIVIA originals: {sum(split == 'train' for split, _, _ in items)} train, "
        f"{sum(split == 'test' for split, _, _ in items)} test. "
        f"Sampling {args.sequence_length} ordered frames per video.",
        flush=True,
    )
    for number, (split, video_path, label_path) in enumerate(items, 1):
        try:
            label, event_timestamp = read_label(label_path)
            class_name = "dumping" if label else "no_dumping"
            output_dir = args.output_root / split / class_name / video_path.stem
            indices, times, fps, status = extract_video(
                video_path, output_dir, args.sequence_length, args.jpeg_quality
            )
            # Presentation timestamps are preferred when available; otherwise
            # index/FPS is only an estimate. Neither changes the sampled frames.
            valid_times = (
                status == "extracted"
                and all(value is not None for value in times)
                and all(
                    left is not None and right is not None and left < right
                    for left, right in zip(times, times[1:])
                )
            )
            sample_times = (
                [float(value) for value in times if value is not None]
                if valid_times
                else [round(index / fps, 6) for index in indices]
            )
            event_in_video = None
            nearest_gap = None
            if event_timestamp is not None:
                event_in_video = 0 <= event_timestamp <= sample_times[-1] + 0.05
                if event_in_video:
                    nearest_gap = round(
                        min(abs(value - event_timestamp) for value in sample_times), 6
                    )
            records.append(
                {
                    "split": split,
                    "video_id": video_path.name,
                    "label": label,
                    "class_name": class_name,
                    "source_frame_count": indices[-1] + 1,
                    "source_fps": round(fps, 6),
                    "sampled_frame_indices_zero_based": indices,
                    "sampled_times_seconds": sample_times,
                    "sampled_time_source": "video_position" if valid_times else "frame_index/fps",
                    "event_timestamp_seconds_diagnostic_only": event_timestamp,
                    "event_timestamp_in_video": event_in_video,
                    "nearest_sample_to_event_seconds": nearest_gap,
                    "output_dir": str(output_dir),
                    "status": status,
                }
            )
        except Exception as exc:
            failures.append(
                {"split": split, "video_id": video_path.name, "error": str(exc)}
            )
            print(f"FAILED {split}/{video_path.name}: {exc}", flush=True)
        if number % args.progress_every == 0 or number == len(items):
            elapsed = time.monotonic() - started
            print(
                f"Processed {number}/{len(items)} videos, "
                f"{len(records)} complete, {len(failures)} failed, {elapsed:.1f}s elapsed",
                flush=True,
            )
            save_report(
                args.report,
                make_report(
                    args.source_root, args.output_root, args.sequence_length,
                    items, records, failures, "in_progress"
                ),
            )

    # Decode every output image, not just count filenames, before calling the
    # generated frame dataset validated.
    validated: list[dict] = []
    for record in records:
        output_dir = Path(record["output_dir"])
        error = None
        if not is_complete_video_dir(output_dir, args.sequence_length):
            error = "Output directory is incomplete"
        else:
            for name in expected_names(args.sequence_length):
                frame = cv2.imread(str(output_dir / name))
                if frame is None or frame.size == 0:
                    error = f"Output image cannot be decoded: {name}"
                    break
        if error:
            failures.append(
                {"split": record["split"], "video_id": record["video_id"], "error": error}
            )
            print(f"FAILED validation {record['video_id']}: {error}", flush=True)
        else:
            validated.append(record)

    final_report = make_report(
        args.source_root, args.output_root, args.sequence_length,
        items, validated, failures, "validated" if not failures else "failed"
    )
    save_report(args.report, final_report)
    summary = final_report["summary"]
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    print(f"Report: {args.report}", flush=True)
    return 0 if not failures else 1


# Littering COCO-to-YOLO preparation (separate from MIVIA extraction).
YOLO_SOURCE = PROJECT_ROOT / "datasets" / "Littering.v8-yolotiny.coco"
YOLO_REPORT = PROJECT_ROOT / "results" / "data" / "littering_yolo_prepare.json"
LABELED_SPLITS = (("train", "train"), ("valid", "val"))
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_yolo_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="data.py prepare-yolo", description="Legacy COCO-to-YOLO converter for an explicit output directory. Use clean.py --dataset littering for the current cleaned training dataset.")
    parser.add_argument("--source-root", type=Path, default=YOLO_SOURCE)
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Explicit export directory; the obsolete Littering-YOLO export is not recreated automatically")
    parser.add_argument("--report", type=Path, default=YOLO_REPORT)
    parser.add_argument("--class-name", default="littering")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    args.source_root = args.source_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.report = args.report.expanduser().resolve()
    if args.source_root == args.output_root:
        parser.error("source-root and output-root must be different")
    if args.output_root in {PROJECT_ROOT, PROJECT_ROOT.parent, Path.home(), Path(args.output_root.anchor)}:
        parser.error("output-root must be a dedicated dataset export directory")
    if args.source_root in args.output_root.parents or args.output_root in args.source_root.parents:
        parser.error("source-root and output-root must not contain each other")
    return args


def find_split_dir(root: Path, requested: str) -> Path:
    aliases = {
        "train": ("train", "training"),
        "valid": ("valid", "val", "validation"),
        "test": ("test", "testing"),
    }[requested]
    for name in aliases:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find {requested!r} split below {root}")


def find_annotations(split_dir: Path) -> Path:
    preferred = (
        split_dir / "_annotations.coco.json",
        split_dir / "annotations.coco.json",
        split_dir / "annotations.json",
    )
    for candidate in preferred:
        if candidate.is_file():
            return candidate
    candidates = sorted(split_dir.glob("*.json"))
    if len(candidates) == 1:
        return candidates[0]
    raise FileNotFoundError(
        f"Could not identify one COCO annotation JSON in {split_dir}; found {len(candidates)} JSON files"
    )


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


def link_or_copy(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
        return "hardlink"
    except OSError:
        shutil.copy2(source, destination)
        return "copy"


def convert_split(
    source_dir: Path,
    output_root: Path,
    output_split: str,
    class_name: str,
) -> dict:
    annotation_path = find_annotations(source_dir)
    payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    category_ids, actual_class_name = choose_categories(payload, class_name)

    images = payload.get("images")
    annotations = payload.get("annotations")
    if not isinstance(images, list) or not isinstance(annotations, list):
        raise ValueError(f"Invalid COCO structure in {annotation_path}")

    by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in annotations:
        if int(annotation.get("category_id", -1)) in category_ids:
            by_image[int(annotation["image_id"])].append(annotation)

    image_output = output_root / "images" / output_split
    label_output = output_root / "labels" / output_split
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)

    box_count = 0
    empty_images = 0
    transfer_modes = Counter()
    for index, image in enumerate(images, 1):
        image_id = int(image["id"])
        width = int(image["width"])
        height = int(image["height"])
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size for image id {image_id}")

        source_image = resolve_image(source_dir, str(image["file_name"]))
        if source_image.suffix.lower() not in IMAGE_SUFFIXES:
            raise ValueError(f"Unsupported image type: {source_image}")

        safe_name = f"{image_id:08d}_{source_image.name}"
        destination_image = image_output / safe_name
        transfer_modes[link_or_copy(source_image, destination_image)] += 1

        lines: list[str] = []
        for annotation in by_image.get(image_id, []):
            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ValueError(f"Invalid bbox for image id {image_id}")
            x, y, w, h = map(float, bbox)
            x0 = max(0.0, min(float(width), x))
            y0 = max(0.0, min(float(height), y))
            x1 = max(0.0, min(float(width), x + w))
            y1 = max(0.0, min(float(height), y + h))
            clipped_w = x1 - x0
            clipped_h = y1 - y0
            if clipped_w <= 0 or clipped_h <= 0:
                continue
            cx = (x0 + x1) / 2.0 / width
            cy = (y0 + y1) / 2.0 / height
            nw = clipped_w / width
            nh = clipped_h / height
            lines.append(f"0 {cx:.8f} {cy:.8f} {nw:.8f} {nh:.8f}")
            box_count += 1

        if not lines:
            empty_images += 1
        (label_output / f"{Path(safe_name).stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        if index % 1000 == 0 or index == len(images):
            print(f"{output_split}: prepared {index}/{len(images)} images", flush=True)

    return {
        "source_split": str(source_dir),
        "annotation_file": str(annotation_path),
        "class_name": actual_class_name,
        "images": len(images),
        "boxes": box_count,
        "images_without_target_boxes": empty_images,
        "image_transfer": dict(transfer_modes),
    }


def write_data_yaml(output_root: Path, class_name: str) -> Path:
    path = output_root / "data.yaml"
    content = (
        f"path: {output_root.as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        f"  0: {json.dumps(class_name)}\n"
    )
    path.write_text(content, encoding="utf-8")
    return path


def prepare_yolo(argv: list[str] | None = None) -> int:
    args = parse_yolo_args(argv)
    if not args.source_root.is_dir():
        raise FileNotFoundError(f"Littering source dataset not found: {args.source_root}")
    if args.output_root.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output already exists: {args.output_root}. Use --overwrite to replace it.")
        shutil.rmtree(args.output_root)

    reports = {}
    for source_name, output_name in LABELED_SPLITS:
        split_dir = find_split_dir(args.source_root, source_name)
        reports[output_name] = convert_split(split_dir, args.output_root, output_name, args.class_name)

    class_names = {r["class_name"] for r in reports.values()}
    if len(class_names) != 1:
        raise ValueError(f"Split category names disagree: {class_names}")
    actual_class_name = next(iter(class_names))
    yaml_path = write_data_yaml(args.output_root, actual_class_name)

    report = {
        "status": "validated",
        "source_root": str(args.source_root),
        "output_root": str(args.output_root),
        "data_yaml": str(yaml_path),
        "class_names": [actual_class_name],
        "splits": reports,
        "total_images": sum(r["images"] for r in reports.values()),
        "total_boxes": sum(r["boxes"] for r in reports.values()),
        "unlabeled_source_test_images": sum(
            1 for path in (args.source_root / "test").glob("*")
            if path.suffix.lower() in IMAGE_SUFFIXES
        ),
        "note": (
            "The source export supplies COCO labels for train and valid only. "
            "Their membership is preserved. The source test images have no annotation JSON, "
            "so they are not used for detection metrics."
        ),
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run extraction by default, or the explicit prepare-yolo subcommand."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments and arguments[0] == "prepare-yolo":
        return prepare_yolo(arguments[1:])
    if arguments and arguments[0] == "extract":
        arguments = arguments[1:]
    return extract_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())


