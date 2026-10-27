"""Fast checks for data cleaning and the shared MIVIA evaluation protocol.

Run: .venv/bin/python test_project.py
The cleaning checks use tiny temporary fixtures. The MIVIA checks inspect the
existing 500-video manifest and one already extracted clip; no training or
full dataset extraction is performed.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from data import (
    IMAGE_SIZE, SEQUENCE_LENGTH, _frame_paths, load_clip, load_manifest, split_records,
)
from evaluate import _read_predictions, evaluate_predictions
from src.optical_flow_svm.main import FEATURE_DIM, flow_features
from clean import (
    clean, decode_frame, duplicate_groups, exclude_confirmed_duplicates,
    export_yolo, hash_video, inspect_image, load_hash_cache, parse_label_data,
    sanitize_bbox, yolo_lines,
)


class ProjectLayoutTests(unittest.TestCase):
    def test_results_locations_are_grouped_by_purpose(self):
        from common import PROJECT_ROOT, make_parser
        from data import DEFAULT_REPORT
        from evaluate import DEFAULT_OUTPUT
        from src.yolo.main import DEFAULT_RESULTS, make_train_parser

        self.assertEqual(DEFAULT_REPORT,
                         PROJECT_ROOT / "results/data/mivia_frames_report_24.json")
        self.assertEqual(DEFAULT_OUTPUT,
                         PROJECT_ROOT / "results/comparison/model_comparison.json")
        self.assertEqual(DEFAULT_RESULTS, PROJECT_ROOT / "results/yolo")
        self.assertEqual(make_train_parser().parse_args([]).results_dir, DEFAULT_RESULTS)
        self.assertIsNone(make_parser("Shared neural training").parse_args([]).results_dir)

    def test_action_checkpoint_mapping_uses_existing_src_models(self):
        from common import CHECKPOINT_PATHS, PROJECT_ROOT

        expected = {
            "optical_flow_svm": PROJECT_ROOT / "src/optical_flow_svm/model.joblib",
            "cnn_lstm": PROJECT_ROOT / "src/cnn_lstm/model.pt",
            "cnn_3d": PROJECT_ROOT / "src/cnn_3d/model.pt",
            "movinet_a0": PROJECT_ROOT / "src/movinet/model.pt",
        }
        self.assertEqual(CHECKPOINT_PATHS, expected)
        for name, checkpoint in CHECKPOINT_PATHS.items():
            with self.subTest(model=name):
                self.assertTrue(checkpoint.is_file(), str(checkpoint))

    def test_historical_primary_checkpoint_aliases_preserve_experiment_identity(self):
        from common import CHECKPOINT_PATHS
        from evaluate import _resolve_model_path

        names = {
            "optical_flow_svm": "optical_flow_svm.joblib",
            "cnn_lstm": "cnn_lstm_best.pt",
            "cnn_3d": "cnn_3d_best.pt",
            "movinet_a0": "movinet_a0_best.pt",
        }
        with tempfile.TemporaryDirectory() as directory:
            previous = Path(directory) / "previous-project" / "models"
            metadata = Path(directory) / "run.json"
            for name, filename in names.items():
                with self.subTest(model=name):
                    historical = previous / filename
                    self.assertEqual(_resolve_model_path(str(historical), metadata),
                                     CHECKPOINT_PATHS[name])
                    experiment = previous / "protocol_v1_positive_f1" / filename
                    self.assertEqual(_resolve_model_path(str(experiment), metadata), experiment)

    def test_yolo_import_and_training_defaults_use_clean_dataset(self):
        from common import PROJECT_ROOT
        from src.yolo.main import DEFAULT_DATA, DEFAULT_MODEL, make_train_parser

        self.assertEqual(DEFAULT_MODEL, PROJECT_ROOT / "src/yolo/model.pt")
        self.assertTrue(DEFAULT_MODEL.is_file())
        expected_data = PROJECT_ROOT / "datasets/Littering-YOLO-clean/data.yaml"
        self.assertEqual(DEFAULT_DATA, expected_data)
        self.assertEqual(make_train_parser().parse_args([]).data, expected_data)


class MiviaProtocolTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.records = load_manifest()
        cls.fit, cls.validation, cls.test = split_records(cls.records)

    def test_official_manifest_and_disjoint_stratified_split(self) -> None:
        self.assertEqual(len(self.records), 500)
        self.assertEqual(Counter(record["split"] for record in self.records),
                         {"train": 400, "test": 100})
        self.assertEqual((len(self.fit), len(self.validation), len(self.test)),
                         (320, 80, 100))
        self.assertEqual(Counter(record["label"] for record in self.fit), {0: 160, 1: 160})
        self.assertEqual(Counter(record["label"] for record in self.validation), {0: 40, 1: 40})
        self.assertEqual(Counter(record["label"] for record in self.test), {0: 50, 1: 50})

        id_sets = [{record["video_id"] for record in group}
                   for group in (self.fit, self.validation, self.test)]
        self.assertEqual([len(ids) for ids in id_sets], [320, 80, 100])
        self.assertFalse(id_sets[0] & id_sets[1])
        self.assertFalse(id_sets[0] & id_sets[2])
        self.assertFalse(id_sets[1] & id_sets[2])
        self.assertEqual(set.union(*id_sets), {record["video_id"] for record in self.records})
        self.assertTrue(all(record["split"] == "train"
                            for record in self.fit + self.validation))
        self.assertTrue(all(record["split"] == "test" for record in self.test))

    def test_all_clips_have_24_chronological_frame_references(self) -> None:
        expected_names = [f"frame_{index:03d}.jpg"
                          for index in range(1, SEQUENCE_LENGTH + 1)]
        for record in self.records:
            with self.subTest(video_id=record["video_id"]):
                indices = record["sampled_frame_indices_zero_based"]
                self.assertEqual(len(indices), SEQUENCE_LENGTH)
                self.assertTrue(all(left < right for left, right in zip(indices, indices[1:])))
                self.assertEqual([path.name for path in _frame_paths(record)], expected_names)

    def test_clip_is_ordered_uint8_rgb_and_flow_is_finite(self) -> None:
        record = self.fit[0]
        clip = load_clip(record)
        self.assertEqual(clip.shape, (24, IMAGE_SIZE, IMAGE_SIZE, 3))
        self.assertEqual(clip.dtype, np.uint8)

        # Match every decoded frame to its corresponding output position, including
        # the BGR-to-RGB conversion, so reversed or shuffled clips are caught.
        for index, path in enumerate(_frame_paths(record)):
            source = cv2.imread(str(path), cv2.IMREAD_COLOR)
            self.assertIsNotNone(source)
            height, width = source.shape[:2]
            scale = min(IMAGE_SIZE / width, IMAGE_SIZE / height)
            target_width, target_height = round(width * scale), round(height * scale)
            resized = cv2.resize(source, (target_width, target_height),
                                 interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
            top = (IMAGE_SIZE - target_height) // 2
            left = (IMAGE_SIZE - target_width) // 2
            np.testing.assert_array_equal(
                clip[index, top:top + target_height, left:left + target_width],
                resized[:, :, ::-1],
            )

        features = flow_features(clip)
        self.assertEqual(FEATURE_DIM, 320)
        self.assertEqual(features.shape, (320,))
        self.assertTrue(np.isfinite(features).all())

    def test_known_confusion_metrics(self) -> None:
        scores = evaluate_predictions([0, 0, 0, 1, 1], [0, 1, 0, 0, 1])
        self.assertEqual(scores["confusion_matrix"], [[2, 1], [1, 1]])
        self.assertEqual((scores["true_negative"], scores["false_positive"],
                          scores["false_negative"], scores["true_positive"]),
                         (2, 1, 1, 1))
        self.assertAlmostEqual(scores["accuracy"], 0.6)
        self.assertAlmostEqual(scores["precision"], 0.5)
        self.assertAlmostEqual(scores["recall"], 0.5)
        self.assertAlmostEqual(scores["f1"], 0.5)

    def test_evaluator_requires_exactly_one_correctly_labeled_row_per_test_video(self) -> None:
        test_by_id = {record["video_id"]: record for record in self.test}
        rows = [{"video_id": record["video_id"], "label": record["label"],
                 "prediction": record["label"]} for record in self.test]

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "predictions.csv"

            def read(candidate: list[dict]):
                with path.open("w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=("video_id", "label", "prediction"))
                    writer.writeheader()
                    writer.writerows(candidate)
                return _read_predictions(path, test_by_id)

            metrics, details = read(rows)
            self.assertEqual(metrics["confusion_matrix"], [[50, 0], [0, 50]])
            self.assertEqual(details["test_videos"], 100)

            with self.subTest(case="missing"), self.assertRaisesRegex(ValueError, "missing"):
                read(rows[:-1])
            with self.subTest(case="duplicate"), self.assertRaisesRegex(ValueError, "duplicate"):
                read(rows + [rows[0]])
            mislabeled = [row.copy() for row in rows]
            mislabeled[0]["label"] = 1 - mislabeled[0]["label"]
            with self.subTest(case="wrong label"), self.assertRaisesRegex(ValueError, "label does not match"):
                read(mislabeled)

            one_negative = next(index for index, row in enumerate(rows) if row["label"] == 0)
            one_positive = next(index for index, row in enumerate(rows) if row["label"] == 1)
            imperfect = [row.copy() for row in rows]
            imperfect[one_negative]["prediction"] = 1
            imperfect[one_positive]["prediction"] = 0
            metrics, _ = read(imperfect)
            self.assertEqual(metrics["confusion_matrix"], [[49, 1], [1, 49]])
            self.assertAlmostEqual(metrics["accuracy"], 0.98)
            self.assertAlmostEqual(metrics["precision"], 0.98)
            self.assertAlmostEqual(metrics["recall"], 0.98)
            self.assertAlmostEqual(metrics["f1"], 0.98)

    def test_movinet_a0_builds_from_scratch_without_download(self) -> None:
        from movinets import MoViNet
        from movinets.config import _C
        from src.movinet.main import ARCHITECTURE, build_model

        with (mock.patch("movinets.MoViNet", wraps=MoViNet) as constructor,
              mock.patch("torch.hub.load_state_dict_from_url",
                         side_effect=AssertionError("pretrained weights requested")) as download):
            model = build_model()

        self.assertIsInstance(model.network, MoViNet)
        self.assertIs(constructor.call_args.args[0], _C.MODEL.MoViNetA0)
        self.assertIs(constructor.call_args.kwargs["pretrained"], False)
        self.assertIs(constructor.call_args.kwargs["causal"], False)
        self.assertEqual(constructor.call_args.kwargs["num_classes"], 1)
        self.assertEqual(ARCHITECTURE["pretrained"], False)
        download.assert_not_called()


class LabelTests(unittest.TestCase):
    def test_labels_are_parsed_without_truthiness_shortcut(self):
        for raw, expected in ((True, 1), (False, 0), (" true ", 1), ("FALSE", 0), (0, 0), (1, 1)):
            with self.subTest(raw=raw):
                parsed = parse_label_data({"label": raw, "timestamp": 2})
                self.assertEqual(parsed["label"], expected)
                self.assertFalse(parsed["errors"])
        for raw in ("no", "1", 2, None, [], 0.5):
            with self.subTest(raw=raw):
                self.assertIsNone(parse_label_data({"label": raw})["label"])
                self.assertIn("invalid_action_label", parse_label_data({"label": raw})["errors"])

    def test_invalid_json_shape_and_metadata(self):
        self.assertIn("label_json_not_object", parse_label_data([])["errors"])
        self.assertIn("metadata_not_object", parse_label_data({"label": True, "metadata": []})["errors"])

    def test_timestamp_defects_never_change_positive_label(self):
        for timestamp in (None, "3", True, float("nan"), float("inf"), -2):
            with self.subTest(timestamp=timestamp):
                parsed = parse_label_data({"label": "true", "timestamp": timestamp})
                self.assertEqual(parsed["label"], 1)
                self.assertFalse(parsed["errors"])
                self.assertTrue(parsed["timestamp_issues"])
        parsed = parse_label_data({"label": "false", "timestamp": -1})
        self.assertFalse(parsed["timestamp_issues"])


class FrameAndHashTests(unittest.TestCase):
    def test_jpeg_is_decoded_not_only_checked_for_nonempty_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good, corrupt, empty = root / "good.jpg", root / "corrupt.jpg", root / "empty.jpg"
            cv2.imwrite(str(good), np.full((12, 15, 3), 80, dtype=np.uint8))
            corrupt.write_bytes(b"nonempty but not an image")
            empty.touch()
            decoded = decode_frame(good)
            self.assertTrue(decoded["valid"])
            self.assertEqual((decoded["height"], decoded["width"]), (12, 15))
            for path in (corrupt, empty, root / "missing.jpg"):
                self.assertFalse(decode_frame(path)["valid"])

    def test_hash_cache_reuses_and_invalidates_changed_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            video, cache_path = root / "sample.mp4", root / "hashes.jsonl"
            video.write_bytes(b"first payload")
            first, reused = hash_video(video, {}, cache_path)
            self.assertFalse(reused)
            cached = load_hash_cache(cache_path)
            second, reused = hash_video(video, cached, cache_path)
            self.assertTrue(reused)
            self.assertEqual(first, second)
            video.write_bytes(b"a different payload, with different length")
            third, reused = hash_video(video, cached, cache_path)
            self.assertFalse(reused)
            self.assertNotEqual(first, third)
            with cache_path.open("a") as handle:
                handle.write("{interrupted")
            self.assertEqual(load_hash_cache(cache_path)[str(video.resolve())]["sha256"], third)
            video.write_bytes(b"new payload after an interrupted cache append")
            fourth, reused = hash_video(video, load_hash_cache(cache_path), cache_path)
            self.assertFalse(reused)
            self.assertEqual(load_hash_cache(cache_path)[str(video.resolve())]["sha256"], fourth)

    def test_duplicate_groups_keep_split_and_label_conflicts(self):
        records = [
            {"sha256": "same", "video_id": "a.mp4", "source_video": "a", "original_split": "train", "model_split": "train", "label": 0},
            {"sha256": "same", "video_id": "b.mp4", "source_video": "b", "original_split": "test", "model_split": "test", "label": 1},
            {"sha256": "other", "video_id": "c.mp4", "source_video": "c", "original_split": "train", "model_split": "validation", "label": 0},
        ]
        groups = duplicate_groups(records)
        self.assertEqual(len(groups), 1)
        self.assertTrue(groups[0]["crosses_official_split"])
        self.assertTrue(groups[0]["crosses_model_split"])
        self.assertTrue(groups[0]["conflicting_labels"])
        self.assertEqual(len(groups[0]["members"]), 2)


class LitteringCleaningTests(unittest.TestCase):
    def test_bbox_rejects_invalid_numbers_nonpositive_sizes_and_no_intersection(self):
        cases = [
            ([float("nan"), 0, 2, 2], "invalid_bbox_format_or_number"),
            ([0, float("inf"), 2, 2], "invalid_bbox_format_or_number"),
            ([0, 0, True, 2], "invalid_bbox_format_or_number"),
            ([0, 0, "2", 2], "invalid_bbox_format_or_number"),
            ([0, 0, 2], "invalid_bbox_format_or_number"),
            ([0, 0, 0, 2], "nonpositive_bbox_size"),
            ([0, 0, 2, -1], "nonpositive_bbox_size"),
            ([101, 0, 10, 10], "bbox_outside_image"),
            ([-10, 0, 10, 10], "bbox_outside_image"),
            ([0, 100, 10, 10], "bbox_outside_image"),
        ]
        for box, reason in cases:
            with self.subTest(box=box):
                self.assertEqual(sanitize_bbox(box, 100, 100), (None, reason))

    def test_large_bbox_clipping_is_quarantined_instead_of_relabelled(self):
        for box in ([-10, 5, 20, 20], [90, 5, 20, 20], [5, 90, 20, 20]):
            with self.subTest(box=box):
                self.assertEqual(sanitize_bbox(box, 100, 100),
                                 (None, "bbox_large_clip_requires_review"))

    def test_small_edge_rounding_has_correct_normalized_yolo_coordinates(self):
        box, action = sanitize_bbox([-0.1, 5, 50.1, 20], 100, 50)
        self.assertEqual(action, "clipped_small_edge_rounding")
        np.testing.assert_allclose(box, [0, 5, 50, 20])
        self.assertEqual(yolo_lines([box], 100, 50),
                         ["0 0.25000000 0.30000000 0.50000000 0.40000000"])
        self.assertEqual(sanitize_bbox([0, 0, 100, 50], 100, 50),
                         ([0.0, 0.0, 100.0, 50.0], "unchanged"))

    def test_decoding_finds_corrupt_images_and_same_pixels_with_different_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            corrupt = root / "broken.png"
            corrupt.write_bytes(b"not an encoded image")
            invalid = inspect_image(corrupt)
            self.assertFalse(invalid["valid"])
            self.assertEqual(invalid["error"], "image_decode_failed")

            pixels = np.random.default_rng(32).integers(0, 256, (24, 32, 3), dtype=np.uint8)
            first, second = root / "plain.png", root / "compressed.png"
            self.assertTrue(cv2.imwrite(str(first), pixels, [cv2.IMWRITE_PNG_COMPRESSION, 0]))
            self.assertTrue(cv2.imwrite(str(second), pixels, [cv2.IMWRITE_PNG_COMPRESSION, 9]))
            a, b = inspect_image(first), inspect_image(second)
            self.assertTrue(a["valid"] and b["valid"])
            self.assertEqual((a["width"], a["height"]), (32, 24))
            self.assertNotEqual(a["sha256"], b["sha256"])
            self.assertEqual(a["pixel_sha256"], b["pixel_sha256"])

    @staticmethod
    def duplicate_record(path, split, boxes):
        return {"path": path, "source_split": split, "pixel_sha256": "same_pixels",
                "boxes": boxes, "errors": []}

    def test_exact_duplicate_removal_keeps_validation_and_ignores_box_order(self):
        train = self.duplicate_record("train/example.png", "train", [[1, 2, 3, 4], [5, 6, 7, 8]])
        valid = self.duplicate_record("valid/example.png", "valid", [[5, 6, 7, 8], [1, 2, 3, 4]])
        decisions = exclude_confirmed_duplicates([train, valid])
        self.assertEqual(valid["errors"], [])
        self.assertEqual(train["errors"], ["confirmed_exact_duplicate_excluded"])
        self.assertEqual(decisions[0]["kept"], valid["path"])
        self.assertTrue(decisions[0]["cross_split"])

    def test_identical_pixels_with_conflicting_annotations_quarantine_both(self):
        records = [self.duplicate_record("train/a.png", "train", [[1, 2, 3, 4]]),
                   self.duplicate_record("valid/a.png", "valid", [[1, 2, 3, 5]])]
        decisions = exclude_confirmed_duplicates(records)
        for record in records:
            self.assertEqual(record["errors"],
                             ["same_pixels_conflicting_annotations_requires_review"])
        self.assertEqual(decisions[0]["action"], "hold_conflicting_annotations")
        self.assertNotIn("kept", decisions[0])

    def test_export_preserves_unrelated_existing_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            sentinel = output / "important.txt"
            sentinel.write_text("keep this", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                export_yolo([], output, "new-signature")
            self.assertEqual(sentinel.read_text(), "keep this")
            self.assertEqual(list(output.iterdir()), [sentinel])

    def test_export_idempotence_and_changed_labels_fail_without_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.png"
            self.assertTrue(cv2.imwrite(str(source), np.full((12, 16, 3), 70, dtype=np.uint8)))
            source_bytes = source.read_bytes()
            records = [{"include": True, "source_split": "train", "image_id": 7,
                        "path": str(source), "yolo_text": "0 0.5 0.5 0.5 0.5\n"}]
            output = root / "clean"
            initial = export_yolo(records, output, "fixed-signature")
            image = output / "images/train/00000007_source.png"
            label = output / "labels/train/00000007_source.txt"
            self.assertEqual(initial["status"], "created")
            self.assertEqual(image.read_bytes(), source_bytes)
            self.assertEqual(label.read_text(), records[0]["yolo_text"])
            self.assertTrue((output / "data.yaml").is_file())

            repeated = export_yolo(records, output, "fixed-signature")
            self.assertEqual(repeated["status"], "reused_matching_export")
            with self.assertRaises(FileExistsError):
                export_yolo(records, output, "different-signature")
            label.write_text("changed by another process", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete or changed"):
                export_yolo(records, output, "fixed-signature")
            self.assertEqual(label.read_text(), "changed by another process")
            self.assertEqual(source.read_bytes(), source_bytes)

    @staticmethod
    def write_fixture_split(source, split, images, annotations):
        directory = source / split
        directory.mkdir(parents=True)
        payload = {"images": images, "annotations": annotations,
                   "categories": [{"id": 1, "name": "littering"}]}
        (directory / "_annotations.coco.json").write_text(json.dumps(payload), encoding="utf-8")
        for index, image in enumerate(images):
            pixels = np.random.default_rng(index + (900 if split == "valid" else 100)).integers(
                0, 256, (20, 30, 3), dtype=np.uint8)
            if not cv2.imwrite(str(directory / image["file_name"]), pixels):
                raise AssertionError("Could not write fixture image")

    def test_audit_quarantines_dimension_mismatch_and_invalid_bbox_without_modifying_sources(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, reports = root / "raw", root / "reports"
            self.write_fixture_split(source, "train", [
                {"id": 1, "file_name": "good.png", "width": 30, "height": 20},
                {"id": 2, "file_name": "mismatch.png", "width": 99, "height": 20},
                {"id": 3, "file_name": "bad_box.png", "width": 30, "height": 20},
            ], [
                {"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 10, 10]},
                {"id": 2, "image_id": 2, "category_id": 1, "bbox": [1, 2, 10, 10]},
                {"id": 3, "image_id": 3, "category_id": 1, "bbox": [-10, 2, 20, 10]},
            ])
            self.write_fixture_split(source, "valid", [
                {"id": 1, "file_name": "valid.png", "width": 30, "height": 20},
            ], [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [1, 2, 10, 10]}])
            before = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}

            with contextlib.redirect_stdout(io.StringIO()):
                report = clean(source, reports, export_root=None, workers=1)

            self.assertEqual(report["summary"]["raw_images"], 4)
            self.assertEqual(report["summary"]["included_clean_image_records"], 2)
            self.assertEqual(report["summary"]["images_quarantined_from_clean_export"], 2)
            self.assertFalse(report["raw_files_modified"])
            self.assertIsNone(report["export"])
            with (reports / "littering_cleaning_manifest.csv").open(encoding="utf-8-sig", newline="") as stream:
                manifest = {row["source_file_name"]: row for row in csv.DictReader(stream)}
            self.assertIn("coco_dimensions_disagree_with_decoded_image",
                          json.loads(manifest["mismatch.png"]["errors"]))
            self.assertIn("bbox_large_clip_requires_review", json.loads(manifest["bad_box.png"]["errors"]))
            self.assertEqual(manifest["good.png"]["include"], "True")
            after = {str(p.relative_to(source)): p.read_bytes() for p in source.rglob("*") if p.is_file()}
            self.assertEqual(before, after)
            self.assertEqual({p.name for p in root.iterdir()}, {"raw", "reports"})


if __name__ == "__main__":
    unittest.main()
