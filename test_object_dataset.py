"""Regression checks for the object-label conversion and leakage protections."""
import unittest

from src.yolo.prepare_objects import assign_splits, object_labels
from src.yolo.train_objects import iou


class ObjectDatasetTests(unittest.TestCase):
    def test_coco_object_box_keeps_its_location_after_resize(self):
        image = {"width": 1000, "height": 500}
        rows = object_labels(image, [{"id": 1, "bbox": [100, 100, 200, 100]}])
        self.assertEqual(rows, ["0 0.20000000 0.30000000 0.20000000 0.20000000"])

    def test_unannotated_image_is_not_automatically_a_negative(self):
        with self.assertRaises(ValueError):
            object_labels({"width": 1000, "height": 500}, [])

    def test_invalid_object_box_stops_preparation(self):
        with self.assertRaises(ValueError):
            object_labels({"width": 1000, "height": 500}, [{"id": 1, "bbox": [5000, 20, 10, 10]}])

    def test_duplicate_images_in_different_source_batches_stay_together(self):
        values = [0xffff0000ffff0000, 0xffff0000ffff0000, 0xaaaaaaaaaaaaaaaa,
                  0x5555555555555555, 0x1111111111111111, 0x2222222222222222]
        rows = [{"id": i, "source_file": f"batch_{i}/picture.jpg", "sha256": "duplicate" if i < 2 else str(i),
                 "phash": f"{value:016x}"} for i, value in enumerate(values)]
        counts, pairs = assign_splits(rows)
        self.assertEqual(rows[0]["split"], rows[1]["split"])
        self.assertIn([0, 1], pairs)
        self.assertTrue(all(counts[key] > 0 for key in ("train", "val", "test")))

    def test_detection_matching_rejects_nonoverlapping_and_person_sized_boxes(self):
        object_box = [.4, .4, .45, .45]
        self.assertEqual(iou(object_box, object_box), 1)
        self.assertEqual(iou(object_box, [.6, .6, .7, .7]), 0)
        self.assertLess(iou(object_box, [.3, .1, .6, .9]), .5)


if __name__ == "__main__":
    unittest.main()
