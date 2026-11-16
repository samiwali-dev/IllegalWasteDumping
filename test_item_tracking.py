"""Regression checks using model observations, not any official test video."""
from types import SimpleNamespace
import unittest
from unittest import mock

import cv2
import numpy as np
import torch
from src.yolo.item_tracking import ItemReleaseDetector


class FakeModel:
    def __init__(self, boxes, score):
        self.frames = boxes
        self.index = 0
        self.score = score

    def predict(self, *args, **kwargs):
        boxes = self.frames[self.index]
        self.index += 1
        return [SimpleNamespace(boxes=SimpleNamespace(xyxy=torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                                                     conf=torch.tensor([self.score]*len(boxes)),
                                                     cls=torch.zeros(len(boxes))), names={0: 'person'})]


def run(objects, actors):
    object_model, actor_model = FakeModel(objects, .72), FakeModel(actors, .91)
    with mock.patch('src.yolo.item_tracking.YOLO', side_effect=[object_model, actor_model]):
        detector = ItemReleaseDetector('unused', .35, 'cpu', 'unused')
    events = []
    for i, boxes in enumerate(objects):
        frame = np.full((240, 320, 3), 180, np.uint8)
        for x1, y1, x2, y2 in boxes:
            cv2.rectangle(frame, (x1, y1), (x2-1, y2-1), (40, 70, 110), -1)
        events.extend(detector.update(frame, i*.3, i))
    return events


class LearnedItemTrackingTests(unittest.TestCase):
    def test_static_item_is_not_an_event_even_when_a_person_is_near(self):
        events = run([[[180, 70, 300, 200]]]*10, [[[160, 0, 319, 220]]]*10)
        self.assertEqual(events, [])

    def test_held_item_can_move_away_from_actor_and_become_a_candidate(self):
        objects = [[[80+min(i, 7)*12, 100, 104+min(i, 7)*12, 130]] for i in range(10)]
        actors = [[[50, 40, 120, 190]]] * 3 + [[[0, 40, 55, 190]]] * 7
        events = run(objects, actors)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event['reason'], 'released_object')
        self.assertAlmostEqual(event['confidence'], .72, places=5)
        self.assertAlmostEqual(event['source_actor']['confidence'], .91, places=5)
        self.assertGreaterEqual(len(event['trail']), 3)

    def test_placed_item_is_detected_without_inventing_a_fall_arrow(self):
        objects = [[]] + [[[160, 160, 220, 210]]] * 9
        actors = [[[130, 30, 220, 190]]] * 4 + [[[250, 30, 300, 190]]] * 6
        events = run(objects, actors)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['reason'], 'left_behind_object')
        self.assertEqual(events[0]['trail'], [])

    def test_carried_item_stays_attached_and_never_becomes_a_release(self):
        objects = [[[80+i*8, 100, 104+i*8, 130]] for i in range(10)]
        actors = [[[50+i*8, 40, 125+i*8, 190]] for i in range(10)]
        self.assertEqual(run(objects, actors), [])

    def test_a_new_item_without_an_observed_source_is_only_an_object(self):
        objects = [[]] + [[[160, 160, 220, 210]]] * 9
        self.assertEqual(run(objects, [[]]*10), [])


if __name__ == '__main__':
    unittest.main()
