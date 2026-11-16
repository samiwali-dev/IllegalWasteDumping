"""Physical-motion regressions; real-video checks are recorded separately."""
import unittest
import cv2
import numpy as np
from unittest import mock
from types import SimpleNamespace

from src.yolo.temporal import ReleaseTracker, draw_candidates
from src.yolo.release_pipeline import LiveReleasePool


def run_clip(frames):
    tracker = ReleaseTracker()
    events = []
    for i, frame in enumerate(frames):
        events.extend(tracker.update(frame, i/15))
    return tracker, events


class ReleaseTrackingTests(unittest.TestCase):
    def test_overlapping_live_requests_do_not_replay_old_frames(self):
        detector = SimpleNamespace(tracker=SimpleNamespace(camera_moving=False), update=mock.Mock(return_value=[]))
        pool = LiveReleasePool()
        frames = [np.zeros((10, 10, 3), np.uint8)] * 3
        with mock.patch('src.yolo.release_pipeline.ReleaseDetector', return_value=detector):
            pool.process(frames, [1000, 1060, 1120], 'camera-one', 'unused.pt', .65, 'cpu')
            pool.process(frames, [1060, 1120, 1180], 'camera-one', 'unused.pt', .65, 'cpu')
        self.assertEqual(detector.update.call_count, 4)
        times = [call.args[1] for call in detector.update.call_args_list]
        self.assertEqual(times, [0, .06, .12, .18])

    def test_separate_camera_sessions_do_not_share_backgrounds(self):
        pool = LiveReleasePool()
        frames = [np.zeros((10, 10, 3), np.uint8)]
        def make_detector(*args):
            return SimpleNamespace(tracker=SimpleNamespace(camera_moving=False), update=mock.Mock(return_value=[]))
        with mock.patch('src.yolo.release_pipeline.ReleaseDetector', side_effect=make_detector):
            pool.process(frames, [1000], 'camera-one', 'unused.pt', .65, 'cpu')
            pool.process(frames, [1000], 'camera-two', 'unused.pt', .65, 'cpu')
        self.assertIsNot(pool.sessions['camera-one']['detector'], pool.sessions['camera-two']['detector'])

    def test_static_black_object_is_not_a_release(self):
        image = np.full((360, 640, 3), 170, np.uint8)
        cv2.rectangle(image, (440, 120), (590, 300), (0, 0, 0), -1)
        _, events = run_clip([image.copy() for _ in range(30)])
        self.assertEqual(events, [])

    def test_walk_across_scene_does_not_make_a_small_release(self):
        frames = []
        for x in range(40, 320, 8):
            image = np.full((360, 640, 3), 170, np.uint8)
            cv2.rectangle(image, (x, 80), (x+65, 300), (20, 20, 20), -1)
            frames.append(image)
        _, events = run_clip(frames)
        self.assertEqual(events, [])

    def test_motion_then_settling_tracks_actual_position_without_fake_confidence(self):
        frames = []
        for i in range(30):
            image = np.full((360, 640, 3), 170, np.uint8)
            if i:
                x, y = 230+min(i, 9)*4, 80+min(i, 9)*6
                cv2.rectangle(image, (x, y), (x+14, y+14), (15, 25, 15), -1)
            frames.append(image)
        _, events = run_clip(frames)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertIsNone(event['confidence'])
        self.assertEqual(event['source'], 'motion_tracker')
        self.assertGreaterEqual(len(event['trail']), 3)
        x1, y1, x2, y2 = event['bbox_xyxy']
        self.assertLessEqual(x1, 270)
        self.assertGreaterEqual(x2, 276)
        self.assertLessEqual(y1, 138)
        self.assertGreaterEqual(y2, 144)
        original = frames[-1].copy()
        plotted = draw_candidates(original, [event])
        self.assertTrue(np.array_equal(original, frames[-1]))
        self.assertFalse(np.array_equal(plotted, original))

    def test_camera_pan_resets_motion_history(self):
        image = np.random.default_rng(12).integers(0, 256, (360, 640, 3), dtype=np.uint8)
        frames = [np.roll(image, i*12, axis=1) for i in range(20)]
        tracker, events = run_clip(frames)
        self.assertTrue(tracker.camera_moving)
        self.assertEqual(events, [])

    def test_illumination_change_does_not_turn_static_item_into_a_release(self):
        image = np.full((360, 640, 3), 150, np.uint8)
        cv2.rectangle(image, (250, 160), (278, 185), (30, 30, 30), -1)
        frames = [cv2.add(image, np.full_like(image, i)) for i in range(30)]
        _, events = run_clip(frames)
        self.assertEqual(events, [])


if __name__ == '__main__':
    unittest.main()
