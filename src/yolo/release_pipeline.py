"""Shared temporal release analysis for recorded videos and live camera windows."""
from pathlib import Path
from collections import OrderedDict
import threading
import time

from ultralytics import YOLO

from src.yolo.temporal import ReleaseTracker, verify_source, still_attached

ACTOR_MODEL = Path(__file__).with_name('actors.pt')


def overlap(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / max(1e-9, union)


class ReleaseDetector:
    def __init__(self, model_path, confidence=.65, device='cpu'):
        self.tracker = ReleaseTracker()
        self.model_path = model_path
        self.confidence = confidence
        self.device = device
        self.actor_model = None
        self.object_model = None

    def update(self, frame, seconds, frame_index):
        accepted = []
        for event in self.tracker.update(frame, seconds):
            if self.actor_model is None:
                if not ACTOR_MODEL.is_file():
                    raise FileNotFoundError(f'Pretrained person/vehicle detector missing: {ACTOR_MODEL}')
                self.actor_model = YOLO(str(ACTOR_MODEL))
            if (not verify_source(event, self.actor_model, self.device)
                    or still_attached(event, frame, self.actor_model, self.device)):
                continue
            event['source_frame_index'] = frame_index
            event['object_confirmed'] = False
            # Check a local crop so a small projectile isn't reduced to a few
            # pixels in a full-scene resize. Unmatched predictions are ignored.
            height, width = frame.shape[:2]
            box = event['bbox_xyxy']
            pad = max(48, round(max(box[2]-box[0], box[3]-box[1]) * 2))
            x1, y1 = max(0, int(box[0])-pad), max(0, int(box[1])-pad)
            x2, y2 = min(width, int(box[2])+pad), min(height, int(box[3])+pad)
            if self.object_model is None:
                self.object_model = YOLO(str(self.model_path))
            result = self.object_model.predict(frame[y1:y2, x1:x2], conf=self.confidence,
                                               imgsz=640, rect=False, device=self.device, verbose=False)[0]
            for predicted, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                absolute = [predicted[0]+x1, predicted[1]+y1, predicted[2]+x1, predicted[3]+y1]
                if overlap(box, absolute) >= .2:
                    event.update(object_confirmed=True, confidence=score, source='yolov8n', class_name='litter_object')
                    break
            accepted.append(event)
        return accepted


class LiveReleasePool:
    """Keep one bounded tracker per camera session across overlapping windows."""
    def __init__(self):
        self.sessions = OrderedDict()
        self.lock = threading.Lock()

    def process(self, frames, timestamps, session_id, model_path, confidence, device):
        if not isinstance(session_id, str) or not 1 <= len(session_id) <= 128:
            raise ValueError('Invalid camera session identifier')
        now = time.monotonic()
        with self.lock:
            expired = [key for key, value in self.sessions.items() if now-value['seen'] > 120]
            for key in expired:
                self.sessions.pop(key)
            if session_id not in self.sessions:
                while len(self.sessions) >= 4:
                    self.sessions.popitem(last=False)
                self.sessions[session_id] = dict(detector=ReleaseDetector(model_path, confidence, device),
                                                 first=timestamps[0], last=float('-inf'), index=0,
                                                 lock=threading.Lock(), seen=now)
            session = self.sessions[session_id]
            session['seen'] = now
            self.sessions.move_to_end(session_id)
        events = []
        with session['lock']:
            for offset, (frame, timestamp) in enumerate(zip(frames, timestamps)):
                if timestamp <= session['last']:
                    continue
                if timestamp-session['last'] > 500 and session['last'] != float('-inf'):
                    session['detector'].tracker = ReleaseTracker()
                for event in session['detector'].update(frame, (timestamp-session['first'])/1000, session['index']):
                    event['source_frame_offset'] = offset
                    events.append(event)
                session['last'] = timestamp
                session['index'] += 1
        peak = events[-1]['source_frame_offset'] if events else len(frames)-1
        return events, peak, session['detector'].tracker.camera_moving
