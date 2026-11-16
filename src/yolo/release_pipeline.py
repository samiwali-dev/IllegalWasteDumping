"""Shared temporal release analysis for recorded videos and live camera windows."""
from pathlib import Path
from collections import OrderedDict
import threading
import time
import math

from ultralytics import YOLO

from src.yolo.temporal import ReleaseTracker, verify_source, still_attached

ACTOR_MODEL = Path(__file__).with_name('actors.pt')


def overlap(a, b):
    intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / max(1e-9, union)


_OBJECT_MODELS = {}
_OBJECT_LOCK = threading.Lock()


def review_motion_candidate(candidate, frames, timestamps, model_path, confidence, device):
    """Validate a measured falling path, then optionally confirm its object crop.

    Motion establishes a review candidate only; a learned object match is kept
    separate and neither result establishes an illegal dumping event.
    """
    if not isinstance(candidate, dict) or not timestamps:
        return None
    stamp = candidate.get('timestamp_ms')
    if not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
        return None
    index = min(range(len(timestamps)), key=lambda i: abs(timestamps[i]-stamp))
    if abs(timestamps[index]-stamp) > 100:
        return None
    frame = frames[index]
    height, width = frame.shape[:2]
    accepted = []
    def finite(value):
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)
    for detection in (candidate.get('detections') or [])[:8]:
        if not isinstance(detection, dict) or detection.get('reason') != 'falling_object':
            continue
        box = detection.get('norm_box')
        trail = detection.get('trail')
        if (not isinstance(box, (list, tuple)) or len(box) != 4
                or not all(finite(v) and 0 <= v <= 1 for v in box)
                or box[2] <= box[0] or box[3] <= box[1]
                or not isinstance(trail, list) or len(trail) < 4):
            continue
        if not all(isinstance(p, dict) and all(finite(p.get(k)) for k in ('cx', 'cy', 't'))
                   and 0 <= p['cx'] <= 1 and 0 <= p['cy'] <= 1 for p in trail):
            continue
        if any(b['t'] <= a['t'] for a, b in zip(trail, trail[1:])):
            continue
        steps = [b['cy']-a['cy'] for a, b in zip(trail, trail[1:])]
        downward = trail[-1]['cy']-trail[0]['cy']
        if (downward < .08 or downward / max(1e-9, sum(abs(v) for v in steps)) < .6
                or sum(v > 0 for v in steps) < .6 * len(steps)):
            continue
        absolute = [box[0]*width, box[1]*height, box[2]*width, box[3]*height]
        event = dict(reason='released_object', source='motion_tracker', confidence=None,
                     class_name='release_candidate', bbox_xyxy=absolute, norm_box=list(box),
                     trail=[dict(cx=p['cx'], cy=p['cy'], t=p['t']) for p in trail],
                     timestamp_ms=timestamps[index], source_frame_index=index,
                     source_frame_offset=index, motion_verified=True, object_confirmed=False)
        if Path(model_path).is_file():
            pad = max(48, round(max(absolute[2]-absolute[0], absolute[3]-absolute[1])*2))
            x1, y1 = max(0, int(absolute[0])-pad), max(0, int(absolute[1])-pad)
            x2, y2 = min(width, int(absolute[2])+pad), min(height, int(absolute[3])+pad)
            try:
                with _OBJECT_LOCK:
                    key = str(model_path)
                    if key not in _OBJECT_MODELS:
                        _OBJECT_MODELS[key] = YOLO(key)
                    result = _OBJECT_MODELS[key].predict(frame[y1:y2, x1:x2], conf=confidence,
                                                        imgsz=640, rect=False, device=device, verbose=False)[0]
                matches = []
                for predicted, score in zip(result.boxes.xyxy.cpu().tolist(), result.boxes.conf.cpu().tolist()):
                    learned = [predicted[0]+x1, predicted[1]+y1, predicted[2]+x1, predicted[3]+y1]
                    if overlap(absolute, learned) >= .1:
                        matches.append((score, learned))
                if matches:
                    score, learned = max(matches, key=lambda match: match[0])
                    event.update(object_confirmed=True, confidence=float(score), source='yolov8n',
                                 class_name='litter_object', bbox_xyxy=learned,
                                 norm_box=[learned[0]/width, learned[1]/height, learned[2]/width, learned[3]/height])
            except Exception as exc:
                event['object_confirmation_error'] = str(exc)
        accepted.append(event)
    return (accepted, index) if accepted else None


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
