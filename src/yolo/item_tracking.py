"""Track detected items through actor separation, including placement and throws.

This development strategy uses learned item boxes, not a fixed tiny-projectile
shape. It produces review candidates, not a probability that an act is illegal.
"""
from collections import deque
import math

import cv2
import numpy as np
from ultralytics import YOLO

from src.yolo.temporal import near


def iou(a, b):
    area = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
    total = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - area
    return area/max(1, total)


class ItemReleaseDetector:
    algorithm_version = 'object_tracks_v2'

    def __init__(self, model_path, confidence, device, actor_path):
        self.object_model = YOLO(str(model_path))
        self.actor_model = YOLO(str(actor_path))
        self.confidence = confidence
        self.device = device
        self.frames_processed = 0
        self.reset_tracking()

    @property
    def tracker(self):
        return self

    def reset_tracking(self):
        self.reference = None
        self.previous = None
        self.scene_started = None
        self.camera_moving = False
        self.last_processed = -1
        self.items = []
        self.actors = deque()
        self.next_id = 1

    def update(self, frame, seconds, frame_index):
        if seconds-self.last_processed < .25:
            return []
        self.last_processed = seconds
        self.frames_processed += 1
        original_h, original_w = frame.shape[:2]
        scale = min(1, 960/original_w)
        image = cv2.resize(frame, (round(original_w*scale), round(original_h*scale)))
        h, w = image.shape[:2]
        small = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.float32)
        if self.reference is None:
            self.reference = cv2.GaussianBlur(image, (3, 3), 0)
            self.scene_started = seconds
        if self.previous is not None:
            shift, response = cv2.phaseCorrelate(self.previous, small)
            textured = sum(np.std(tile) > 4 for row in np.array_split(small, 3)
                           for tile in np.array_split(row, 3, axis=1))
            self.camera_moving = textured >= 4 and response > .3 and math.hypot(*shift) > 2
            if self.camera_moving:
                self.reference = cv2.GaussianBlur(image, (3, 3), 0)
                self.scene_started = seconds
                self.items.clear(); self.actors.clear()
        self.previous = small
        objects = self.object_model.predict(image, conf=self.confidence, imgsz=1280,
                                             rect=False, device=self.device, max_det=40, verbose=False)[0]
        actors = self.actor_model.predict(image, classes=[0, 1, 2, 3, 5, 7], conf=.35,
                                          imgsz=640, rect=False, device=self.device, verbose=False)[0]
        current_actors = [dict(box=box, confidence=score, class_name=actors.names[int(category)], seconds=seconds)
                          for box, score, category in zip(actors.boxes.xyxy.cpu().tolist(), actors.boxes.conf.cpu().tolist(), actors.boxes.cls.cpu().tolist())]
        self.actors.extend(current_actors)
        while self.actors and seconds-self.actors[0]['seconds'] > 2:
            self.actors.popleft()
        blurred = cv2.GaussianBlur(image, (3, 3), 0)
        delta = np.median((blurred[::8, ::8].astype(float)-self.reference[::8, ::8]).reshape(-1, 3), axis=0)
        observations = []
        for box, score in zip(objects.boxes.xyxy.cpu().tolist(), objects.boxes.conf.cpu().tolist()):
            x1, y1, x2, y2 = [int(v) for v in box]
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
            if x2-x1 < 3 or y2-y1 < 3:
                continue
            box = [x1, y1, x2, y2]
            difference = np.abs(blurred[y1:y2, x1:x2].astype(float)-self.reference[y1:y2, x1:x2]-delta).mean(axis=2)
            observations.append(dict(box=box, score=score, center=np.array([(x1+x2)/2, (y1+y2)/2]),
                                     color=image[y1:y2, x1:x2].mean(axis=(0, 1)),
                                     new_region=float(difference.mean()) >= 15 and float(np.mean(difference > 20)) >= .25))
        self.items = [item for item in self.items if seconds-item['last_seen'] <= 2]
        matches = []
        for i, item in enumerate(self.items):
            for j, observation in enumerate(observations):
                old, new = item['box'], observation['box']
                diagonal = math.hypot(old[2]-old[0], old[3]-old[1])
                distance = float(np.linalg.norm(item['center']-observation['center']))
                ratio = ((new[2]-new[0])*(new[3]-new[1])) / max(1, (old[2]-old[0])*(old[3]-old[1]))
                overlap = iou(old, new)
                if (overlap >= .1 or distance <= diagonal*.7) and .25 <= ratio <= 4 and np.linalg.norm(item['color']-observation['color']) < 100:
                    matches.append((overlap-distance/max(1, diagonal)*.1, i, j))
        used_items, used_observations = set(), set()
        for _, i, j in sorted(matches, reverse=True):
            if i in used_items or j in used_observations:
                continue
            item = self.items[i]
            item.update(observations[j], last_seen=seconds, seen=item['seen']+1)
            used_items.add(i); used_observations.add(j)
        for j, observation in enumerate(observations):
            if j in used_observations:
                continue
            self.items.append(dict(**observation, id=self.next_id, first_seen=seconds, last_seen=seconds,
                                   origin=observation['center'].copy(), seen=1, source_actor=None,
                                   detached_since=None, emitted=False, initial=not observation['new_region'], trail=[]))
            self.next_id += 1
        events = []
        for item in self.items:
            if item['last_seen'] != seconds:
                continue
            box = item['box']
            if item['source_actor'] is None:
                possible = [a for a in self.actors if near(box, a['box'], w*.015)]
                if possible:
                    item['source_actor'] = max(possible, key=lambda a: a['confidence'])
            attached = any(near(box, a['box'], w*.005) for a in current_actors)
            if attached:
                item['detached_since'] = None
            elif item['detached_since'] is None:
                item['detached_since'] = seconds
            point = dict(cx=float(item['center'][0]/w), cy=float(item['center'][1]/h), t=seconds)
            trail = item['trail']
            if not trail or math.hypot(point['cx']-trail[-1]['cx'], point['cy']-trail[-1]['cy'])*w > 3:
                trail.append(point)
                item['trail'] = trail[-24:]
            moved = np.linalg.norm(item['center']-item['origin']) > max(5, math.hypot(box[2]-box[0], box[3]-box[1])*.2)
            if (item['emitted'] or item['seen'] < 3 or not item['new_region'] or item['source_actor'] is None
                    or item['detached_since'] is None or seconds-item['detached_since'] < .5
                    or (item['initial'] and not moved) or self.camera_moving):
                continue
            item['emitted'] = True
            actor = item['source_actor']
            observed_motion = len(item['trail']) >= 3 and moved
            events.append(dict(track_id=item['id'], source='yolov8n', class_name='litter_object',
                               confidence=item['score'], object_confirmed=True,
                               reason='released_object' if observed_motion else 'left_behind_object',
                               bbox_xyxy=[v/scale for v in box], norm_box=[box[0]/w, box[1]/h, box[2]/w, box[3]/h],
                               source_actor=dict(class_name=actor['class_name'], confidence=actor['confidence']),
                               persistent_observations=item['seen'], trail=item['trail'] if observed_motion else [],
                               timestamp_seconds=seconds, source_frame_index=frame_index))
        return events
