"""Track a small object separating from an actor and then coming to rest.

This is a motion heuristic, not a trained action classifier. Static YOLO boxes
are not event candidates. A motion-only candidate never gets a YOLO percentage.
"""
from __future__ import annotations

import math
import cv2
import numpy as np


def near(box, actor, margin):
    return (box[0] < actor[2] + margin and box[2] > actor[0] - margin
            and box[1] < actor[3] + margin and box[3] > actor[1] - margin)


class ReleaseTracker:
    def __init__(self, width=960):
        self.width = width
        # Shadow removal would also discard dark objects on light pavement.
        self.background = cv2.createBackgroundSubtractorMOG2(
            history=150, varThreshold=32, detectShadows=False)
        self.tracks = []
        self.previous = None
        self.reference = None
        self.previous_full_gray = None
        self.previous_seconds = None
        self.next_id = 1
        self.camera_moving = False

    def update(self, frame, seconds):
        height, width = frame.shape[:2]
        scale = min(1, self.width / width)
        image = cv2.resize(frame, (round(width * scale), round(height * scale)))
        image = cv2.GaussianBlur(image, (3, 3), 0)
        h, w = image.shape[:2]
        full_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(full_gray, (160, 90)).astype(np.float32)
        if self.previous is None:
            self.background.apply(image, learningRate=1)
            self.previous = gray
            self.reference = image.copy()
            self.previous_full_gray = full_gray
            self.previous_seconds = seconds
            return []
        shift, response = cv2.phaseCorrelate(self.previous, gray)
        self.previous = gray
        # A moving item on a blank wall must not stand in for camera motion.
        # Require texture spread across the scene before trusting phase shift.
        textured = sum(np.std(tile) > 4 for row in np.array_split(gray, 3)
                       for tile in np.array_split(row, 3, axis=1))
        self.camera_moving = textured >= 4 and response > .25 and math.hypot(*shift) > .45
        mask = self.background.apply(image, learningRate=.012)
        if self.camera_moving or np.mean(mask > 0) > .45:
            self.tracks.clear()
            self.background.apply(image, learningRate=1)
            self.reference = image.copy()
            self.previous_full_gray = full_gray
            self.previous_seconds = seconds
            return []
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        global_delta = np.median((image[::8, ::8].astype(float) - self.reference[::8, ::8]).reshape(-1, 3), axis=0)

        def change(box):
            x1, y1, x2, y2 = box
            difference = np.abs(image[y1:y2, x1:x2].astype(float)
                                - self.reference[y1:y2, x1:x2] - global_delta).mean(axis=2)
            return float(difference.mean()), float(np.mean(difference > 25))
        _, labels, stats, centers = cv2.connectedComponentsWithStats(mask)
        blobs, actors = [], []
        for label, ((x, y, bw, bh, area), (cx, cy)) in enumerate(zip(stats[1:], centers[1:]), 1):
            box = [int(x), int(y), int(x + bw), int(y + bh)]
            if area > w * h * .007 and bh > h * .06:
                actors.append(box)
            if not (12 <= area <= w * h * .003 and 3 <= bw <= w * .12
                    and 3 <= bh <= h * .15 and .3 <= bw / bh <= 3.5):
                continue
            selected = labels[y:y+bh, x:x+bw] == label
            color = image[y:y+bh, x:x+bw][selected].mean(axis=0)
            blobs.append(dict(box=box, cx=float(cx), cy=float(cy), area=int(area),
                              color=color, seconds=seconds, appearance_change=change(box)[0]))

        # Sparse optical flow checks physical continuity. Forward/backward
        # agreement prevents a background speck jumping to an unrelated blob.
        flow_predictions = {}
        if self.tracks:
            starts = []
            for track in self.tracks:
                point = track['points'][-1]
                x1, y1, x2, y2 = point['box']
                starts.extend([(x1, y1), (min(w-1, x2), y1),
                               (x1, min(h-1, y2)), (min(w-1, x2), min(h-1, y2))])
            starts = np.float32(starts).reshape(-1, 1, 2)
            forward, valid, error = cv2.calcOpticalFlowPyrLK(self.previous_full_gray, full_gray, starts, None,
                                                           winSize=(15, 15), maxLevel=3)
            backward, back_valid, _ = cv2.calcOpticalFlowPyrLK(full_gray, self.previous_full_gray, forward, None,
                                                             winSize=(15, 15), maxLevel=3)
            for i, track in enumerate(self.tracks):
                deltas = []
                for j in range(i*4, i*4+4):
                    if valid[j, 0] and back_valid[j, 0] and error[j, 0] < 35 and np.linalg.norm(backward[j]-starts[j]) < 2:
                        deltas.append(forward[j, 0]-starts[j, 0])
                if len(deltas) >= 2:
                    last = track['points'][-1]
                    flow_predictions[i] = np.array([last['cx'], last['cy']]) + np.median(deltas, axis=0)

        pairs = []
        for i, track in enumerate(self.tracks):
            last = track['points'][-1]
            gap = seconds - last['seconds']
            if last['seconds'] != self.previous_seconds or i not in flow_predictions or not 0 < gap <= .22:
                continue
            for j, blob in enumerate(blobs):
                distance = math.hypot(blob['cx'] - last['cx'], blob['cy'] - last['cy'])
                color_distance = float(np.linalg.norm(blob['color'] - last['color']))
                ratio = max(blob['area'], last['area']) / min(blob['area'], last['area'])
                flow_distance = float(np.linalg.norm(flow_predictions[i]-[blob['cx'], blob['cy']]))
                diagonal = math.hypot(last['box'][2]-last['box'][0], last['box'][3]-last['box'][1])
                if (distance > max(12, w * .035 * gap / .067) or color_distance > 65 or ratio > 5
                        or flow_distance > max(3, diagonal * .25)):
                    continue
                pairs.append((flow_distance + color_distance * .2, i, j))
        assigned_tracks, assigned_blobs = set(), set()
        for _, i, j in sorted(pairs):
            if i in assigned_tracks or j in assigned_blobs:
                continue
            self.tracks[i]['points'].append(blobs[j])
            assigned_tracks.add(i)
            assigned_blobs.add(j)
        for j, blob in enumerate(blobs):
            if j not in assigned_blobs:
                self.tracks.append(dict(id=self.next_id, points=[blob], source_context=None, emitted=False))
                self.next_id += 1
        self.tracks = [t for t in self.tracks if seconds - t['points'][-1]['seconds'] <= .22]
        self.previous_full_gray = full_gray
        self.previous_seconds = seconds

        events = []
        for track in self.tracks:
            points = track['points']
            if len(points) > 90:
                points = track['points'] = points[-90:]
            last = points[-1]
            if last['seconds'] != seconds:
                continue
            age = seconds - points[0]['seconds']
            box = last['box']
            margin = max(4, (box[2] - box[0]) * .4)
            if age <= .25:
                sources = [a for a in actors if near(box, a, margin * 3)]
                if track['source_context'] is None:
                    actor = min(sources, key=lambda a: (a[2]-a[0])*(a[3]-a[1])) if sources else box
                    pad = max(30, round(max(actor[2]-actor[0], actor[3]-actor[1]) * .25))
                    if not sources:
                        # A stationary person can release an object too. Keep
                        # context for the semantic actor check, not just MOG motion.
                        pad = max(pad, round(h * .22))
                    x1, y1 = max(0, min(actor[0], box[0])-pad), max(0, min(actor[1], box[1])-pad)
                    x2, y2 = min(w, max(actor[2], box[2])+pad), min(h, max(actor[3], box[3])+pad)
                    track['source_context'] = dict(image=image[y1:y2, x1:x2].copy(),
                                                   contact_box=[box[0]-x1, box[1]-y1, box[2]-x1, box[3]-y1],
                                                   foreground_actor=bool(sources))
            if track['emitted'] or track['source_context'] is None or len(points) < 8 or not .5 <= age <= 2:
                continue
            # Only the recent release matters; don't join unrelated old motion.
            recent = [p for p in points if seconds - p['seconds'] <= 2]
            tail = [p for p in recent if seconds - p['seconds'] <= .35]
            if len(tail) < 4 or tail[-1]['seconds'] - tail[0]['seconds'] < .25:
                continue
            xy = np.array([[p['cx'], p['cy']] for p in recent])
            tail_xy = np.array([[p['cx'], p['cy']] for p in tail])
            spread = np.linalg.norm(tail_xy - tail_xy.mean(axis=0), axis=1).max()
            distance = float(np.linalg.norm(xy[-1] - xy[0]))
            path = float(np.linalg.norm(np.diff(xy, axis=0), axis=1).sum())
            if (spread > max(1.8, (box[2]-box[0]) * .2)
                    or distance < max(w * .012, math.hypot(box[2]-box[0], box[3]-box[1]))
                    or distance / max(1, path) < .55
                    or any(near(box, actor, margin) for actor in actors)):
                continue
            trail = []
            for point in recent:
                normalized = dict(cx=point['cx']/w, cy=point['cy']/h, t=point['seconds'])
                if not trail or math.hypot(normalized['cx']-trail[-1]['cx'], normalized['cy']-trail[-1]['cy']) * w > 1.5:
                    trail.append(normalized)
            if len(trail) < 3:
                continue
            # The landing position must contain something new, while the place
            # it passed through must return toward its original background.
            final_change, changed_fraction = change(box)
            restored_change, _ = change(recent[0]['box'])
            start_change = recent[0]['appearance_change']
            if (final_change < 22 or changed_fraction < .3 or start_change < 18
                    or restored_change > max(12, start_change * .6)):
                continue
            track['emitted'] = True
            events.append(dict(track_id=track['id'], reason='released_object', source='motion_tracker',
                               class_name='release_candidate', confidence=None,
                               bbox_xyxy=[v/scale for v in box],
                               norm_box=[box[0]/w, box[1]/h, box[2]/w, box[3]/h],
                               trail=trail, timestamp_seconds=seconds,
                               _source_context=track['source_context']))
        return events


def verify_source(event, actor_model, device='cpu'):
    """Require an actual COCO person/vehicle at the observed separation point."""
    context = event.pop('_source_context', None)
    if context is None:
        return False
    result = actor_model.predict(context['image'], classes=[0, 1, 2, 3, 5, 7],
                                 conf=.35, imgsz=640, rect=False, device=device, verbose=False)[0]
    for box, score, category in zip(result.boxes.xyxy.cpu().tolist(),
                                     result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
        if int(category) != 0 and not context['foreground_actor']:
            # A nearby parked vehicle is not evidence of an object being released.
            continue
        margin = max(12, min(box[2]-box[0], box[3]-box[1]) * .25)
        if near(context['contact_box'], box, margin):
            event['source_actor'] = dict(class_name=result.names[int(category)], confidence=score)
            return True
    return False


def still_attached(event, frame, actor_model, device='cpu'):
    """Reject a sleeve, wheel or carried item still inside a person/vehicle."""
    h, w = frame.shape[:2]
    box = event['bbox_xyxy']
    # Include the whole person, rather than asking COCO to identify a foot crop.
    pad = max(80, round(h * .5))
    x1, y1 = max(0, int(box[0])-pad), max(0, int(box[1])-pad)
    x2, y2 = min(w, int(box[2])+pad), min(h, int(box[3])+pad)
    result = actor_model.predict(frame[y1:y2, x1:x2], classes=[0, 1, 2, 3, 5, 7],
                                 conf=.35, imgsz=640, rect=False, device=device, verbose=False)[0]
    local = [box[0]-x1, box[1]-y1, box[2]-x1, box[3]-y1]
    return any(near(local, actor, max(4, (actor[3]-actor[1]) * .08))
               for actor in result.boxes.xyxy.cpu().tolist())


def draw_candidates(frame, detections):
    """Use actual trajectories; percentages are reserved for real YOLO scores."""
    image = frame.copy()
    h, w = image.shape[:2]
    for detection in detections:
        color = (250, 165, 96) if detection.get('source') == 'yolov8n' else (11, 158, 245)
        x1, y1, x2, y2 = [round(v) for v in detection['bbox_xyxy']]
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        score = detection.get('confidence')
        text = f'YOLO item {score:.0%} + release' if score is not None else 'Release candidate'
        text_y = max(24, y1 - 7)
        cv2.putText(image, text, (max(0, min(x1, w-240)), text_y), cv2.FONT_HERSHEY_SIMPLEX,
                    max(.5, w/2200), color, 2, cv2.LINE_AA)
        points = np.array([[round(p['cx']*w), round(p['cy']*h)] for p in detection['trail']], np.int32)
        if len(points) >= 2:
            cv2.polylines(image, [points], False, (11, 158, 245), 2, cv2.LINE_AA)
            cv2.arrowedLine(image, tuple(points[-2]), tuple(points[-1]), (11, 158, 245), 2,
                            cv2.LINE_AA, tipLength=.4)
    return image
