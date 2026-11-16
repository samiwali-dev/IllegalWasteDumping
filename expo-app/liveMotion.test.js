const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const test = require('node:test');
const { motionTrailGeometry, yoloDetectionLabel } = require('./motionGeometry');
const source = readFileSync(join(__dirname, 'liveEdgeAi.js'), 'utf8')
  .replace(/^import .*;\s*$/gm, '').replace(/^export /gm, '');
const weights = JSON.parse(readFileSync(join(__dirname, 'nnWeights.json'), 'utf8'));
const api = new Function('Platform', 'nnWeights', 'motionTrailGeometry', `${source}\nreturn { analyzeLiveFramesOnEdge, detectDumpedObjectsOnly };`)
  ({ OS: 'test' }, weights, motionTrailGeometry);

function clip(positions, largePerson = false) {
  return positions.map((position, i) => {
    const rgb = new Float32Array(112 * 112 * 3).fill(0.8);
    const gray = new Float32Array(112 * 112).fill(204);
    const left = largePerson ? position : 49, top = largePerson ? 20 : position;
    const width = largePerson ? 40 : 10, height = largePerson ? 70 : 10;
    for (let y = top; position != null && y < Math.min(112, top + height); y++) {
      for (let x = left; x < Math.min(112, left + width); x++) {
        rgb.fill(0, (y * 112 + x) * 3, (y * 112 + x) * 3 + 3);
        gray[y * 112 + x] = 0;
      }
    }
    return { isDecodedFrame: true, rgb, gray, ts: 1000 + i * 60, uri: null };
  });
}

test('all interface text is English', () => {
  for (const path of ['App.js', 'liveEdgeAi.js', 'app.json']) {
    assert.doesNotMatch(readFileSync(join(__dirname, path), 'utf8'), /[\u0600-\u06ff]/);
  }
});

test('a stationary black object does not trigger dumping', async () => {
  const result = await api.analyzeLiveFramesOnEdge(clip(Array(8).fill(30)), ['neural_network']);
  assert.equal(result.yolo_live.total_detections, 0);
  assert.equal(result.event_detected, false);
});

test('a person crossing the image does not trigger dumping', async () => {
  const result = await api.analyzeLiveFramesOnEdge(clip([4, 9, 14, 19, 24, 29, 34], true), ['neural_network']);
  assert.equal(result.yolo_live.total_detections, 0);
  assert.equal(result.event_detected, false);
});

test('a falling object with an initial empty background produces a box', async () => {
  const result = await api.analyzeLiveFramesOnEdge(clip([null, 0, 7, 14, 21, 28, 35, 42, 49, 56]), ['neural_network']);
  assert.ok(result.yolo_live.total_detections > 0);
  assert.equal(result.event_detected, true);
  assert.equal(result.yolo_live.detections[0].confidence, null);
  const lastPoint = result.yolo_live.detections[0].trail.at(-1);
  assert.equal(result.yolo_live.source_timestamp_ms, 1000 + lastPoint.t * 60);
  const geometry = motionTrailGeometry(result.yolo_live.detections[0].trail, 640, 480);
  assert.ok(geometry.points.length >= 3);
  assert.deepEqual(geometry.arrow[1], geometry.points.at(-1));
  assert.ok(geometry.arrow[0].y < geometry.arrow[1].y);
  assert.ok(geometry.arrow[2].y < geometry.arrow[1].y);
});

test('the fall path and arrow mirror horizontally with the camera', () => {
  const trail = [{ cx: 0.2, cy: 0.1 }, { cx: 0.3, cy: 0.6 }];
  const normal = motionTrailGeometry(trail, 640, 480);
  const mirrored = motionTrailGeometry(trail, 640, 480, true);
  normal.points.forEach((point, i) => {
    assert.equal(mirrored.points[i].x, 640 - point.x);
    assert.equal(mirrored.points[i].y, point.y);
  });
  normal.arrow.forEach(point => assert.ok(mirrored.arrow.some(other => Math.abs(other.x - (640 - point.x)) < 0.001 && Math.abs(other.y - point.y) < 0.001)));
});

test('a missing or stationary trajectory never invents a direction arrow', () => {
  assert.equal(motionTrailGeometry(undefined, 640, 480).arrow.length, 0);
  assert.equal(motionTrailGeometry([{ cx: 0.2, cy: 0.3 }, { cx: 0.2, cy: 0.3 }], 640, 480).arrow.length, 0);
});

test('only an actual YOLO confidence value produces a percentage label', () => {
  assert.equal(yoloDetectionLabel({ class_name: 'littering', confidence: 0.864 }), 'YOLO · littering 86%');
  assert.equal(yoloDetectionLabel({ class_name: 'littering' }), 'YOLO · littering');
  assert.equal(yoloDetectionLabel({ class_name: 'litter_object', confidence: .864 }), 'YOLO · item 86%');
  assert.equal(yoloDetectionLabel({ source: 'motion_tracker', class_name: 'release_candidate', confidence: null }), 'Release candidate');
});

test('overlay mirroring follows the actual video, including a mirrored desktop camera', () => {
  const video = { readyState: 4, videoWidth: 1280, videoHeight: 720 };
  const document = { querySelectorAll: () => [video] };
  let transform = 'matrix(-1, 0, 0, 1, 0, 0)';
  const window = { getComputedStyle: () => ({ transform }) };
  const mirrored = new Function('Platform', 'document', 'window', `${source}\nreturn isWebCameraMirrored;`)
    ({ OS: 'web' }, document, window);
  assert.equal(mirrored(), true);
  transform = 'matrix3d(-1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1)';
  assert.equal(mirrored(), true);
  transform = 'none';
  assert.equal(mirrored(), false);
  transform = 'matrix(1, 0, 0, 1, 0, 0)';
  assert.equal(mirrored(), false);
});

test('a model prediction alone is not a detected release', async () => {
  const result = await api.analyzeLiveFramesOnEdge(clip(Array(8).fill(30)), ['neural_network', 'movinet_a0', 'optical_flow_svm']);
  assert.deepEqual(result.actions.map(a => a.model), ['neural_network']);
  assert.equal(result.event_detected, false);
});

test('a small light paper can fall while a separate person moves', async () => {
  const frames = Array.from({ length: 14 }, (_, i) => {
    const frame = clip([null])[0];
    const paint = (left, top, width, height, color) => {
      for (let y=top; y<Math.min(112,top+height); y++) for (let x=left; x<Math.min(112,left+width); x++) {
        frame.rgb.set(color,(y*112+x)*3);
        frame.gray[y*112+x]=(color[0]*.299+color[1]*.587+color[2]*.114)*255;
      }
    };
    paint(2+i*2,20,25,70,[.65,.4,.28]);
    if(i) paint(85,4+i*5,6,6,[.98,.98,.98]);
    frame.ts=1000+i*60; return frame;
  });
  const result=await api.analyzeLiveFramesOnEdge(frames,['neural_network']);
  assert.equal(result.event_detected,true);
  assert.ok(result.yolo_live.detections[0].norm_box[0]>.6);
  assert.ok(result.yolo_live.detections[0].trail.length>=4);
});
