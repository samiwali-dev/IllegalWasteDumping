const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const test = require('node:test');
const { detectDropMotion } = require('./liveMotion');

function frame(y, ts, { hand = false, brightness = 0.8, color = [0, 0, 0] } = {}) {
  const rgb = new Float32Array(112 * 112 * 3).fill(brightness);
  function rectangle(x, top, width, height, pixels) {
    for (let row = Math.max(0, top); row < Math.min(112, top + height); row++) {
      for (let col = x; col < x + width; col++) rgb.set(pixels, (row * 112 + col) * 3);
    }
  }
  if (y !== null) rectangle(49, y, 10, 10, color);
  if (hand && y !== null) rectangle(36, y, 12, 10, [0.8, 0.5, 0.3]);
  return { isDecodedFrame: true, rgb, gray: new Float32Array(112 * 112), ts, uri: null };
}

function clip(positions, options = {}) {
  return positions.map((y, index) => frame(y, 1000 + index * 60, options));
}

// Supply the platform imports to the actual browser entry point so its output
// contract is covered without accessing a real camera or importing native UI.
const edgeSource = readFileSync(join(__dirname, 'liveEdgeAi.js'), 'utf8')
  .replace(/^import .*;\s*$/gm, '').replace(/^export /gm, '');
const analyze = new Function('Platform', 'nnWeights', 'detectDropMotion',
  `${edgeSource}\nreturn analyzeLiveFramesOnEdge;`)({ OS: 'test' }, {}, detectDropMotion);

const fall = [6, 7, 10, 15, 22, 31, 42, 55, 70, 87];

test('a stationary black object does not trigger an event', () => {
  assert.deepEqual(detectDropMotion(clip(Array(12).fill(42))).boxes, []);
});

test('a newly visible stationary object is not dumping, including a locked empty baseline', async () => {
  const result = await analyze(clip([null, null, 42, 42, 42, 42, 42, 42, 42]),
    ['neural_network', 'movinet_a0', 'optical_flow_svm', 'yolov8n'], frame(null, 0));
  assert.equal(result.event_detected, false);
  assert.equal(result.yolo_live.total_detections, 0);
  assert.equal(result.actions[0].prediction, 0);
  assert.equal(result.actions[0].score, null);
});

test('an object moved while held next to a visible hand is rejected', () => {
  for (const color of [[0, 0, 0], [0.1, 0.2, 0.9], [1, 1, 1]]) {
    for (const speed of [3, 5, 7]) {
      for (const offset of [0, 3, 6, 9]) {
        const positions = Array.from({ length: 12 }, (_, i) => 5 + offset + i * speed);
        assert.equal(detectDropMotion(clip(positions, { color, brightness: 0.5, hand: true })).boxes.length, 0,
          `constant movement: speed=${speed}, offset=${offset}, color=${color}`);
      }
    }
  }
});

test('even accelerating movement while still next to the hand is rejected', () => {
  assert.deepEqual(detectDropMotion(clip(fall, { hand: true })).boxes, []);
});

test('exposure changes and a short two-step movement do not trigger an event', () => {
  const lighting = Array.from({ length: 12 }, (_, i) => frame(null, i * 60, { brightness: i % 2 ? 0.2 : 0.9 }));
  assert.deepEqual(detectDropMotion(lighting).boxes, []);
  assert.deepEqual(detectDropMotion(clip([10, 24, 38])).boxes, []);
});

test('an accelerating falling object remains detectable and is labeled as a possibility', async () => {
  for (const color of [[0, 0, 0], [0.1, 0.2, 0.9], [1, 1, 1]]) {
    assert.equal(detectDropMotion(clip(fall, { color, brightness: 0.5 })).boxes.length, 1);
  }
  const result = await analyze(clip(fall), ['yolov8n']);
  assert.equal(result.event_detected, true);
  assert.equal(result.actions[0].verdict, 'POSSIBLE_FALL');
  assert.equal(result.yolo_live.strongest_detection.class_name, 'possible_fall');
  assert.equal(result.yolo_live.detector, 'temporal_motion');
  assert.ok(result.yolo_live.strongest_detection.trail.length >= 3);
});

test('a fall separated from a stationary hand can still be detected', () => {
  const frames = clip(fall);
  for (const image of frames) {
    for (let y = 6; y < 16; y++) {
      for (let x = 36; x < 48; x++) image.rgb.set([0.8, 0.5, 0.3], (y * 112 + x) * 3);
    }
  }
  assert.equal(detectDropMotion(frames).boxes.length, 1);
});

test('invalid capture times and large gaps cannot create a falling trajectory', () => {
  assert.deepEqual(detectDropMotion(clip(fall).map(f => ({ ...f, ts: 0 }))).boxes, []);
  assert.deepEqual(detectDropMotion(clip(fall).map((f, i) => ({ ...f, ts: i * 500 }))).boxes, []);
});

test('browser output never impersonates the trained Python models or invents their scores', async () => {
  for (const models of [['neural_network'], ['movinet_a0'], ['optical_flow_svm'], ['yolov8n']]) {
    const result = await analyze(clip(fall), models);
    assert.deepEqual(result.actions.map(action => action.model), ['edge_motion']);
    assert.equal(result.actions[0].score_kind, 'فحص حركة للأسفل عبر عدة إطارات');
  }
});

test('native fallback rejects encoded bytes instead of treating base64 characters as pixels', async () => {
  await assert.rejects(analyze(['data:image/jpeg;base64,not-a-decoded-frame'], ['neural_network']),
    /خادم Python/);
});

// A quick drop may not provide the six accelerating observations previously required.
test('a quick drop is detected from four captured frames without an acceleration fit', () => {
  const result = detectDropMotion(clip([10, 24, 38, 52]));
  assert.equal(result.boxes.length, 1);
  assert.equal(result.boxes[0].reason, 'downward_motion');
});

test('horizontal or upward movement alone does not trigger a drop', () => {
  assert.equal(detectDropMotion(clip([80, 65, 50, 35, 20])).boxes.length, 0);
  const horizontal = [5, 17, 29, 41, 53].map((x, i) => {
    const image = frame(null, 1000 + i * 60);
    for (let y = 40; y < 50; y++) {
      for (let col = x; col < x + 10; col++) image.rgb.fill(0, (y * 112 + col) * 3, (y * 112 + col) * 3 + 3);
    }
    return image;
  });
  assert.equal(detectDropMotion(horizontal).boxes.length, 0);
});


test('a moving item gets a box immediately without claiming a fall', async () => {
  const result = await analyze(clip([20, 32]), ['yolov8n'], { annotate: false });
  assert.equal(result.event_detected, false);
  assert.ok(result.yolo_live.detections.length > 0);
  assert.ok(result.yolo_live.detections.every(box => box.class_name === 'motion'));
  assert.equal(result.actions[0].prediction, 0);
});

test('a brief movement between analysis ticks still produces a visible box', async () => {
  const result = await analyze(clip([20, 32, 32, 32, 32]), [], { annotate: false });
  assert.equal(result.event_detected, false);
  assert.ok(result.yolo_live.detections.length > 0);
});

test('unrelated movement elsewhere does not hide a falling item', () => {
  const frames = clip([10, 24, 38, 52]);
  frames.forEach((image, i) => {
    for (let y = 90; y < 100; y++) {
      for (let x = 2 + i * 2; x < 22 + i * 2; x++) image.rgb.fill(0, (y * 112 + x) * 3, (y * 112 + x) * 3 + 3);
    }
  });
  const result = detectDropMotion(frames);
  assert.ok(result.motionBoxes.length >= 2);
  assert.equal(result.boxes.length, 1);
});

test('the web capture preserves a widescreen camera aspect ratio', () => {
  const video = { readyState: 4, videoWidth: 1280, videoHeight: 720 };
  const document = {
    querySelectorAll: () => [video],
    createElement: () => ({
      getContext: () => ({ drawImage() {}, getImageData: () => ({ data: new Uint8ClampedArray(112 * 112 * 4) }) }),
      toDataURL: () => 'data:image/jpeg;base64,snapshot',
    }),
  };
  const camera = new Function('Platform', 'nnWeights', 'detectDropMotion', 'document',
    `${edgeSource}\nreturn { grabWebVideoFrameFast, getWebCameraAspectRatio };`)({ OS: 'web' }, {}, detectDropMotion, document);
  assert.equal(camera.getWebCameraAspectRatio(), 16 / 9);
  const captured = camera.grabWebVideoFrameFast(true);
  assert.equal(captured.width, 320);
  assert.equal(captured.height, 180);
});
