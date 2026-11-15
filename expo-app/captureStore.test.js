const assert = require('node:assert/strict');
const test = require('node:test');
const { frameUri, captureFromResult, retainCaptures, shouldCaptureModelResult } = require('./captureStore');

const alert = {
  mode: 'edge_browser', actions: [{ prediction: 1 }],
  yolo_live: { total_detections: 1, annotated_data_uri: 'data:image/jpeg;base64,event-frame' },
};

test('an alert saves its annotated source image and timestamp', () => {
  const capture = captureFromResult(alert, [{ uri: 'later-frame' }], 1500);
  assert.equal(capture.uri, alert.yolo_live.annotated_data_uri);
  assert.equal(capture.capturedAt, 1500);
  assert.equal(capture.kind, 'Motion alert');
  assert.equal(capture.regions, 1);
});

test('a clear result cannot erase the previous alert capture', () => {
  const captures = retainCaptures([], captureFromResult(alert, [], 1500));
  const clear = { actions: [{ prediction: 0 }], yolo_live: { total_detections: 0 } };
  assert.equal(retainCaptures(captures, captureFromResult(clear, ['clear-frame'])), captures);
  assert.equal(captures[0].uri, alert.yolo_live.annotated_data_uri);
});

test('an alert without annotation uses the latest available camera image', () => {
  const result = { mode: 'edge_browser', actions: [{ prediction: 1 }] };
  assert.equal(captureFromResult(result, ['first-frame', { uri: 'last-frame' }, { uri: null }]).uri, 'last-frame');
  assert.equal(captureFromResult(result, []), null);
});

test('manual frames can be retained without any detector prediction', () => {
  const capture = { id: 'manual', uri: frameUri({ uri: 'manual-photo' }), kind: 'Manual capture' };
  assert.equal(retainCaptures([], capture)[0].uri, 'manual-photo');
  assert.equal(frameUri('native-photo'), 'native-photo');
});

test('history keeps the latest six unique captures', () => {
  let captures = [];
  for (let id = 0; id < 8; id++) captures = retainCaptures(captures, { id: String(id), uri: `photo-${id}` });
  assert.deepEqual(captures.map(item => item.id), ['7', '6', '5', '4', '3', '2']);
  captures = retainCaptures(captures, { id: '5', uri: 'updated-photo' });
  assert.equal(captures[0].uri, 'updated-photo');
  assert.equal(captures.length, 6);
});

test('classification alone cannot replace a fall photo with an unmarked frame', () => {
  const classification = { actions: [{ prediction: 1 }], yolo_live: { total_detections: 0 } };
  assert.equal(shouldCaptureModelResult(classification, 0, 10000), false);
  assert.equal(shouldCaptureModelResult(alert, 9500, 10000), false);
  assert.equal(shouldCaptureModelResult(alert, 5000, 10000), true);
});
