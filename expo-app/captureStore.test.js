const assert = require('node:assert/strict');
const test = require('node:test');
const { frameUri, captureFromResult, retainCaptures, captureDetectionsForFall, motionCandidateForFrames } = require('./captureStore');

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
  const result = { mode: 'edge_browser', yolo_live: { total_detections: 1 }, actions: [{ prediction: 1 }] };
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

test('classification or a static YOLO object cannot create a fall capture', () => {
  const classification = { actions: [{ prediction: 1 }], yolo_live: { total_detections: 0 } };
  assert.deepEqual(captureDetectionsForFall(classification, null, 10000), []);
  const object = { yolo_live: { detections: [{ norm_box: [.4, .4, .6, .6], confidence: .9 }] } };
  assert.deepEqual(captureDetectionsForFall(object, null, 10000), []);
});

test('a model capture requires a fresh falling track at the same object location', () => {
  const detection = { norm_box: [.4, .4, .6, .6], class_name: 'litter_object', confidence: .84 };
  const result = { yolo_live: { detections: [detection] } };
  const track = { reason: 'falling_object', norm_box: [.3, .3, .7, .7], trail: [{ cx: .5, cy: .1 }, { cx: .5, cy: .3 }, { cx: .5, cy: .5 }] };
  const matched = captureDetectionsForFall(result, { at: 9500, detections: [track] }, 10000);
  assert.equal(matched.length, 1);
  assert.equal(matched[0].confidence, .84);
  assert.equal(matched[0].trail, track.trail);
  assert.deepEqual(captureDetectionsForFall(result, { at: 5000, detections: [track] }, 10000), []);
  assert.deepEqual(captureDetectionsForFall(result, { at: 9500, detections: [{ ...track, norm_box: [0, 0, .1, .1] }] }, 10000), []);
  assert.deepEqual(captureDetectionsForFall(result, { at: 9500, detections: [{ ...track, trail: [] }] }, 10000), []);
});

test('a backend release uses its own physical trajectory without inventing object confidence', () => {
  const candidate = { reason: 'released_object', source: 'motion_tracker', confidence: null,
    source_actor: { class_name: 'car', confidence: .91 },
    trail: [{ cx: .4, cy: .3 }, { cx: .45, cy: .4 }, { cx: .5, cy: .4 }] };
  const result = { yolo_live: { task: 'temporal_release_candidates', detections: [candidate] } };
  assert.deepEqual(captureDetectionsForFall(result, null), [candidate]);
  assert.equal(captureDetectionsForFall(result, null)[0].confidence, null);
  result.yolo_live.detections = [{ ...candidate, source_actor: null }];
  assert.deepEqual(captureDetectionsForFall(result, null), []);
});

test('a high classifier score without a tracked object cannot create evidence', () => {
  assert.equal(captureFromResult({ actions: [{ prediction: 1, score: .998 }], yolo_live: { total_detections: 0 } }, ['frame']), null);
});

test('a backend verified four-point fall preserves its annotated source photo without an actor', () => {
  const candidate = { reason: 'released_object', source: 'motion_tracker', confidence: null,
    motion_verified: true, object_confirmed: false,
    trail: [{ cx: .7, cy: .1 }, { cx: .7, cy: .2 }, { cx: .7, cy: .3 }, { cx: .7, cy: .4 }] };
  const result = { mode: 'python_api', yolo_live: { task: 'temporal_release_candidates', total_detections: 1,
    detections: [candidate], source_timestamp_ms: 1200, annotated_data_uri: 'data:image/jpeg;base64,backend-box-and-arrow' } };
  assert.deepEqual(captureDetectionsForFall(result, null), [candidate]);
  const capture = captureFromResult(result, [{ ts: 1200, uri: 'source' }, { ts: 1500, uri: 'later' }]);
  assert.equal(capture.uri, result.yolo_live.annotated_data_uri);
  assert.equal(capture.sourceTimestamp, 1200);
  const retained = retainCaptures([], capture);
  assert.equal(retainCaptures(retained, captureFromResult({ yolo_live: { total_detections: 0 } }, ['later'])), retained);
  result.yolo_live.detections = [{ ...candidate, trail: candidate.trail.slice(0, 3) }];
  assert.deepEqual(captureDetectionsForFall(result, null), []);
});

test('an unannotated event selects its timestamp source and never a later frame', () => {
  const result = { yolo_live: { total_detections: 1, source_timestamp_ms: 1200 } };
  assert.equal(captureFromResult(result, [{ ts: 1200, uri: 'event' }, { ts: 1500, uri: 'later' }]).uri, 'event');
  assert.equal(captureFromResult(result, [{ ts: 1500, uri: 'later' }]), null);
});

test('only a recent physical candidate whose exact frame is buffered is submitted', () => {
  const detection = { source: 'motion_tracker', reason: 'falling_object', confidence: null,
    trail: [{ cx: .5, cy: .1 }, { cx: .5, cy: .2 }, { cx: .5, cy: .3 }, { cx: .5, cy: .4 }] };
  const motion = { at: 1500, timestamp_ms: 1200, detections: [detection] };
  assert.deepEqual(motionCandidateForFrames(motion, [1200, 1500], 2000), { timestamp_ms: 1200, detections: [detection] });
  assert.equal(motionCandidateForFrames(motion, [1500], 2000), null);
  assert.equal(motionCandidateForFrames(motion, [1200], 3000), null);
  assert.equal(motionCandidateForFrames({ ...motion, detections: [{ ...detection, trail: [] }] }, [1200], 2000), null);
});
