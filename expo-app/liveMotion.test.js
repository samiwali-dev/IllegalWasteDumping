const assert = require('node:assert/strict');
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const { createHash } = require('node:crypto');
const test = require('node:test');
const source = readFileSync(join(__dirname, 'liveEdgeAi.js'), 'utf8')
  .replace(/^import .*;\s*$/gm, '').replace(/^export /gm, '');
const weights = JSON.parse(readFileSync(join(__dirname, 'nnWeights.json'), 'utf8'));
const api = new Function('Platform', 'nnWeights', `${source}\nreturn { analyzeLiveFramesOnEdge, detectDumpedObjectsOnly };`)
  ({ OS: 'test' }, weights);

function clip(positions, largePerson = false) {
  return positions.map((position, i) => {
    const rgb = new Float32Array(112 * 112 * 3).fill(0.8);
    const gray = new Float32Array(112 * 112).fill(204);
    const left = largePerson ? position : 49, top = largePerson ? 20 : position;
    const width = largePerson ? 40 : 10, height = largePerson ? 70 : 10;
    for (let y = top; y < Math.min(112, top + height); y++) {
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

test('the original detection calculations are restored unchanged', () => {
  const normalized = api.detectDumpedObjectsOnly.toString()
    .replace(/\/\/[^\n]*|\/\*[\s\S]*?\*\//g, '').replace(/\s+/g, '');
  const checksum = createHash('sha256').update(normalized).digest('hex');
  assert.equal(checksum, 'cbf3d4c595c76fdf3a4534286b03577d541bf5bd9eb77eff71c1265f6048934a');
});

test('a stationary black object does not trigger dumping', async () => {
  const result = await api.analyzeLiveFramesOnEdge(clip(Array(8).fill(30)), ['neural_network']);
  assert.equal(result.yolo_live.total_detections, 0);
  assert.equal(result.actions[0].prediction, 0);
});

test('a person crossing the image does not trigger dumping', async () => {
  const result = await api.analyzeLiveFramesOnEdge(clip([4, 9, 14, 19, 24, 29, 34], true), ['neural_network']);
  assert.equal(result.yolo_live.total_detections, 0);
  assert.equal(result.actions[0].prediction, 0);
});

test('the original downward-object detection still produces a box', async () => {
  const result = await api.analyzeLiveFramesOnEdge(clip([0, 14, 28, 42, 56]), ['neural_network']);
  assert.ok(result.yolo_live.total_detections > 0);
  assert.equal(result.actions[0].prediction, 1);
});
