// Simple browser motion check: three moving positions descending in sequence.
// It can flag a possible fall, but cannot prove that an object is waste.
const SIZE = 112;
const PIXEL_CHANGE = 0.10;
const MIN_DROP = 0.08;

function isSkin(r, g, b) {
  return r > 0.30 && r > g && g > b && r - g > 0.08 && r - b > 0.12;
}

function movingObject(previous, current, t) {
  let changed = 0, weight = 0, sumX = 0, sumY = 0;
  let minX = SIZE, minY = SIZE, maxX = 0, maxY = 0;
  const skin = [];
  for (let y = 0; y < SIZE; y += 2) {
    for (let x = 0; x < SIZE; x += 2) {
      const p = (y * SIZE + x) * 3;
      const currentSkin = isSkin(current.rgb[p], current.rgb[p + 1], current.rgb[p + 2]);
      if (currentSkin) skin.push([x, y]);
      const difference = (Math.abs(current.rgb[p] - previous.rgb[p]) +
        Math.abs(current.rgb[p + 1] - previous.rgb[p + 1]) +
        Math.abs(current.rgb[p + 2] - previous.rgb[p + 2])) / 3;
      if (difference < PIXEL_CHANGE) continue;
      changed++;
      if (currentSkin || isSkin(previous.rgb[p], previous.rgb[p + 1], previous.rgb[p + 2])) continue;
      weight += difference;
      sumX += x * difference;
      sumY += y * difference;
      minX = Math.min(minX, x); maxX = Math.max(maxX, x + 2);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y + 2);
    }
  }
  // Ignore camera/exposure changes affecting much of the image.
  if (!weight || changed > (SIZE * SIZE / 4) * 0.18 ||
      maxX - minX > SIZE * 0.5 || maxY - minY > SIZE * 0.6) return null;
  return {
    t, ts: current.ts, cx: sumX / weight / SIZE, cy: sumY / weight / SIZE,
    bounds: [minX / SIZE, minY / SIZE, maxX / SIZE, maxY / SIZE],
    nearHand: skin.some(([x, y]) => x >= minX - 5 && x <= maxX + 5 && y >= minY - 5 && y <= maxY + 5),
  };
}

function detectDropMotion(frames) {
  const empty = { boxes: [], peakFrameIdx: Math.max(0, frames.length - 1) };
  if (frames.length < 4 || frames.some(frame => !frame?.rgb || frame.rgb.length !== SIZE * SIZE * 3 ||
      !Number.isFinite(frame.ts))) return empty;
  let trail = [];
  for (let i = 1; i < frames.length; i++) {
    const gap = frames[i].ts - frames[i - 1].ts;
    const point = gap > 0 && gap <= 350 ? movingObject(frames[i - 1], frames[i], i) : null;
    if (!point) { trail = []; continue; }
    const previous = trail[trail.length - 1];
    if (previous && (point.cy - previous.cy < 0.01 || Math.abs(point.cx - previous.cx) > 0.20)) trail = [];
    trail.push(point);
    if (trail.length < 3 || point.nearHand || point.cy - trail[0].cy < MIN_DROP) continue;
    return {
      peakFrameIdx: point.t,
      boxes: [{
        class_id: 0, class_name: 'possible_fall', confidence: null,
        norm_box: point.bounds,
        trail: trail.map(({ cx, cy, t }) => ({ cx, cy, t })),
        reason: 'downward_motion',
      }],
    };
  }
  return empty;
}

module.exports = { detectDropMotion };
