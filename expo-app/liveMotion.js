// Find separate moving regions; only a downward path is a possible fall.
const SIZE = 112;
const STEP = 2;
const WIDTH = SIZE / STEP;

function isSkin(r, g, b) {
  return r > 0.30 && r > g && g > b && r - g > 0.08 && r - b > 0.12;
}

function movingRegions(previous, current, t) {
  const mask = new Uint8Array(WIDTH * WIDTH);
  const hands = [];
  let changed = 0;
  for (let y = 0; y < WIDTH; y++) {
    for (let x = 0; x < WIDTH; x++) {
      const p = (y * STEP * SIZE + x * STEP) * 3;
      const difference = (Math.abs(current.rgb[p] - previous.rgb[p]) +
        Math.abs(current.rgb[p + 1] - previous.rgb[p + 1]) +
        Math.abs(current.rgb[p + 2] - previous.rgb[p + 2])) / 3;
      if (difference < 0.10) continue;
      changed++;
      if (isSkin(current.rgb[p], current.rgb[p + 1], current.rgb[p + 2]) ||
          isSkin(previous.rgb[p], previous.rgb[p + 1], previous.rgb[p + 2])) {
        hands.push([x * STEP, y * STEP]);
      } else {
        mask[y * WIDTH + x] = 1;
      }
    }
  }
  if (changed > mask.length * 0.45) return [];
  const regions = [];
  for (let start = 0; start < mask.length; start++) {
    if (!mask[start]) continue;
    const queue = [start];
    mask[start] = 0;
    let minX = WIDTH, minY = WIDTH, maxX = 0, maxY = 0, sumX = 0, sumY = 0;
    for (let i = 0; i < queue.length; i++) {
      const x = queue[i] % WIDTH, y = Math.floor(queue[i] / WIDTH);
      minX = Math.min(minX, x); maxX = Math.max(maxX, x + 1);
      minY = Math.min(minY, y); maxY = Math.max(maxY, y + 1);
      sumX += x; sumY += y;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx, ny = y + dy, index = ny * WIDTH + nx;
          if (nx < 0 || ny < 0 || nx >= WIDTH || ny >= WIDTH || !mask[index]) continue;
          mask[index] = 0;
          queue.push(index);
        }
      }
    }
    if (queue.length < 4 || queue.length > mask.length * 0.18 ||
        maxX - minX > WIDTH * 0.5 || maxY - minY > WIDTH * 0.6) continue;
    regions.push({
      t, ts: current.ts, cx: sumX / queue.length / WIDTH, cy: sumY / queue.length / WIDTH,
      bounds: [minX / WIDTH, minY / WIDTH, maxX / WIDTH, maxY / WIDTH],
      nearHand: hands.some(([x, y]) => x >= minX * STEP - 5 && x <= maxX * STEP + 5 &&
        y >= minY * STEP - 5 && y <= maxY * STEP + 5),
    });
  }
  return regions;
}

function boxFor(point, falling = false, trail = []) {
  return {
    class_id: 0, class_name: falling ? 'possible_fall' : 'motion', confidence: null,
    norm_box: point.bounds, reason: falling ? 'downward_motion' : 'moving_region',
    trail: trail.map(({ cx, cy, t }) => ({ cx, cy, t })),
  };
}

function detectDropMotion(frames) {
  const result = { boxes: [], motionBoxes: [], peakFrameIdx: Math.max(0, frames.length - 1) };
  if (frames.length < 2 || frames.some(frame => !frame?.rgb || frame.rgb.length !== SIZE * SIZE * 3 ||
      !Number.isFinite(frame.ts))) return result;
  let tracks = [], motionFrame = 0;
  for (let i = 1; i < frames.length; i++) {
    const gap = frames[i].ts - frames[i - 1].ts;
    const regions = gap > 0 && gap <= 350 ? movingRegions(frames[i - 1], frames[i], i) : [];
    tracks = regions.map(point => {
      const matches = tracks.filter(track => {
        const last = track[track.length - 1];
        return point.cy - last.cy >= 0.01 && Math.abs(point.cx - last.cx) <= 0.20;
      });
      matches.sort((a, b) => Math.hypot(point.cx - a.at(-1).cx, point.cy - a.at(-1).cy) -
        Math.hypot(point.cx - b.at(-1).cx, point.cy - b.at(-1).cy));
      return [...(matches[0] || []), point];
    });
    for (const track of tracks) {
      const point = track[track.length - 1];
      if (track.length < 3 || point.nearHand || point.cy - track[0].cy < 0.08) continue;
      result.boxes = [boxFor(point, true, track)];
      result.peakFrameIdx = i;
    }
    // Show movement immediately, including before a fall has been established.
    if (regions.length) {
      result.motionBoxes = regions.map(point => boxFor(point));
      motionFrame = i;
    } else if (i - motionFrame > 4) result.motionBoxes = [];
  }
  return result;
}

module.exports = { detectDropMotion };
