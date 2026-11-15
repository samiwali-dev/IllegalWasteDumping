// Conservative motion screening for the browser. This does not identify waste
// or prove a release from a hand; a positive result is a possible falling object.
const SIZE = 112;
const GRID = 8;

function isSkin(r, g, b) {
  r *= 255; g *= 255; b *= 255;
  if (!(r > 85 && g > 35 && b > 18 && r > g && r > b &&
        Math.abs(r - g) > 12 && Math.max(r, g, b) - Math.min(r, g, b) > 14)) return false;
  const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
  const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
  return cr >= 130 && cr <= 178 && cb >= 75 && cb <= 130;
}

function candidatesForFrame(previous, current, t) {
  const cellSize = SIZE / GRID;
  const differences = new Float32Array(GRID * GRID);
  const momentsX = new Float32Array(GRID * GRID);
  const momentsY = new Float32Array(GRID * GRID);
  const skin = new Float32Array(GRID * GRID);
  const skinPoints = [];
  let movingCells = 0;
  for (let row = 0; row < GRID; row++) {
    for (let col = 0; col < GRID; col++) {
      let difference = 0, momentX = 0, momentY = 0, skinCount = 0, count = 0;
      for (let y = row * cellSize; y < (row + 1) * cellSize; y += 2) {
        for (let x = col * cellSize; x < (col + 1) * cellSize; x += 2) {
          const p = (y * SIZE + x) * 3;
          const pixelDifference = (Math.abs(current.rgb[p] - previous.rgb[p]) +
            Math.abs(current.rgb[p + 1] - previous.rgb[p + 1]) +
            Math.abs(current.rgb[p + 2] - previous.rgb[p + 2])) / 3;
          difference += pixelDifference;
          momentX += ((x + 0.5) / SIZE) * pixelDifference;
          momentY += ((y + 0.5) / SIZE) * pixelDifference;
          if (isSkin(current.rgb[p], current.rgb[p + 1], current.rgb[p + 2])) {
            skinCount++;
            skinPoints.push([x / SIZE, y / SIZE]);
          }
          count++;
        }
      }
      const index = row * GRID + col;
      differences[index] = difference / count;
      momentsX[index] = momentX / count;
      momentsY[index] = momentY / count;
      skin[index] = skinCount / count;
      if (differences[index] > 0.05) movingCells++;
    }
  }
  // Large changes include camera motion and exposure changes. Do not bridge a
  // track through these frames, even if another compact region also moved.
  if (movingCells > 14) return { unstable: true, candidates: [] };

  const visited = new Set();
  const candidates = [];
  for (let start = 0; start < differences.length; start++) {
    if (visited.has(start) || differences[start] < 0.052 || skin[start] >= 0.34) continue;
    const queue = [start];
    visited.add(start);
    let minR = GRID, maxR = 0, minC = GRID, maxC = 0;
    let sumX = 0, sumY = 0, energy = 0;
    for (let cursor = 0; cursor < queue.length; cursor++) {
      const index = queue[cursor], row = Math.floor(index / GRID), col = index % GRID;
      const weight = differences[index];
      minR = Math.min(minR, row); maxR = Math.max(maxR, row);
      minC = Math.min(minC, col); maxC = Math.max(maxC, col);
      // Use pixel centroids: snapping to grid-cell centers can make constant
      // motion look like acceleration as an object crosses a cell boundary.
      sumX += momentsX[index];
      sumY += momentsY[index];
      energy += weight;
      for (let dr = -1; dr <= 1; dr++) {
        for (let dc = -1; dc <= 1; dc++) {
          const nr = row + dr, nc = col + dc, next = nr * GRID + nc;
          if (nr < 0 || nr >= GRID || nc < 0 || nc >= GRID || visited.has(next) ||
              differences[next] < 0.045 || skin[next] >= 0.34) continue;
          visited.add(next);
          queue.push(next);
        }
      }
    }
    if (queue.length > 6 || maxR - minR > 2 || maxC - minC > 2) continue;
    const bounds = [minC / GRID, minR / GRID, (maxC + 1) / GRID, (maxR + 1) / GRID];
    const nearSkin = skinPoints.some(([x, y]) => x >= bounds[0] - 0.055 &&
      x <= bounds[2] + 0.055 && y >= bounds[1] - 0.055 && y <= bounds[3] + 0.055);
    candidates.push({ t, ts: current.ts, cx: sumX / energy, cy: sumY / energy,
      bounds, nearSkin, cells: queue.length });
  }
  return { unstable: false, candidates };
}

function solve(matrix, values) {
  const rows = matrix.map((row, i) => [...row, values[i]]);
  for (let col = 0; col < rows.length; col++) {
    let pivot = col;
    for (let row = col + 1; row < rows.length; row++) {
      if (Math.abs(rows[row][col]) > Math.abs(rows[pivot][col])) pivot = row;
    }
    if (Math.abs(rows[pivot][col]) < 1e-9) return null;
    [rows[col], rows[pivot]] = [rows[pivot], rows[col]];
    const divisor = rows[col][col];
    for (let j = col; j <= rows.length; j++) rows[col][j] /= divisor;
    for (let row = 0; row < rows.length; row++) {
      if (row === col) continue;
      const factor = rows[row][col];
      for (let j = col; j <= rows.length; j++) rows[row][j] -= factor * rows[col][j];
    }
  }
  return rows.map(row => row[rows.length]);
}

function assessTrajectory(chain) {
  if (chain.length < 6) return null;
  const first = chain[0], last = chain[chain.length - 1];
  const duration = (last.ts - first.ts) / 1000;
  const distance = last.cy - first.cy;
  if (duration < 0.18 || duration > 0.85 || distance < 0.18 ||
      distance < Math.abs(last.cx - first.cx) * 1.5) return null;
  // Holding an object near a visible hand is evidence against a release.
  // Skin color is only a veto here, never positive evidence of littering.
  if (chain.slice(-3).some(point => point.nearSkin)) return null;

  // Fit y = a + b*u + c*u² against actual capture times. A straight downward
  // hand movement must not pass simply because its position changed.
  const powers = Array(5).fill(0), targets = Array(3).fill(0);
  const samples = chain.map(point => ({ u: (point.ts - first.ts) / 1000 / duration, y: point.cy }));
  for (const { u, y } of samples) {
    for (let k = 0; k < 5; k++) powers[k] += u ** k;
    for (let k = 0; k < 3; k++) targets[k] += y * u ** k;
  }
  const curve = solve([[powers[0], powers[1], powers[2]],
    [powers[1], powers[2], powers[3]], [powers[2], powers[3], powers[4]]], targets);
  const line = solve([[powers[0], powers[1]], [powers[1], powers[2]]], targets.slice(0, 2));
  if (!curve || !line) return null;
  const [a, b, c] = curve;
  const initialSpeed = b / duration, finalSpeed = (b + 2 * c) / duration;
  if (c < 0.06 || c < distance * 0.25 || initialSpeed < -0.05 || finalSpeed < 0.65 ||
      finalSpeed < Math.max(0.1, initialSpeed) * 1.5) return null;
  let curveError = 0, lineError = 0, variance = 0;
  const mean = samples.reduce((sum, sample) => sum + sample.y, 0) / samples.length;
  for (const { u, y } of samples) {
    curveError += (y - a - b * u - c * u * u) ** 2;
    lineError += (y - line[0] - line[1] * u) ** 2;
    variance += (y - mean) ** 2;
  }
  if (curveError > lineError * 0.4 || curveError > variance * 0.08) return null;
  return { distance, duration, fit: 1 - curveError / Math.max(variance, 1e-9) };
}

function detectDropMotion(frames) {
  if (frames.length < 6) return { boxes: [], peakFrameIdx: 0 };
  for (let i = 0; i < frames.length; i++) {
    if (!frames[i]?.rgb || frames[i].rgb.length !== SIZE * SIZE * 3 ||
        !Number.isFinite(frames[i].ts) || (i > 0 && frames[i].ts <= frames[i - 1].ts)) {
      return { boxes: [], peakFrameIdx: 0 };
    }
  }
  const observations = frames.slice(1).map((frame, i) => candidatesForFrame(frames[i], frame, i + 1));
  let best = null;
  for (let start = 0; start < observations.length; start++) {
    for (const candidate of observations[start].candidates) {
      const chain = [candidate];
      for (let t = start + 1; t < observations.length; t++) {
        if (observations[t].unstable) break;
        const previous = chain[chain.length - 1];
        if (frames[t + 1].ts - previous.ts > 180) break;
        const matches = observations[t].candidates.filter(point => {
          const dy = point.cy - previous.cy, dx = Math.abs(point.cx - previous.cx);
          return dy >= -0.01 && dy <= 0.27 && dx <= 0.12 &&
            Math.max(point.cells, previous.cells) / Math.min(point.cells, previous.cells) <= 3;
        });
        matches.sort((a, b) => Math.hypot(a.cx - previous.cx, a.cy - previous.cy) -
          Math.hypot(b.cx - previous.cx, b.cy - previous.cy));
        if (matches.length) chain.push(matches[0]);
      }
      // Evaluate the complete observed track, not a briefly accelerating prefix
      // which later frames reveal to be ordinary hand movement.
      const evidence = assessTrajectory(chain);
      if (evidence && (!best || chain.length > best.chain.length)) best = { chain, evidence };
    }
  }
  if (!best) return { boxes: [], peakFrameIdx: frames.length - 1 };
  const target = best.chain[best.chain.length - 1];
  return {
    peakFrameIdx: target.t,
    boxes: [{
      class_id: 0,
      class_name: 'possible_fall',
      // This is a trajectory fit, not a probability from YOLO or another model.
      confidence: best.evidence.fit,
      score_kind: 'trajectory_fit_not_probability',
      norm_box: target.bounds,
      trail: best.chain.map(point => ({ cx: point.cx, cy: point.cy, t: point.t })),
      reason: 'accelerating_downward_motion',
    }],
  };
}

module.exports = { detectDropMotion };
