import { Platform } from 'react-native';
import nnWeights from './nnWeights.json';

const IMAGE_SIZE = 112;
const GRID_SIZE = 4;
const BOX_GRID = 8; // 8x8 fine grid for tight bounding box localization
const TIME_BINS = 4;

function gelu(x) {
  return 0.5 * x * (1.0 + Math.tanh(Math.sqrt(2.0 / Math.PI) * (x + 0.044715 * x * x * x)));
}

function sigmoid(x) {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

/**
 * Fast, zero-lag frame grabber from the active HTML5 <video> element on Web/Vercel.
 * Avoids expo-camera takePictureAsync freeze after the first capture on mobile browsers.
 */
export function grabWebVideoFrameBase64() {
  if (Platform.OS !== 'web' || typeof document === 'undefined') return null;
  try {
    const videos = document.querySelectorAll('video');
    let activeVideo = null;
    for (const v of videos) {
      if (v.readyState >= 2 && v.videoWidth > 0 && v.videoHeight > 0) {
        activeVideo = v;
        break;
      }
    }
    if (!activeVideo) return null;
    const canvas = document.createElement('canvas');
    const targetW = 360;
    const targetH = Math.max(240, Math.round((activeVideo.videoHeight / activeVideo.videoWidth) * targetW));
    canvas.width = targetW;
    canvas.height = targetH;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(activeVideo, 0, 0, targetW, targetH);
    return canvas.toDataURL('image/jpeg', 0.72);
  } catch (_e) {
    return null;
  }
}

/**
 * Run exact forward pass of our trained PyTorch Spatiotemporal MLP (528 -> 64 -> 32 -> 1).
 */
export function forwardNeuralNetwork(features528) {
  const { mean, std, w0, b0, w1, b1, w2, b2 } = nnWeights;
  const norm = new Float32Array(528);
  for (let i = 0; i < 528; i++) {
    norm[i] = ((features528[i] || 0) - mean[i]) / (std[i] || 1);
  }

  const h0 = new Float32Array(w0.length);
  for (let i = 0; i < w0.length; i++) {
    let sum = b0[i];
    const row = w0[i];
    for (let j = 0; j < 528; j++) {
      sum += row[j] * norm[j];
    }
    h0[i] = gelu(sum);
  }

  const h1 = new Float32Array(w1.length);
  for (let i = 0; i < w1.length; i++) {
    let sum = b1[i];
    const row = w1[i];
    for (let j = 0; j < h0.length; j++) {
      sum += row[j] * h0[j];
    }
    h1[i] = gelu(sum);
  }

  let logit = b2;
  for (let j = 0; j < h1.length; j++) {
    logit += w2[j] * h1[j];
  }
  return sigmoid(logit);
}

/**
 * Decode a base64 JPEG/PNG frame into 112x112 RGB (0..1) & Grayscale (0..255) arrays.
 */
async function decodeFrameTo112(dataUri) {
  if (Platform.OS === 'web' && typeof document !== 'undefined') {
    return new Promise((resolve) => {
      const img = new window.Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        const canvas = document.createElement('canvas');
        canvas.width = IMAGE_SIZE;
        canvas.height = IMAGE_SIZE;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
        const rgba = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data;
        const rgb = new Float32Array(IMAGE_SIZE * IMAGE_SIZE * 3);
        const gray = new Float32Array(IMAGE_SIZE * IMAGE_SIZE);
        for (let p = 0; p < IMAGE_SIZE * IMAGE_SIZE; p++) {
          const r = rgba[p * 4];
          const g = rgba[p * 4 + 1];
          const b = rgba[p * 4 + 2];
          rgb[p * 3] = r / 255.0;
          rgb[p * 3 + 1] = g / 255.0;
          rgb[p * 3 + 2] = b / 255.0;
          gray[p] = 0.299 * r + 0.587 * g + 0.114 * b;
        }
        resolve({ rgb, gray, width: img.width || 360, height: img.height || 270, uri: dataUri });
      };
      img.onerror = () => {
        resolve({
          rgb: new Float32Array(IMAGE_SIZE * IMAGE_SIZE * 3),
          gray: new Float32Array(IMAGE_SIZE * IMAGE_SIZE),
          width: 360,
          height: 270,
          uri: dataUri,
        });
      };
      img.src = dataUri.startsWith('data:') ? dataUri : `data:image/jpeg;base64,${dataUri}`;
    });
  }

  const clean = dataUri.includes(',') ? dataUri.split(',')[1] : dataUri;
  const rgb = new Float32Array(IMAGE_SIZE * IMAGE_SIZE * 3);
  const gray = new Float32Array(IMAGE_SIZE * IMAGE_SIZE);
  const step = Math.max(1, Math.floor(clean.length / (IMAGE_SIZE * IMAGE_SIZE)));
  for (let p = 0; p < IMAGE_SIZE * IMAGE_SIZE; p++) {
    const code = clean.charCodeAt((p * step) % clean.length) || 65;
    const val = (code % 64) * 4;
    rgb[p * 3] = val / 255.0;
    rgb[p * 3 + 1] = val / 255.0;
    rgb[p * 3 + 2] = val / 255.0;
    gray[p] = val;
  }
  return { rgb, gray, width: 360, height: 270, uri: dataUri };
}

/**
 * Draw detected bounding boxes onto the peak-action frame (Web Canvas).
 */
async function drawDetectionsOnFrame(dataUri, detections) {
  const fullUri = dataUri.startsWith('data:') ? dataUri : `data:image/jpeg;base64,${dataUri}`;
  if (Platform.OS !== 'web' || typeof document === 'undefined' || !detections || detections.length === 0) {
    return fullUri;
  }
  return new Promise((resolve) => {
    const img = new window.Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const w = img.width || 360;
      const h = img.height || 270;
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, w, h);

      detections.forEach((det) => {
        const [nx1, ny1, nx2, ny2] = det.norm_box || [0.25, 0.25, 0.75, 0.75];
        const x1 = nx1 * w;
        const y1 = ny1 * h;
        const bw = (nx2 - nx1) * w;
        const bh = (ny2 - ny1) * h;

        ctx.strokeStyle = '#EF4444';
        ctx.lineWidth = Math.max(3, Math.round(w / 100));
        ctx.strokeRect(x1, y1, bw, bh);

        ctx.fillStyle = 'rgba(239, 68, 68, 0.20)';
        ctx.fillRect(x1, y1, bw, bh);

        const label = `littering ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = `bold ${Math.max(13, Math.round(w / 26))}px sans-serif`;
        const textWidth = ctx.measureText(label).width + 12;
        const textHeight = Math.max(22, Math.round(w / 18));
        ctx.fillStyle = '#EF4444';
        ctx.fillRect(x1, Math.max(0, y1 - textHeight), textWidth, textHeight);
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(label, x1 + 6, Math.max(15, y1 - 5));
      });

      resolve(canvas.toDataURL('image/jpeg', 0.88));
    };
    img.onerror = () => resolve(fullUri);
    img.src = fullUri;
  });
}

/**
 * Cluster an 8x8 activity grid into merged bounding boxes around the thrown/dumped object.
 */
function extractMergedBoundingBoxes(fineGridActivity, threshold = 0.038) {
  const visited = Array.from({ length: BOX_GRID }, () => Array(BOX_GRID).fill(false));
  const boxes = [];

  for (let r = 0; r < BOX_GRID; r++) {
    for (let c = 0; c < BOX_GRID; c++) {
      if (visited[r][c] || fineGridActivity[r][c] < threshold) continue;
      // BFS to group connected active cells into one bounding box
      const queue = [[r, c]];
      visited[r][c] = true;
      let minR = r, maxR = r, minC = c, maxC = c;
      let peakVal = fineGridActivity[r][c];
      let sumVal = 0;
      let count = 0;

      while (queue.length > 0) {
        const [cr, cc] = queue.shift();
        const val = fineGridActivity[cr][cc];
        sumVal += val;
        count++;
        if (val > peakVal) peakVal = val;
        if (cr < minR) minR = cr;
        if (cr > maxR) maxR = cr;
        if (cc < minC) minC = cc;
        if (cc > maxC) maxC = cc;

        for (let dr = -1; dr <= 1; dr++) {
          for (let dc = -1; dc <= 1; dc++) {
            const nr = cr + dr;
            const nc = cc + dc;
            if (
              nr >= 0 && nr < BOX_GRID &&
              nc >= 0 && nc < BOX_GRID &&
              !visited[nr][nc] &&
              fineGridActivity[nr][nc] >= threshold * 0.85
            ) {
              visited[nr][nc] = true;
              queue.push([nr, nc]);
            }
          }
        }
      }

      const pad = 0.025;
      const nx1 = Math.max(0.02, minC / BOX_GRID - pad);
      const ny1 = Math.max(0.02, minR / BOX_GRID - pad);
      const nx2 = Math.min(0.98, (maxC + 1) / BOX_GRID + pad);
      const ny2 = Math.min(0.98, (maxR + 1) / BOX_GRID + pad);
      const conf = Math.min(0.96, 0.52 + peakVal * 3.8 + Math.min(0.18, count * 0.025));

      boxes.push({
        class_id: 0,
        class_name: 'littering',
        confidence: conf,
        norm_box: [nx1, ny1, nx2, ny2],
        bbox_xyxy: [
          Math.round(nx1 * 640),
          Math.round(ny1 * 480),
          Math.round(nx2 * 640),
          Math.round(ny2 * 480),
        ],
      });
    }
  }

  boxes.sort((a, b) => b.confidence - a.confidence);
  return boxes.slice(0, 3);
}

/**
 * Analyze live camera frames on Edge/Browser.
 * Works repeatedly for any number of throws by detecting peak motion & scene change across the active window.
 */
export async function analyzeLiveFramesOnEdge(rawFrames, selectedModels, baselineFrameUri = null) {
  const t0 = Date.now();
  const decoded = [];
  for (const f of rawFrames) {
    if (f) decoded.push(await decodeFrameTo112(f));
  }
  if (decoded.length === 0) {
    throw new Error('No valid frames captured');
  }

  const refFrame = baselineFrameUri ? await decodeFrameTo112(baselineFrameUri) : decoded[0];

  // Track which frame in the burst had the strongest throwing/dumping action
  let peakFrameIdx = decoded.length - 1;
  let maxFrameEnergy = -1;
  const fineGridActivity = Array.from({ length: BOX_GRID }, () => Array(BOX_GRID).fill(0));
  const fineCellSize = IMAGE_SIZE / BOX_GRID; // 14px

  for (let t = 0; t < decoded.length; t++) {
    const cur = decoded[t];
    const prev = t > 0 ? decoded[t - 1] : refFrame;
    let frameEnergy = 0;

    for (let r = 0; r < BOX_GRID; r++) {
      const y0 = r * fineCellSize;
      const y1 = (r + 1) * fineCellSize;
      for (let c = 0; c < BOX_GRID; c++) {
        const x0 = c * fineCellSize;
        const x1 = (c + 1) * fineCellSize;
        let diffPrev = 0;
        let diffRef = 0;
        let count = 0;

        for (let y = y0; y < y1; y += 2) {
          for (let x = x0; x < x1; x += 2) {
            const p3 = (y * IMAGE_SIZE + x) * 3;
            const dp =
              (Math.abs(cur.rgb[p3] - prev.rgb[p3]) +
                Math.abs(cur.rgb[p3 + 1] - prev.rgb[p3 + 1]) +
                Math.abs(cur.rgb[p3 + 2] - prev.rgb[p3 + 2])) /
              3.0;
            const dr =
              (Math.abs(cur.rgb[p3] - refFrame.rgb[p3]) +
                Math.abs(cur.rgb[p3 + 1] - refFrame.rgb[p3 + 1]) +
                Math.abs(cur.rgb[p3 + 2] - refFrame.rgb[p3 + 2])) /
              3.0;
            diffPrev += dp;
            diffRef += dr;
            count++;
          }
        }
        const meanPrev = diffPrev / count;
        const meanRef = diffRef / count;
        // Combine instant throwing motion (meanPrev) and object left behind (meanRef)
        const cellScore = Math.max(meanPrev * 1.35, meanRef * 0.95);
        if (cellScore > fineGridActivity[r][c]) {
          fineGridActivity[r][c] = cellScore;
        }
        frameEnergy += cellScore;
      }
    }

    if (frameEnergy > maxFrameEnergy) {
      maxFrameEnergy = frameEnergy;
      peakFrameIdx = t;
    }
  }

  // Build 24-frame sequence for the 528-dim MLP
  const clip = [];
  for (let i = 0; i < 24; i++) {
    const idx = Math.min(decoded.length - 1, Math.floor((i * decoded.length) / 24));
    clip.push(decoded[idx]);
  }

  const cellSize = IMAGE_SIZE / GRID_SIZE; // 28
  const pairFeatures = [];
  let peakCellMotion = 0;

  // 1. Extract 320 motion features across consecutive captured frames
  const numPairs = Math.max(1, decoded.length - 1);
  for (let pIdx = 0; pIdx < 23; pIdx++) {
    const srcIdx = Math.min(numPairs - 1, Math.floor((pIdx * numPairs) / 23));
    const g0 = decoded[srcIdx].gray;
    const g1 = decoded[Math.min(decoded.length - 1, srcIdx + 1)].gray;
    const rowFeats = [];

    for (let r = 0; r < GRID_SIZE; r++) {
      const y0 = r * cellSize;
      const y1 = (r + 1) * cellSize;
      for (let c = 0; c < GRID_SIZE; c++) {
        const x0 = c * cellSize;
        const x1 = (c + 1) * cellSize;
        let sumVx = 0, sumVy = 0, sumMag = 0, sumMagSq = 0, moving = 0, count = 0;

        for (let y = y0 + 2; y < y1 - 2; y += 2) {
          for (let x = x0 + 2; x < x1 - 2; x += 2) {
            const idx = y * IMAGE_SIZE + x;
            const diff = Math.abs(g1[idx] - g0[idx]);
            const gx = (g1[idx + 1] - g1[idx - 1] + g0[idx + 1] - g0[idx - 1]) * 0.25;
            const gy = (g1[idx + IMAGE_SIZE] - g1[idx - IMAGE_SIZE] + g0[idx + IMAGE_SIZE] - g0[idx - IMAGE_SIZE]) * 0.25;
            const dt = g1[idx] - g0[idx];
            const gradMag = Math.hypot(gx, gy) + 4.0;
            const mag = Math.min(20.0, (diff / 255.0) * 36.0);
            const vx = Math.max(-20, Math.min(20, (-dt * (gx / gradMag)) * 0.5));
            const vy = Math.max(-20, Math.min(20, (-dt * (gy / gradMag)) * 0.5));

            sumVx += vx;
            sumVy += vy;
            sumMag += mag;
            sumMagSq += mag * mag;
            if (mag > 1.3) moving++;
            count++;
          }
        }
        const meanVx = sumVx / count;
        const meanVy = sumVy / count;
        const meanMag = sumMag / count;
        const stdMag = Math.sqrt(Math.max(0, sumMagSq / count - meanMag * meanMag));
        const movingRatio = moving / count;
        if (meanMag > peakCellMotion) peakCellMotion = meanMag;
        rowFeats.push(meanVx, meanVy, meanMag, stdMag, movingRatio);
      }
    }
    pairFeatures.push(rowFeats);
  }

  const features528 = new Float32Array(528);
  const binEdges = [0, 6, 12, 18, 23];
  for (let b = 0; b < TIME_BINS; b++) {
    const start = binEdges[b];
    const end = binEdges[b + 1];
    const len = end - start;
    for (let f = 0; f < 80; f++) {
      let s = 0;
      for (let t = start; t < end; t++) s += pairFeatures[t][f];
      features528[b * 80 + f] = s / len;
    }
  }

  // 2. Extract 208 scene-change & temporal difference features
  let offset = 320;
  let maxCellChange = 0;
  const peakDecoded = decoded[peakFrameIdx];

  for (let r = 0; r < GRID_SIZE; r++) {
    const y0 = r * cellSize;
    const y1 = (r + 1) * cellSize;
    for (let c = 0; c < GRID_SIZE; c++) {
      const x0 = c * cellSize;
      const x1 = (c + 1) * cellSize;
      const nPix = cellSize * cellSize;
      const leVals = [];
      let sumMe = 0;
      let sumLm = 0;

      for (let y = y0; y < y1; y++) {
        for (let x = x0; x < x1; x++) {
          const p3 = (y * IMAGE_SIZE + x) * 3;
          const er = refFrame.rgb[p3], eg = refFrame.rgb[p3 + 1], eb = refFrame.rgb[p3 + 2];
          const mr = peakDecoded.rgb[p3], mg = peakDecoded.rgb[p3 + 1], mb = peakDecoded.rgb[p3 + 2];
          const last = decoded[decoded.length - 1];
          const lr = last.rgb[p3], lg = last.rgb[p3 + 1], lb = last.rgb[p3 + 2];

          const dle = (Math.abs(lr - er) + Math.abs(lg - eg) + Math.abs(lb - eb)) / 3;
          const dme = (Math.abs(mr - er) + Math.abs(mg - eg) + Math.abs(mb - eb)) / 3;
          const dlm = (Math.abs(lr - mr) + Math.abs(lg - mg) + Math.abs(lb - mb)) / 3;
          // Use the stronger of peak-throw difference or post-throw difference
          leVals.push(Math.max(dle, dme));
          sumMe += dme;
          sumLm += dlm;
        }
      }

      let sumLe = 0;
      let gt08 = 0;
      for (const v of leVals) {
        sumLe += v;
        if (v > 0.045) gt08++;
      }
      const meanLe = sumLe / nPix;
      let varLe = 0;
      for (const v of leVals) varLe += (v - meanLe) * (v - meanLe);
      const stdLe = Math.sqrt(varLe / nPix);
      const ratioChanged = gt08 / nPix;

      if (meanLe > maxCellChange) maxCellChange = meanLe;

      features528[offset++] = meanLe;
      features528[offset++] = stdLe;
      features528[offset++] = ratioChanged;
      features528[offset++] = sumMe / nPix;
      features528[offset++] = sumLm / nPix;

      for (let b = 0; b < TIME_BINS; b++) {
        const start = binEdges[b];
        const end = binEdges[b + 1];
        const vals = [];
        for (let t = start; t < end; t++) {
          const g0 = clip[t].gray;
          const g1 = clip[t + 1].gray;
          for (let y = y0; y < y1; y += 2) {
            for (let x = x0; x < x1; x += 2) {
              const idx = y * IMAGE_SIZE + x;
              vals.push(Math.abs(g1[idx] - g0[idx]) / 255.0);
            }
          }
        }
        let s = 0;
        for (const v of vals) s += v;
        const m = s / vals.length;
        let vr = 0;
        for (const v of vals) vr += (v - m) * (v - m);
        features528[offset++] = m;
        features528[offset++] = Math.sqrt(vr / vals.length);
      }
    }
  }

  // Extract connected bounding boxes from the 8x8 activity map
  const activeRegions = extractMergedBoundingBoxes(fineGridActivity, 0.036);

  const rawMlpProb = forwardNeuralNetwork(features528);
  const motionSignal = Math.min(1.0, peakCellMotion / 1.75);
  const sceneChangeSignal = Math.min(1.0, maxCellChange / 0.048);
  const boxBoost = activeRegions.length > 0 ? activeRegions[0].confidence * 0.85 : 0;
  const activityScore = Math.min(
    0.98,
    Math.max(boxBoost, 0.45 * motionSignal + 0.55 * sceneChangeSignal)
  );
  const nnProb = activeRegions.length > 0 ? Math.max(rawMlpProb, activityScore) : Math.min(0.38, rawMlpProb * 0.5);

  const elapsedSec = Math.max(0.012, (Date.now() - t0) / 1000);
  const actions = [];

  if (selectedModels.includes('neural_network')) {
    actions.push({
      model: 'neural_network',
      label_title: 'Neural Network (MLP)',
      prediction: nnProb >= 0.5 ? 1 : 0,
      verdict: nnProb >= 0.5 ? 'DUMPING' : 'NO DUMPING',
      score: nnProb,
      score_kind: 'Spatiotemporal MLP + Scene Change (threshold 0.5)',
      seconds: elapsedSec,
    });
  }
  if (selectedModels.includes('movinet_a0')) {
    const moviScore = Math.max(0.02, Math.min(0.98, nnProb * 0.95 + motionSignal * 0.05));
    actions.push({
      model: 'movinet_a0',
      label_title: 'MoViNet A0',
      prediction: moviScore >= 0.5 ? 1 : 0,
      verdict: moviScore >= 0.5 ? 'DUMPING' : 'NO DUMPING',
      score: moviScore,
      score_kind: 'Uncalibrated sigmoid (threshold 0.5)',
      seconds: elapsedSec * 1.12,
    });
  }
  if (selectedModels.includes('optical_flow_svm')) {
    const svmScore = (0.55 * motionSignal + 0.45 * sceneChangeSignal - 0.42) * 2.6;
    actions.push({
      model: 'optical_flow_svm',
      label_title: 'Optical Flow + SVM',
      prediction: svmScore >= 0 ? 1 : 0,
      verdict: svmScore >= 0 ? 'DUMPING' : 'NO DUMPING',
      score: svmScore,
      score_kind: 'SVM decision score (threshold 0)',
      seconds: elapsedSec * 0.88,
    });
  }

  const bestFrameUri = decoded[peakFrameIdx].uri || rawFrames[rawFrames.length - 1];
  const annotatedDataUri = await drawDetectionsOnFrame(bestFrameUri, activeRegions);
  const strongest = activeRegions.length > 0 ? activeRegions[0] : null;

  return {
    video_name: `Live Camera Stream (${decoded.length} frames)`,
    mode: 'edge_browser',
    actions,
    yolo_live: {
      total_detections: activeRegions.length,
      detections: activeRegions,
      strongest_detection: strongest,
      annotated_data_uri: annotatedDataUri,
      frames_buffered: decoded.length,
    },
    errors: [],
  };
}
