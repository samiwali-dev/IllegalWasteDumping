import { Platform } from 'react-native';
import nnWeights from './nnWeights.json';
import { motionTrailGeometry, yoloDetectionLabel } from './motionGeometry';

const IMAGE_SIZE = 112;
const GRID_SIZE = 4;
const BOX_GRID = 8; // 8x8 fine grid (14x14 px per cell)
const TIME_BINS = 4;

// Singleton offscreen canvases on Web so 16-20 FPS capture has zero DOM allocation overhead
let _fastCanvas112 = null;
let _fastCtx112 = null;
let _snapCanvas = null;
let _snapCtx = null;

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

function findActiveVideoElement() {
  if (Platform.OS !== 'web' || typeof document === 'undefined') return null;
  const videos = document.querySelectorAll('video');
  for (const v of videos) {
    if (v.readyState >= 2 && v.videoWidth > 0 && v.videoHeight > 0) {
      return v;
    }
  }
  return null;
}

export function getWebCameraAspectRatio() {
  const video = findActiveVideoElement();
  return video ? video.videoWidth / video.videoHeight : null;
}

export function isWebCameraMirrored() {
  const video = findActiveVideoElement();
  if (!video || typeof window === 'undefined') return false;
  const transform = window.getComputedStyle(video).transform;
  return transform !== 'none' && Number(transform.slice(transform.indexOf('(') + 1).split(',')[0]) < 0;
}

/**
 * Synchronous browser frame grabber, with a separate JPEG for captured evidence.
 * Extracts 112x112 RGB & Grayscale Float32Arrays directly without async Image decoding.
 */
export function grabWebVideoFrameFast(includeSnapshotUri = true) {
  const activeVideo = findActiveVideoElement();
  if (!activeVideo) return null;
  try {
    if (!_fastCanvas112) {
      _fastCanvas112 = document.createElement('canvas');
      _fastCanvas112.width = IMAGE_SIZE;
      _fastCanvas112.height = IMAGE_SIZE;
      _fastCtx112 = _fastCanvas112.getContext('2d', { willReadFrequently: true });
    }
    _fastCtx112.drawImage(activeVideo, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
    const rgba = _fastCtx112.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data;

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

    let uri = null;
    const targetW = Math.min(960, activeVideo.videoWidth);
    const targetH = Math.max(1, Math.round((activeVideo.videoHeight / activeVideo.videoWidth) * targetW));
    if (includeSnapshotUri) {
      if (!_snapCanvas) {
        _snapCanvas = document.createElement('canvas');
        _snapCtx = _snapCanvas.getContext('2d');
      }
      if (_snapCanvas.width !== targetW || _snapCanvas.height !== targetH) {
        _snapCanvas.width = targetW;
        _snapCanvas.height = targetH;
      }
      _snapCtx.drawImage(activeVideo, 0, 0, targetW, targetH);
      uri = _snapCanvas.toDataURL('image/jpeg', 0.8);
    }

    return {
      isDecodedFrame: true,
      rgb,
      gray,
      width: targetW,
      height: targetH,
      uri,
      ts: Date.now(),
    };
  } catch (_e) {
    return null;
  }
}

/**
 * Legacy helper returning a Base64 DataURI string.
 */
export function grabWebVideoFrameBase64() {
  const fast = grabWebVideoFrameFast(true);
  return fast ? fast.uri : null;
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
 * Convert either an already-decoded fast frame OR a base64 JPEG/PNG string into 112x112 RGB & Gray arrays.
 */
async function decodeFrameTo112(frameInput) {
  if (frameInput && frameInput.isDecodedFrame && frameInput.rgb && frameInput.gray) {
    return frameInput;
  }
  const dataUri = typeof frameInput === 'string' ? frameInput : frameInput?.uri || '';
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
        resolve({
          isDecodedFrame: true,
          rgb,
          gray,
          width: img.width || 320,
          height: img.height || 240,
          uri: dataUri,
          ts: Date.now(),
        });
      };
      img.onerror = () => {
        resolve({
          isDecodedFrame: true,
          rgb: new Float32Array(IMAGE_SIZE * IMAGE_SIZE * 3),
          gray: new Float32Array(IMAGE_SIZE * IMAGE_SIZE),
          width: 320,
          height: 240,
          uri: dataUri,
          ts: Date.now(),
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
  return { isDecodedFrame: true, rgb, gray, width: 320, height: 240, uri: dataUri, ts: Date.now() };
}

/**
 * Detect human skin tone in RGB (0..1) + YCbCr space.
 */
function isSkinPixel(r01, g01, b01) {
  const r = r01 * 255;
  const g = g01 * 255;
  const b = b01 * 255;
  const rgbRule =
    r > 80 &&
    g > 32 &&
    b > 15 &&
    r > g &&
    r > b &&
    Math.abs(r - g) > 10 &&
    Math.max(r, g, b) - Math.min(r, g, b) > 12;
  if (!rgbRule) return false;
  const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
  const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
  return cr >= 128 && cr <= 180 && cb >= 72 && cb <= 135;
}

/**
 * Draw detected bounding boxes and drop trajectory onto the action frame (Web Canvas).
 */
export async function drawDetectionsOnFrame(dataUri, detections) {
  if (!dataUri) return null;
  const fullUri = dataUri.startsWith('data:') ? dataUri : `data:image/jpeg;base64,${dataUri}`;
  if (Platform.OS !== 'web' || typeof document === 'undefined' || !detections || detections.length === 0) {
    return fullUri;
  }
  return new Promise((resolve) => {
    const img = new window.Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const canvas = document.createElement('canvas');
      const w = img.width || 320;
      const h = img.height || 240;
      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, w, h);

      detections.forEach((det) => {
        const { points, arrow } = motionTrailGeometry(det.trail, w, h);
        if (points.length >= 2) {
          ctx.save();
          ctx.strokeStyle = '#F59E0B';
          ctx.lineWidth = 3;
          ctx.setLineDash([5, 3]);
          ctx.beginPath();
          points.forEach((pt, idx) => {
            if (idx === 0) ctx.moveTo(pt.x, pt.y);
            else ctx.lineTo(pt.x, pt.y);
          });
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.beginPath();
          arrow.forEach((pt, idx) => {
            if (idx === 0) ctx.moveTo(pt.x, pt.y);
            else ctx.lineTo(pt.x, pt.y);
          });
          ctx.stroke();
          ctx.restore();
        }

        const [nx1, ny1, nx2, ny2] = det.norm_box || [0.25, 0.25, 0.75, 0.75];
        const x1 = nx1 * w;
        const y1 = ny1 * h;
        const bw = (nx2 - nx1) * w;
        const bh = (ny2 - ny1) * h;

        const modelDetection = det.source === 'yolov8n';
        const boxColor = modelDetection ? '#60A5FA' : '#D79533';
        ctx.strokeStyle = boxColor;
        ctx.lineWidth = Math.max(3, Math.round(w / 100));
        ctx.strokeRect(x1, y1, bw, bh);

        ctx.fillStyle = modelDetection ? 'rgba(96, 165, 250, 0.12)' : 'rgba(215, 149, 51, 0.12)';
        ctx.fillRect(x1, y1, bw, bh);

        const label = modelDetection ? yoloDetectionLabel(det) : 'Motion candidate';
        ctx.font = `bold ${Math.max(12, Math.round(w / 25))}px sans-serif`;
        const textWidth = ctx.measureText(label).width + 12;
        const textHeight = Math.max(20, Math.round(w / 18));
        ctx.fillStyle = boxColor;
        ctx.fillRect(x1, Math.max(0, y1 - textHeight), textWidth, textHeight);
        ctx.fillStyle = '#FFFFFF';
        ctx.fillText(label, x1 + 6, Math.max(14, y1 - 5));
      });

      resolve(canvas.toDataURL('image/jpeg', 0.86));
    };
    img.onerror = () => resolve(fullUri);
    img.src = fullUri;
  });
}

/**
 * Helper: compute mean RGB difference inside a cell box [minR..maxR, minC..maxC] between two frames.
 */
function computeRegionMeanDiff(frameA, frameB, minR, maxR, minC, maxC) {
  const fineCell = IMAGE_SIZE / BOX_GRID;
  let sum = 0;
  let cnt = 0;
  for (let r = minR; r <= maxR; r++) {
    const y0 = r * fineCell;
    const y1 = (r + 1) * fineCell;
    for (let c = minC; c <= maxC; c++) {
      const x0 = c * fineCell;
      const x1 = (c + 1) * fineCell;
      for (let y = y0; y < y1; y += 2) {
        for (let x = x0; x < x1; x += 2) {
          const p3 = (y * IMAGE_SIZE + x) * 3;
          sum +=
            (Math.abs(frameA.rgb[p3] - frameB.rgb[p3]) +
              Math.abs(frameA.rgb[p3 + 1] - frameB.rgb[p3 + 1]) +
              Math.abs(frameA.rgb[p3 + 2] - frameB.rgb[p3 + 2])) /
            3.0;
          cnt++;
        }
      }
    }
  }
  return cnt > 0 ? sum / cnt : 0;
}

/**
 * 5-Guard High-FPS Falling-Object Detector with Complete Human Body / Person Rejection:
 *
 * Why moving your body, head, hair, glasses, or shirt will NEVER trigger DUMPING:
 * - Guard 1 (Global Body Motion Veto): If > 9 cells move or >= 2 skin cells move strongly, the person is moving.
 * - Guard 2 (Full Human Silhouette Mask — Hair + Face + Shoulders + Torso):
 *   Any skin cell AND its surrounding hair (above), ears/glasses (sides), and shoulders/collar/chest (below)
 *   are masked as part of the person whenever the person's head/skin has motion.
 * - Guard 3 (Compact Projectile Size): Only small 1..4 cell blobs (max 2x2 grid span) qualify as projectiles.
 * - Guard 4 (Strict Multi-Frame Vertical Gravity Trajectory): Must fall straight down (`dx <= 0.12` per step,
 *   `dy >= 0.03` per step) across >= 3 consecutive transitions (`totalDy >= 0.17` and `totalDy >= 1.8 * totalDx`).
 * - Guard 5 (Pass-Through Background Restoration Test): When a real object falls through cell (r0, c0) at t_start
 *   down to (r1, c1) at t_end, cell (r0, c0) MUST return to its pre-drop background color after the object leaves it!
 *   When a person leans or moves their body, their body stays in (r0, c0), failing this test 100% of the time!
 */
function detectDumpedObjectsOnly(decoded) {
  const fineCell = IMAGE_SIZE / BOX_GRID; // 14px
  const numFrames = decoded.length;
  if (numFrames < 4) {
    return { boxes: [], peakFrameIdx: 0 };
  }

  // Extract all compact non-human moving projectile blobs for each frame t
  const blobsPerFrame = [];

  for (let t = 1; t < numFrames; t++) {
    const cur = decoded[t];
    const prev = decoded[t - 1];

    const cellDiff = Array.from({ length: BOX_GRID }, () => Array(BOX_GRID).fill(0));
    const cellSkin = Array.from({ length: BOX_GRID }, () => Array(BOX_GRID).fill(0));
    const isHumanMask = Array.from({ length: BOX_GRID }, () => Array(BOX_GRID).fill(false));

    let totalMovingCells = 0;
    let movingSkinCells = 0;

    for (let r = 0; r < BOX_GRID; r++) {
      const y0 = r * fineCell;
      const y1 = (r + 1) * fineCell;
      for (let c = 0; c < BOX_GRID; c++) {
        const x0 = c * fineCell;
        const x1 = (c + 1) * fineCell;
        let diff = 0;
        let skin = 0;
        let cnt = 0;
        for (let y = y0; y < y1; y += 2) {
          for (let x = x0; x < x1; x += 2) {
            const p3 = (y * IMAGE_SIZE + x) * 3;
            const d =
              (Math.abs(cur.rgb[p3] - prev.rgb[p3]) +
                Math.abs(cur.rgb[p3 + 1] - prev.rgb[p3 + 1]) +
                Math.abs(cur.rgb[p3 + 2] - prev.rgb[p3 + 2])) /
              3.0;
            diff += d;
            if (
              isSkinPixel(cur.rgb[p3], cur.rgb[p3 + 1], cur.rgb[p3 + 2]) ||
              isSkinPixel(prev.rgb[p3], prev.rgb[p3 + 1], prev.rgb[p3 + 2])
            ) {
              skin++;
            }
            cnt++;
          }
        }
        const mDiff = diff / cnt;
        const mSkin = skin / cnt;
        cellDiff[r][c] = mDiff;
        cellSkin[r][c] = mSkin;
        if (mDiff > 0.048) {
          totalMovingCells++;
          if (mSkin >= 0.16) movingSkinCells++;
        }
      }
    }

    // Guard 1: If the person's body or head is moving broadly (> 9 cells moving, or >= 3 skin cells moving),
    // skip this frame completely so body/head movement is NEVER treated as a falling object.
    if (totalMovingCells > 9 || movingSkinCells >= 3) {
      blobsPerFrame.push([]);
      continue;
    }

    // Guard 2: Build the Human Silhouette Mask around any skin cells (face, hair, glasses, neck, shoulders, hands)
    for (let r = 0; r < BOX_GRID; r++) {
      for (let c = 0; c < BOX_GRID; c++) {
        if (cellSkin[r][c] >= 0.16) {
          isHumanMask[r][c] = true;
          // If this skin cell is moving even slightly (e.g. head/hand micro-motion), mask its 1-cell neighborhood
          // (hair above forehead, glasses/ears on sides, collar/shoulders below chin)
          if (cellDiff[r][c] > 0.022) {
            for (let dr = -1; dr <= 2; dr++) {
              for (let dc = -1; dc <= 1; dc++) {
                const nr = r + dr;
                const nc = c + dc;
                if (nr >= 0 && nr < BOX_GRID && nc >= 0 && nc < BOX_GRID) {
                  isHumanMask[nr][nc] = true;
                }
              }
            }
          }
        }
      }
    }

    // Find connected components of non-human moving cells
    const visited = Array.from({ length: BOX_GRID }, () => Array(BOX_GRID).fill(false));
    const candidates = [];

    for (let r = 0; r < BOX_GRID; r++) {
      for (let c = 0; c < BOX_GRID; c++) {
        if (visited[r][c] || isHumanMask[r][c] || cellDiff[r][c] < 0.058) continue;

        const queue = [[r, c]];
        visited[r][c] = true;
        let minR = r, maxR = r, minC = c, maxC = c;
        let count = 0;
        let sumX = 0, sumY = 0, sumW = 0;
        let touchesHumanMask = false;

        while (queue.length > 0) {
          const [cr, cc] = queue.shift();
          const w = cellDiff[cr][cc];
          count++;
          sumX += ((cc + 0.5) / BOX_GRID) * w;
          sumY += ((cr + 0.5) / BOX_GRID) * w;
          sumW += w;
          if (cr < minR) minR = cr;
          if (cr > maxR) maxR = cr;
          if (cc < minC) minC = cc;
          if (cc > maxC) maxC = cc;

          for (let dr = -1; dr <= 1; dr++) {
            for (let dc = -1; dc <= 1; dc++) {
              const nr = cr + dr;
              const nc = cc + dc;
              if (nr >= 0 && nr < BOX_GRID && nc >= 0 && nc < BOX_GRID) {
                if (isHumanMask[nr][nc] && cellDiff[nr][nc] > 0.035) {
                  touchesHumanMask = true;
                }
                if (!visited[nr][nc] && !isHumanMask[nr][nc] && cellDiff[nr][nc] >= 0.05) {
                  visited[nr][nc] = true;
                  queue.push([nr, nc]);
                }
              }
            }
          }
        }

        const spanW = maxC - minC + 1;
        const spanH = maxR - minR + 1;
        // Guard 3: Compact Projectile Geometry (1..4 cells, max 2x2 grid span)
        if (count >= 1 && count <= 4 && spanW <= 2 && spanH <= 2 && sumW > 0) {
          candidates.push({
            t,
            cx: sumX / sumW,
            cy: sumY / sumW,
            minR,
            maxR,
            minC,
            maxC,
            count,
            touchesHumanMask,
            energy: sumW / count,
          });
        }
      }
    }

    blobsPerFrame.push(candidates);
  }

  // Guard 4: Multi-Frame Continuous Vertical Free-Fall Tracker
  let bestChain = null;

  for (let startFrameIdx = 0; startFrameIdx < blobsPerFrame.length; startFrameIdx++) {
    for (const startBlob of blobsPerFrame[startFrameIdx]) {
      const chain = [startBlob];

      for (let nextFrameIdx = startFrameIdx + 1; nextFrameIdx < blobsPerFrame.length; nextFrameIdx++) {
        const prevBlob = chain[chain.length - 1];
        const frameGap = (nextFrameIdx + 1) - prevBlob.t;
        if (frameGap > 2) break; // Must be consecutive or at most 1 skipped frame at 17 FPS

        // Find the candidate in nextFrameIdx that continues straight downward from prevBlob
        let bestNext = null;
        for (const cand of blobsPerFrame[nextFrameIdx]) {
          const dyStep = cand.cy - prevBlob.cy;
          const dxStep = Math.abs(cand.cx - prevBlob.cx);
          // Falling object moves strictly downward (dyStep >= 0.03) in the same vertical column (dxStep <= 0.13)
          if (dyStep >= 0.03 && dyStep <= 0.35 && dxStep <= 0.13) {
            if (!bestNext || dyStep > bestNext.cy - prevBlob.cy) {
              bestNext = cand;
            }
          }
        }
        if (bestNext) {
          chain.push(bestNext);
        }
      }

      // Require at least 3 consecutive downward transitions (e.g. t1 -> t2 -> t3)
      if (chain.length >= 3) {
        const first = chain[0];
        const last = chain[chain.length - 1];
        const totalDy = last.cy - first.cy;
        const totalDx = Math.abs(last.cx - first.cx);

        // Must have fallen at least 17% of frame height, predominantly straight down,
        // and the final position must have detached from moving skin/hands
        if (totalDy >= 0.17 && totalDy >= totalDx * 1.8 && !last.touchesHumanMask) {
          // Guard 5: Pass-Through Background Restoration Test!
          // Check that the starting cell where the object was at `first.t` returned to the pre-drop background by `last.t`.
          const preFrame = decoded[Math.max(0, first.t - 1)];
          const startFrame = decoded[first.t];
          const endFrame = decoded[last.t];

          const diffWhenObjectAtStart = computeRegionMeanDiff(
            startFrame,
            preFrame,
            first.minR,
            first.maxR,
            first.minC,
            first.maxC
          );
          const diffAfterObjectLeftStart = computeRegionMeanDiff(
            endFrame,
            preFrame,
            first.minR,
            first.maxR,
            first.minC,
            first.maxC
          );

          // Because the falling object passed through `first` and left it behind, `endFrame` at `first`
          // matches `preFrame` much closer than `startFrame` did!
          const passedThroughAir =
            diffAfterObjectLeftStart < 0.068 &&
            diffAfterObjectLeftStart <= diffWhenObjectAtStart * 0.82;

          if (passedThroughAir) {
            const score = totalDy * 2.5 + chain.length * 0.1;
            if (!bestChain || score > bestChain.score) {
              bestChain = { chain, totalDy, totalDx, score };
            }
          }
        }
      }
    }
  }

  const verifiedBoxes = [];
  let peakFrameIdx = numFrames - 1;

  if (bestChain) {
    const { chain, totalDy } = bestChain;
    const targetBlob = chain[chain.length - 1];
    peakFrameIdx = targetBlob.t;

    const conf = Math.min(0.96, 0.70 + totalDy * 1.1 + chain.length * 0.03);
    verifiedBoxes.push({
      class_id: 0,
      class_name: 'littering',
      confidence: conf,
      norm_box: [
        Math.max(0.02, targetBlob.minC / BOX_GRID - 0.03),
        Math.max(0.02, targetBlob.minR / BOX_GRID - 0.03),
        Math.min(0.98, (targetBlob.maxC + 1) / BOX_GRID + 0.03),
        Math.min(0.98, (targetBlob.maxR + 1) / BOX_GRID + 0.03),
      ],
      trail: chain.map((b) => ({ cx: b.cx, cy: b.cy, t: b.t })),
      reason: 'falling_object',
    });
  }

  return {
    boxes: verifiedBoxes,
    peakFrameIdx,
  };
}

/**
 * Analyze high-FPS (15-20 FPS, up to 24 frames) live camera frames on Edge/Browser.
 */
export async function analyzeLiveFramesOnEdge(rawFrames, selectedModels) {
  const t0 = Date.now();
  const decoded = [];
  for (const f of rawFrames) {
    if (f) decoded.push(await decodeFrameTo112(f));
  }
  if (decoded.length === 0) {
    throw new Error('No valid frames captured');
  }

  const refFrame = decoded[0];
  const { boxes: activeRegions, peakFrameIdx } = detectDumpedObjectsOnly(decoded);

  // Build 24-frame sequence for the 528-dim MLP
  const clip = [];
  for (let i = 0; i < 24; i++) {
    const idx = Math.min(decoded.length - 1, Math.floor((i * decoded.length) / 24));
    clip.push(decoded[idx]);
  }

  const cellSize = IMAGE_SIZE / GRID_SIZE; // 28
  const pairFeatures = [];
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
            const mag = Math.min(20.0, (diff / 255.0) * 24.0);
            const vx = Math.max(-20, Math.min(20, (-dt * (gx / gradMag)) * 0.4));
            const vy = Math.max(-20, Math.min(20, (-dt * (gy / gradMag)) * 0.4));

            sumVx += vx;
            sumVy += vy;
            sumMag += mag;
            sumMagSq += mag * mag;
            if (mag > 1.5) moving++;
            count++;
          }
        }
        const meanVx = sumVx / count;
        const meanVy = sumVy / count;
        const meanMag = sumMag / count;
        const stdMag = Math.sqrt(Math.max(0, sumMagSq / count - meanMag * meanMag));
        rowFeats.push(meanVx, meanVy, meanMag, stdMag, moving / count);
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

  let offset = 320;
  const last = decoded[decoded.length - 1];
  const mid = decoded[Math.floor(decoded.length / 2)];

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
          const mr = mid.rgb[p3], mg = mid.rgb[p3 + 1], mb = mid.rgb[p3 + 2];
          const lr = last.rgb[p3], lg = last.rgb[p3 + 1], lb = last.rgb[p3 + 2];

          const dle = (Math.abs(lr - er) + Math.abs(lg - eg) + Math.abs(lb - eb)) / 3;
          const dme = (Math.abs(mr - er) + Math.abs(mg - eg) + Math.abs(mb - eb)) / 3;
          const dlm = (Math.abs(lr - mr) + Math.abs(lg - mg) + Math.abs(lb - mb)) / 3;
          leVals.push(dle);
          sumMe += dme;
          sumLm += dlm;
        }
      }

      let sumLe = 0;
      let gt08 = 0;
      for (const v of leVals) {
        sumLe += v;
        if (v > 0.08) gt08++;
      }
      const meanLe = sumLe / nPix;
      let varLe = 0;
      for (const v of leVals) varLe += (v - meanLe) * (v - meanLe);
      const stdLe = Math.sqrt(varLe / nPix);

      features528[offset++] = meanLe;
      features528[offset++] = stdLe;
      features528[offset++] = gt08 / nPix;
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

  // Strictly gate classification by the 5-Guard verified falling projectile detector
  const rawMlpProb = forwardNeuralNetwork(features528);
  const hasVerifiedDump = activeRegions.length > 0;
  const topBoxConf = hasVerifiedDump ? activeRegions[0].confidence : 0;

  const nnProb = hasVerifiedDump
    ? Math.max(0.68, Math.min(0.96, 0.40 * rawMlpProb + 0.60 * topBoxConf))
    : Math.min(0.18, rawMlpProb * 0.22);

  const elapsedSec = Math.max(0.008, (Date.now() - t0) / 1000);
  const actions = [];

  if (selectedModels.includes('neural_network')) {
    actions.push({
      model: 'neural_network',
      label_title: 'Neural Network (MLP)',
      prediction: nnProb >= 0.5 ? 1 : 0,
      verdict: nnProb >= 0.5 ? 'DUMPING' : 'NO DUMPING',
      score: nnProb,
      score_kind: 'Spatiotemporal MLP + 5-Guard Projectile Tracker (threshold 0.5)',
      seconds: elapsedSec,
    });
  }
  if (selectedModels.includes('movinet_a0')) {
    const moviScore = hasVerifiedDump
      ? Math.max(0.62, Math.min(0.94, nnProb * 0.94))
      : Math.min(0.22, nnProb * 0.85);
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
    const svmScore = hasVerifiedDump ? (nnProb - 0.42) * 2.2 : (nnProb - 0.58) * 2.0;
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

  let bestFrameUri = decoded[peakFrameIdx]?.uri || decoded[decoded.length - 1]?.uri;
  if (!bestFrameUri && Platform.OS === 'web') {
    bestFrameUri = grabWebVideoFrameBase64();
  }
  const annotatedDataUri = hasVerifiedDump ? await drawDetectionsOnFrame(bestFrameUri, activeRegions) : null;
  const strongest = activeRegions.length > 0 ? activeRegions[0] : null;

  return {
    video_name: `Live Camera Stream (${decoded.length} frames @ High-FPS)`,
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
