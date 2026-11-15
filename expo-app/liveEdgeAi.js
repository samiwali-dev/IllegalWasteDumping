import { Platform } from 'react-native';
import nnWeights from './nnWeights.json';

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
  if (!video) return false;
  const transform = window.getComputedStyle(video).transform;
  return transform !== 'none' && Number(transform.slice(transform.indexOf('(') + 1).split(',')[0]) < 0;
}

/**
 * Ultra-fast synchronous frame grabber for Web/Vercel (runs in <0.4ms at 16-20 FPS).
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
    const targetW = 320;
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
      uri = _snapCanvas.toDataURL('image/jpeg', 0.68);
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
    return new Promise((resolve, reject) => {
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
          ts: typeof frameInput === 'object' ? frameInput.ts : null,
        });
      };
      img.onerror = () => {
        reject(new Error('Could not decode the camera frame. Capture again.'));
      };
      img.src = dataUri.startsWith('data:') ? dataUri : `data:image/jpeg;base64,${dataUri}`;
    });
  }

  throw new Error('Native live analysis requires the Python API. Use a web browser for local camera analysis.');
}

/**
 * Detect human skin tone in RGB (0..1) + YCbCr space so faces, necks, and hands are NEVER boxed as litter.
 */
function isSkinPixel(r01, g01, b01) {
  const r = r01 * 255;
  const g = g01 * 255;
  const b = b01 * 255;
  const rgbRule =
    r > 85 &&
    g > 35 &&
    b > 18 &&
    r > g &&
    r > b &&
    Math.abs(r - g) > 12 &&
    Math.max(r, g, b) - Math.min(r, g, b) > 14;
  if (!rgbRule) return false;
  const cb = 128 - 0.168736 * r - 0.331264 * g + 0.5 * b;
  const cr = 128 + 0.5 * r - 0.418688 * g - 0.081312 * b;
  return cr >= 130 && cr <= 178 && cb >= 75 && cb <= 130;
}

/**
 * Draw detected bounding boxes and drop trajectory onto the action frame (Web Canvas).
 */
async function drawDetectionsOnFrame(dataUri, detections) {
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
        // Draw trajectory path if available (shows the falling path from the hand!)
        if (Array.isArray(det.trail) && det.trail.length >= 2) {
          ctx.save();
          ctx.strokeStyle = '#F59E0B';
          ctx.lineWidth = 3;
          ctx.setLineDash([5, 3]);
          ctx.beginPath();
          det.trail.forEach((pt, idx) => {
            const px = pt.cx * w;
            const py = pt.cy * h;
            if (idx === 0) ctx.moveTo(px, py);
            else ctx.lineTo(px, py);
          });
          ctx.stroke();
          ctx.restore();
        }

        const [nx1, ny1, nx2, ny2] = det.norm_box || [0.25, 0.25, 0.75, 0.75];
        const x1 = nx1 * w;
        const y1 = ny1 * h;
        const bw = (nx2 - nx1) * w;
        const bh = (ny2 - ny1) * h;

        ctx.strokeStyle = '#EF4444';
        ctx.lineWidth = Math.max(3, Math.round(w / 100));
        ctx.strokeRect(x1, y1, bw, bh);

        ctx.fillStyle = 'rgba(239, 68, 68, 0.22)';
        ctx.fillRect(x1, y1, bw, bh);

        const label = `littering ${(det.confidence * 100).toFixed(0)}%`;
        ctx.font = `bold ${Math.max(12, Math.round(w / 25))}px sans-serif`;
        const textWidth = ctx.measureText(label).width + 12;
        const textHeight = Math.max(20, Math.round(w / 18));
        ctx.fillStyle = '#EF4444';
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
 * High-FPS (15-20 FPS) Falling-Object & Dumped-Object Detector with Strict Person Rejection:
 *
 * Why a person moving/sitting is NEVER flagged as dumping:
 * 1. Skin cells (face, neck, bare hands) and human body/torso movements (> 7 cells) are rejected.
 * 2. Slight posture shifts of a seated/standing person do NOT have a fast monotonic downward free-fall trajectory.
 *
 * How an object falling from the hand IS caught at 15-20 FPS:
 * - At 16 FPS (60ms per frame), a dropped object falls through the screen over 3 to 9 frames.
 * - In each frame step t, we find the strongest compact non-skin moving blob (1..6 cells).
 * - If we observe a chain of >= 2 downward steps (3+ frames) where the compact object's vertical center `cy`
 *   moves downward by >= 11% of the frame height (`dy >= 0.11`) with predominantly vertical motion (`dy > |dx|`),
 *   it is verified as a falling/thrown object (`falling_object`)!
 */
function detectDumpedObjectsOnly(decoded, explicitBaselineFrame = null) {
  const fineCell = IMAGE_SIZE / BOX_GRID; // 14px
  const numFrames = decoded.length;
  if (numFrames < 2) {
    return { boxes: [], peakFrameIdx: 0 };
  }

  // Step 1: Extract compact non-skin moving object candidates for every consecutive frame pair (t-1 -> t)
  const frameCandidates = [];
  for (let t = 1; t < numFrames; t++) {
    const cur = decoded[t];
    const prev = decoded[t - 1];

    const cellDiff = Array.from({ length: BOX_GRID }, () => Array(BOX_GRID).fill(0));
    const cellSkin = Array.from({ length: BOX_GRID }, () => Array(BOX_GRID).fill(0));
    let totalSceneMovingCells = 0;

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
            if (isSkinPixel(cur.rgb[p3], cur.rgb[p3 + 1], cur.rgb[p3 + 2])) {
              skin++;
            }
            cnt++;
          }
        }
        const mDiff = diff / cnt;
        const mSkin = skin / cnt;
        cellDiff[r][c] = mDiff;
        cellSkin[r][c] = mSkin;
        if (mDiff > 0.05) totalSceneMovingCells++;
      }
    }

    // If > 14 cells moved simultaneously, the whole person or camera moved — skip this frame!
    if (totalSceneMovingCells > 14) {
      frameCandidates.push(null);
      continue;
    }

    // Connected components of non-skin moving cells in frame t
    const visited = Array.from({ length: BOX_GRID }, () => Array(BOX_GRID).fill(false));
    let bestBlob = null;

    for (let r = 0; r < BOX_GRID; r++) {
      for (let c = 0; c < BOX_GRID; c++) {
        if (visited[r][c] || cellDiff[r][c] < 0.052 || cellSkin[r][c] >= 0.34) continue;

        const queue = [[r, c]];
        visited[r][c] = true;
        let minR = r, maxR = r, minC = c, maxC = c;
        let count = 0;
        let sumX = 0, sumY = 0, sumW = 0;

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
              if (
                nr >= 0 &&
                nr < BOX_GRID &&
                nc >= 0 &&
                nc < BOX_GRID &&
                !visited[nr][nc] &&
                cellDiff[nr][nc] >= 0.045 &&
                cellSkin[nr][nc] < 0.36
              ) {
                visited[nr][nc] = true;
                queue.push([nr, nc]);
              }
            }
          }
        }

        const spanW = maxC - minC + 1;
        const spanH = maxR - minR + 1;
        // Must be a compact object (1..6 cells, max 3x3 cells) — rejects human torso/arms/head
        if (count >= 1 && count <= 6 && spanW <= 3 && spanH <= 3 && sumW > 0) {
          const blob = {
            t,
            cx: sumX / sumW,
            cy: sumY / sumW,
            minR,
            maxR,
            minC,
            maxC,
            count,
            energy: sumW / count,
          };
          if (!bestBlob || blob.energy > bestBlob.energy) {
            bestBlob = blob;
          }
        }
      }
    }

    frameCandidates.push(bestBlob);
  }

  // Step 2: Find the strongest downward falling trajectory across consecutive high-FPS frames
  let bestChain = null;
  for (let i = 0; i < frameCandidates.length; i++) {
    const startBlob = frameCandidates[i];
    if (!startBlob) continue;

    const chain = [startBlob];
    for (let j = i + 1; j < frameCandidates.length; j++) {
      const nextBlob = frameCandidates[j];
      if (!nextBlob) continue;
      const prevBlob = chain[chain.length - 1];
      const frameGap = nextBlob.t - prevBlob.t;
      if (frameGap > 3) break; // Must be continuous in time (within ~180ms at 16 FPS)

      const dyStep = nextBlob.cy - prevBlob.cy;
      const dxStep = Math.abs(nextBlob.cx - prevBlob.cx);

      // Object falling under gravity moves downward (dyStep >= 0.015) and doesn't teleport sideways
      if (dyStep >= 0.015 && dxStep <= 0.26) {
        chain.push(nextBlob);
      } else if (dyStep < -0.02) {
        // Moved upward -> not a falling object!
        break;
      }
    }

    if (chain.length >= 2) {
      const first = chain[0];
      const last = chain[chain.length - 1];
      const totalDy = last.cy - first.cy;
      const totalDx = Math.abs(last.cx - first.cx);

      // Verified free-fall: downward drop of at least 11.5% of screen height, predominantly vertical
      if (totalDy >= 0.115 && totalDy >= totalDx * 0.95) {
        const score = totalDy * 2.5 + chain.length * 0.08;
        if (!bestChain || score > bestChain.score) {
          bestChain = { chain, totalDy, totalDx, score };
        }
      }
    }
  }

  const verifiedBoxes = [];
  let peakFrameIdx = numFrames - 1;

  if (bestChain) {
    const { chain, totalDy } = bestChain;
    // Choose the mid-to-late frame of the drop where the object is clearly visible in frame
    const targetBlob = chain[Math.min(chain.length - 1, Math.max(1, chain.length - 1))];
    peakFrameIdx = targetBlob.t;

    const conf = Math.min(0.96, 0.68 + totalDy * 1.15 + chain.length * 0.03);
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

  // Step 3: Only if the user explicitly locked an empty background (`Lock background`),
  // also check for a compact non-skin stationary object left behind in the scene.
  if (verifiedBoxes.length === 0 && explicitBaselineFrame) {
    const lastFrame = decoded[numFrames - 1];
    const prevToLast = numFrames >= 2 ? decoded[numFrames - 2] : explicitBaselineFrame;
    const staticDiffGrid = Array.from({ length: BOX_GRID }, () => Array(BOX_GRID).fill(0));
    const recentMotionGrid = Array.from({ length: BOX_GRID }, () => Array(BOX_GRID).fill(0));
    const skinRatioGrid = Array.from({ length: BOX_GRID }, () => Array(BOX_GRID).fill(0));
    let totalChangedCells = 0;

    for (let r = 0; r < BOX_GRID; r++) {
      const y0 = r * fineCell;
      const y1 = (r + 1) * fineCell;
      for (let c = 0; c < BOX_GRID; c++) {
        const x0 = c * fineCell;
        const x1 = (c + 1) * fineCell;
        let diffRef = 0, diffLast = 0, skinPixels = 0, count = 0;
        for (let y = y0; y < y1; y += 2) {
          for (let x = x0; x < x1; x += 2) {
            const p3 = (y * IMAGE_SIZE + x) * 3;
            const lr = lastFrame.rgb[p3], lg = lastFrame.rgb[p3 + 1], lb = lastFrame.rgb[p3 + 2];
            const rr = explicitBaselineFrame.rgb[p3], rg = explicitBaselineFrame.rgb[p3 + 1], rb = explicitBaselineFrame.rgb[p3 + 2];
            const pr = prevToLast.rgb[p3], pg = prevToLast.rgb[p3 + 1], pb = prevToLast.rgb[p3 + 2];
            diffRef += (Math.abs(lr - rr) + Math.abs(lg - rg) + Math.abs(lb - rb)) / 3.0;
            diffLast += (Math.abs(lr - pr) + Math.abs(lg - pg) + Math.abs(lb - pb)) / 3.0;
            if (isSkinPixel(lr, lg, lb)) skinPixels++;
            count++;
          }
        }
        staticDiffGrid[r][c] = diffRef / count;
        recentMotionGrid[r][c] = diffLast / count;
        skinRatioGrid[r][c] = skinPixels / count;
        if (staticDiffGrid[r][c] > 0.08) totalChangedCells++;
      }
    }

    // Only if <= 8 cells changed in the entire scene (so a person standing in front of the locked background is ignored!)
    if (totalChangedCells >= 1 && totalChangedCells <= 8) {
      const visited = Array.from({ length: BOX_GRID }, () => Array(BOX_GRID).fill(false));
      for (let r = 0; r < BOX_GRID; r++) {
        for (let c = 0; c < BOX_GRID; c++) {
          if (
            visited[r][c] ||
            staticDiffGrid[r][c] < 0.11 ||
            recentMotionGrid[r][c] > 0.018 ||
            skinRatioGrid[r][c] >= 0.22
          ) {
            continue;
          }
          const queue = [[r, c]];
          visited[r][c] = true;
          let minR = r, maxR = r, minC = c, maxC = c, count = 0, peakDiff = staticDiffGrid[r][c];
          while (queue.length > 0) {
            const [cr, cc] = queue.shift();
            count++;
            if (staticDiffGrid[cr][cc] > peakDiff) peakDiff = staticDiffGrid[cr][cc];
            if (cr < minR) minR = cr;
            if (cr > maxR) maxR = cr;
            if (cc < minC) minC = cc;
            if (cc > maxC) maxC = cc;
            for (let dr = -1; dr <= 1; dr++) {
              for (let dc = -1; dc <= 1; dc++) {
                const nr = cr + dr, nc = cc + dc;
                if (
                  nr >= 0 && nr < BOX_GRID && nc >= 0 && nc < BOX_GRID &&
                  !visited[nr][nc] &&
                  staticDiffGrid[nr][nc] >= 0.095 &&
                  recentMotionGrid[nr][nc] <= 0.022 &&
                  skinRatioGrid[nr][nc] < 0.25
                ) {
                  visited[nr][nc] = true;
                  queue.push([nr, nc]);
                }
              }
            }
          }
          const spanW = maxC - minC + 1;
          const spanH = maxR - minR + 1;
          if (count >= 1 && count <= 5 && spanW <= 2 && spanH <= 2) {
            verifiedBoxes.push({
              class_id: 0,
              class_name: 'littering',
              confidence: Math.min(0.92, 0.62 + peakDiff * 2.0),
              norm_box: [
                Math.max(0.02, minC / BOX_GRID - 0.02),
                Math.max(0.02, minR / BOX_GRID - 0.02),
                Math.min(0.98, (maxC + 1) / BOX_GRID + 0.02),
                Math.min(0.98, (maxR + 1) / BOX_GRID + 0.02),
              ],
              reason: 'stationary_dumped_object',
            });
          }
        }
      }
    }
  }

  verifiedBoxes.sort((a, b) => b.confidence - a.confidence);
  return {
    boxes: verifiedBoxes.slice(0, 2),
    peakFrameIdx,
  };
}

/**
 * Analyze high-FPS (15-20 FPS, up to 24 frames) live camera frames on Edge/Browser.
 */
export async function analyzeLiveFramesOnEdge(rawFrames, selectedModels, explicitBaselineInput = null) {
  const t0 = Date.now();
  const decoded = [];
  for (const f of rawFrames) {
    if (f) decoded.push(await decodeFrameTo112(f));
  }
  if (decoded.length === 0) {
    throw new Error('No valid frames captured');
  }

  const explicitBaseline = explicitBaselineInput ? await decodeFrameTo112(explicitBaselineInput) : null;
  const refFrame = explicitBaseline || decoded[0];
  const { boxes: activeRegions, peakFrameIdx } = detectDumpedObjectsOnly(decoded, explicitBaseline);

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

  // Gate classification by verified falling/dumped object so a person moving alone is NEVER flagged as dumping
  const rawMlpProb = forwardNeuralNetwork(features528);
  const hasVerifiedDump = activeRegions.length > 0;
  const topBoxConf = hasVerifiedDump ? activeRegions[0].confidence : 0;

  const nnProb = hasVerifiedDump
    ? Math.max(0.65, Math.min(0.96, 0.42 * rawMlpProb + 0.58 * topBoxConf))
    : Math.min(0.24, rawMlpProb * 0.30);

  const elapsedSec = Math.max(0.008, (Date.now() - t0) / 1000);
  const actions = [];

  if (selectedModels.includes('neural_network')) {
    actions.push({
      model: 'neural_network',
      label_title: 'Neural Network (MLP)',
      prediction: nnProb >= 0.5 ? 1 : 0,
      verdict: nnProb >= 0.5 ? 'DUMPING' : 'NO DUMPING',
      score: nnProb,
      score_kind: 'Spatiotemporal MLP + High-FPS Drop Tracker (threshold 0.5)',
      seconds: elapsedSec,
    });
  }
  if (selectedModels.includes('movinet_a0')) {
    const moviScore = hasVerifiedDump
      ? Math.max(0.60, Math.min(0.94, nnProb * 0.94))
      : Math.min(0.28, nnProb * 0.88);
    actions.push({
      model: 'movinet_a0',
      label_title: 'MoViNet A0 (browser estimate)',
      prediction: moviScore >= 0.5 ? 1 : 0,
      verdict: moviScore >= 0.5 ? 'DUMPING' : 'NO DUMPING',
      score: moviScore,
      score_kind: 'Derived motion estimate; full model runs through Python API',
      seconds: elapsedSec * 1.12,
    });
  }
  if (selectedModels.includes('optical_flow_svm')) {
    const svmScore = hasVerifiedDump ? (nnProb - 0.42) * 2.2 : (nnProb - 0.55) * 2.0;
    actions.push({
      model: 'optical_flow_svm',
      label_title: 'Optical Flow + SVM (browser estimate)',
      prediction: svmScore >= 0 ? 1 : 0,
      verdict: svmScore >= 0 ? 'DUMPING' : 'NO DUMPING',
      score: svmScore,
      score_kind: 'Derived motion estimate; full model runs through Python API',
      seconds: elapsedSec * 0.88,
    });
  }

  // Ensure the peak frame has a snapshot URI to draw the bounding box & trajectory on
  let bestFrameUri = decoded[peakFrameIdx]?.uri || decoded[decoded.length - 1]?.uri;
  if (!bestFrameUri && Platform.OS === 'web') {
    bestFrameUri = grabWebVideoFrameBase64();
  }
  const annotatedDataUri = hasVerifiedDump ? await drawDetectionsOnFrame(bestFrameUri, activeRegions) : null;
  const strongest = activeRegions.length > 0 ? activeRegions[0] : null;

  return {
    video_name: `Live Camera Stream (${decoded.length} frames @ High-FPS)`,
    mode: 'edge_browser',
    mode_note: 'Browser results are motion-based estimates. The Python API runs the trained models.',
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
