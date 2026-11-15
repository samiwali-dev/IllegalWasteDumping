import { Platform } from 'react-native';
import nnWeights from './nnWeights.json';
import { detectDropMotion } from './liveMotion';

const IMAGE_SIZE = 112;

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
    const targetH = Math.max(200, Math.round((activeVideo.videoHeight / activeVideo.videoWidth) * targetW));
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
        reject(new Error('تعذر قراءة إطار الكاميرا. أعد الالتقاط.'));
      };
      img.src = dataUri.startsWith('data:') ? dataUri : `data:image/jpeg;base64,${dataUri}`;
    });
  }

  throw new Error('تحليل اللايف على تطبيق الجوال يحتاج اتصالًا بخادم Python. يمكن استخدام كاميرا المتصفح للتحليل المحلي.');
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

        const label = 'Possible fall - review';
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
 * Browser fallback screens temporal motion only. Trained Python models are not
 * emulated here; a possible fall needs human review and is not proof of dumping.
 */
export async function analyzeLiveFramesOnEdge(rawFrames, _selectedModels, _explicitBaselineInput = null) {
  const started = Date.now();
  const decoded = [];
  for (const frame of rawFrames) {
    if (frame) decoded.push(await decodeFrameTo112(frame));
  }
  if (!decoded.length) throw new Error('No valid frames captured');
  const { boxes, peakFrameIdx } = detectDropMotion(decoded);
  const detected = boxes.length > 0;
  const snapshot = decoded[peakFrameIdx]?.uri;
  const annotated = detected ? await drawDetectionsOnFrame(snapshot, boxes) : null;
  return {
    video_name: `Live Camera Stream (${decoded.length} frames)`,
    mode: 'edge_browser',
    mode_note: 'فحص حركة محلي: النتيجة اشتباه سقوط يحتاج مراجعة، وليست تشغيلًا لنماذج MoViNet أو SVM أو YOLO.',
    event_detected: detected,
    actions: [{
      model: 'edge_motion',
      label_title: 'فحص مسار الحركة المحلي',
      prediction: detected ? 1 : 0,
      verdict: detected ? 'POSSIBLE_FALL' : 'NO_EVENT',
      score: null,
      score_kind: 'فحص حركة للأسفل عبر عدة إطارات',
      seconds: (Date.now() - started) / 1000,
    }],
    yolo_live: {
      detector: 'temporal_motion',
      total_detections: boxes.length,
      detections: boxes,
      strongest_detection: boxes[0] || null,
      annotated_data_uri: annotated,
      frames_buffered: decoded.length,
    },
    errors: [],
  };
}
