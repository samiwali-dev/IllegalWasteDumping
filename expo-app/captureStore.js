// Captures are independent of the latest prediction, so a clear frame cannot erase evidence.
function frameUri(frame) {
  return typeof frame === 'string' ? frame : frame?.uri;
}

function captureFromResult(result, frames = [], now = Date.now()) {
  const detected = (result?.yolo_live?.total_detections || 0) > 0 ||
    result?.actions?.some(action => action.prediction === 1);
  if (!detected) return null;
  const uri = result?.yolo_live?.annotated_data_uri || [...frames].reverse().map(frameUri).find(Boolean);
  if (!uri) return null;
  return {
    id: String(now), uri, capturedAt: now,
    kind: result.mode === 'edge_browser' ? 'Motion alert' : 'Model alert',
    regions: result?.yolo_live?.total_detections || 0,
  };
}

function retainCaptures(previous, capture) {
  if (!capture?.uri) return previous;
  return [capture, ...previous.filter(item => item.id !== capture.id)].slice(0, 6);
}

function captureDetectionsForFall(result, motion, now = Date.now()) {
  if (result?.yolo_live?.task === 'temporal_release_candidates') {
    return (result.yolo_live.detections || []).filter(d => d.reason === 'released_object' &&
      d.trail?.length >= 3 && Number.isFinite(d.source_actor?.confidence));
  }
  if (!motion?.at || now < motion.at || now - motion.at > 1500) return [];
  return (result?.yolo_live?.detections || []).flatMap(detection => {
    const track = motion.detections?.find(candidate => {
      if (candidate.reason !== 'falling_object' || !(candidate.trail?.length >= 3)) return false;
      const a = detection.norm_box, b = candidate.norm_box;
      if (!a || !b) return false;
      const intersection = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])) *
        Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1]));
      const union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection;
      return union > 0 && intersection / union >= 0.1;
    });
    return track ? [{ ...detection, trail: track.trail, source: 'yolov8n' }] : [];
  });
}

module.exports = { frameUri, captureFromResult, retainCaptures, captureDetectionsForFall };
