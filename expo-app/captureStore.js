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

module.exports = { frameUri, captureFromResult, retainCaptures };
