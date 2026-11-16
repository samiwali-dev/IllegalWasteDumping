// Shared by the live overlay and the downloaded photo.
function motionTrailGeometry(trail, width, height, mirrored = false) {
  const points = [];
  for (const point of trail || []) {
    if (!Number.isFinite(point?.cx) || !Number.isFinite(point?.cy)) continue;
    const x = Math.max(0, Math.min(1, point.cx));
    const next = { x: (mirrored ? 1 - x : x) * width, y: Math.max(0, Math.min(1, point.cy)) * height };
    const previous = points[points.length - 1];
    if (!previous || Math.hypot(next.x - previous.x, next.y - previous.y) > 0.5) points.push(next);
  }
  if (points.length < 2) return { points, arrow: [] };
  const tip = points[points.length - 1];
  const previous = points[points.length - 2];
  const angle = Math.atan2(tip.y - previous.y, tip.x - previous.x);
  const length = Math.max(7, Math.min(16, Math.min(width, height) * 0.035));
  const arm = offset => ({ x: tip.x - length * Math.cos(angle + offset), y: tip.y - length * Math.sin(angle + offset) });
  return { points, arrow: [arm(-Math.PI / 6), tip, arm(Math.PI / 6)] };
}

function yoloDetectionLabel(detection) {
  if (detection?.source === 'motion_tracker') return 'Release candidate';
  const score = detection?.confidence;
  const name = detection?.class_name === 'litter_object' ? 'item' : detection?.class_name || 'Object';
  return `YOLO · ${name}${Number.isFinite(score) ? ` ${Math.round(score * 100)}%` : ''}`;
}

module.exports = { motionTrailGeometry, yoloDetectionLabel };
