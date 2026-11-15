import React, { memo, useState } from 'react';
import { StyleSheet, View } from 'react-native';
import { motionTrailGeometry } from './motionGeometry';

function segments(points) {
  return points.slice(1).map((end, i) => {
    const start = points[i];
    const length = Math.hypot(end.x - start.x, end.y - start.y);
    return <View key={i} style={[styles.line, {
      left: (start.x + end.x - length) / 2, top: (start.y + end.y) / 2 - 1.5,
      width: length, transform: [{ rotate: `${Math.atan2(end.y - start.y, end.x - start.x)}rad` }],
    }]} />;
  });
}

function MotionTrail({ trail, mirrored }) {
  const [size, setSize] = useState({ width: 0, height: 0 });
  const geometry = motionTrailGeometry(trail, size.width, size.height, mirrored);
  return <View pointerEvents="none" testID="motion-trail" style={StyleSheet.absoluteFill}
    onLayout={({ nativeEvent }) => setSize({ width: nativeEvent.layout.width, height: nativeEvent.layout.height })}>
    {segments(geometry.points)}
    {geometry.arrow.length > 0 && <View testID="motion-arrowhead" style={StyleSheet.absoluteFill}>{segments(geometry.arrow)}</View>}
  </View>;
}

const styles = StyleSheet.create({
  line: { position: 'absolute', height: 3, borderRadius: 1.5, backgroundColor: '#F59E0B' },
});

export default memo(MotionTrail);
