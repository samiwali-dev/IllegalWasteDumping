"""Reproduce development checks on existing clips, not a blind event benchmark."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.yolo.main import detect_video

CLIPS = ('vid0012', 'vid0005', 'vid0018', 'vid0021', 'vid0038', 'vid0040', 'vid0043')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--clips', nargs='+', default=list(CLIPS))
    args = parser.parse_args()
    output = ROOT/'results/yolo/release_v1'
    output.mkdir(parents=True, exist_ok=True)
    summary = dict(scope='Inspected development clips reused during implementation; not a blind accuracy benchmark.',
                   positive_case='vid0012: visually reviewed thrown-object region, not dense object ground truth.', videos=[])
    for name in args.clips:
        started = time.monotonic()
        report = detect_video(ROOT/'datasets/MIVIA-IWDD 500/test/videos'/f'{name}.mp4',
                              device=args.device, output_dir=output/'frames')
        row = dict(video=name, count=len(report['events']), frames_scanned=report['frames_scanned'],
                   seconds=round(time.monotonic()-started, 2), events=report['events'])
        if name == 'vid0012':
            centers = [((e['bbox_xyxy'][0]+e['bbox_xyxy'][2])/2,
                        (e['bbox_xyxy'][1]+e['bbox_xyxy'][3])/2) for e in report['events']]
            row['target_region_hits'] = sum(890 <= x <= 950 and 390 <= y <= 450 for x, y in centers)
            row['static_blue_cover_hits'] = sum(1290 <= x <= 1690 and 330 <= y <= 800 for x, y in centers)
        summary['videos'].append(row)
        (output/f'{name}_report.json').write_text(json.dumps(report, indent=2)+'\n')
        (output/'video_checks.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(name, 'candidates', row['count'], 'seconds', row['seconds'], flush=True)


if __name__ == '__main__':
    main()
