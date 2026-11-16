"""Evaluate development clips; official test videos cannot be selected here."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.yolo.main import detect_video
from training_protocol import development_records, require_development_video
from evaluate import evaluate_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--split', choices=['fit', 'validation'], default='validation')
    parser.add_argument('--clips', nargs='+')
    parser.add_argument('--model', type=Path)
    parser.add_argument('--confidence', type=float, default=.35)
    parser.add_argument('--strategy', choices=['release_v1', 'object_tracks_v2'], default='object_tracks_v2')
    parser.add_argument('--name', default='development', help='A new directory name for this evaluation')
    args = parser.parse_args()
    fit, validation = development_records()
    records = fit if args.split == 'fit' else validation
    selected = args.clips or [r['video_id'] for r in records]
    if len(set(selected)) != len(selected):
        raise ValueError('Duplicate evaluation clips')
    # Validate the entire request before decoding any source video.
    for video_id in selected:
        require_development_video(video_id, args.split)
    if Path(args.name).name != args.name or args.name in ('.', '..'):
        raise ValueError('Evaluation name must be one directory name')
    output = ROOT/'results/yolo/release_v2'/args.split/args.name
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = args.model or ROOT/'src/yolo/model.pt'
    summary = dict(scope='Development evaluation only; official test videos excluded.', split=args.split,
                   expected_videos=len(selected), strategy=args.strategy, confidence_threshold=args.confidence,
                   checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                   code_sha256={name: hashlib.sha256((ROOT/'src/yolo'/name).read_bytes()).hexdigest()
                                for name in ('main.py', 'release_pipeline.py', 'item_tracking.py', 'temporal.py')},
                   metric_note='Candidate presence compared with video labels. This does not measure correct object localization, and is not an independent test or live-camera accuracy estimate.',
                   test_used_for_selection=False, videos=[])
    labels = {r['video_id']: r['label'] for r in records}
    for video_id in selected:
        name = Path(video_id).stem
        started = time.monotonic()
        report = detect_video(ROOT/'datasets/MIVIA-IWDD 500/train/videos'/video_id,
                              device=args.device, output_dir=output/'frames', conf=args.confidence,
                              tracking_strategy=args.strategy,
                              **({'model_path': args.model} if args.model else {}))
        row = dict(video=video_id, label=labels[video_id], count=len(report['events']), frames_scanned=report['frames_scanned'],
                   seconds=round(time.monotonic()-started, 2), events=report['events'])
        summary['videos'].append(row)
        summary['candidate_presence_metrics'] = evaluate_predictions(
            [r['label'] for r in summary['videos']], [int(r['count'] > 0) for r in summary['videos']])
        summary['completed_videos'] = len(summary['videos'])
        (output/f'{name}_report.json').write_text(json.dumps(report, indent=2)+'\n')
        (output/'video_checks.json').write_text(json.dumps(summary, indent=2)+'\n')
        print(name, 'candidates', row['count'], 'seconds', row['seconds'], flush=True)


if __name__ == '__main__':
    main()
