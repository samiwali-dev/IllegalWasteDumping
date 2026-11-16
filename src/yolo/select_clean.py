"""Compare object checkpoints and select a threshold on validation images only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from src.yolo.metrics import counts
from src.yolo.train_clean import validate_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', action='append', required=True, help='NAME=checkpoint.pt')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--imgsz', type=int, choices=[640, 1280], default=640)
    args = parser.parse_args()
    output = ROOT / 'results/retraining_v2/yolo'
    data = ROOT / 'datasets/release_v2/objects/data.yaml'
    manifest_path = output / 'dataset.json'
    manifest = json.loads(manifest_path.read_text())
    validate_data(manifest, data)
    records = {Path(r['file']).name: r for r in manifest['records'] if r['role'] == 'validation'}
    report = dict(protocol='official_train_development_only_v2',
                  data_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                  validation_images=len(records), inference=dict(imgsz=args.imgsz, rect=False),
                  policy='Maximize validation F0.5 times background-image rejection, with object recall >= 0.35; confidence grid 0.10..0.80, step 0.05. Compare fresh transfer, fresh scratch and unchanged legacy head.',
                  test_used_for_training=False, test_used_for_selection=False, candidates={})
    for argument in args.candidate:
        name, separator, checkpoint = argument.partition('=')
        if not separator or not name or name in report['candidates']:
            raise ValueError('Candidates must have unique NAME=checkpoint.pt arguments')
        path = Path(checkpoint).resolve()
        model = YOLO(str(path))
        rows = []
        for result in model.predict(source=str(data.parent/'val.txt'), imgsz=args.imgsz, rect=False,
                                    conf=.05, device=args.device, batch=2, stream=True, verbose=False):
            image = Path(result.path)
            if image.name not in records:
                raise ValueError('Unexpected non-validation image')
            truth = []
            for line in (data.parent/'labels/val'/f'{image.stem}.txt').read_text().splitlines():
                category, cx, cy, w, h = map(float, line.split())
                if category != 0:
                    raise ValueError('Unexpected class')
                truth.append([cx-w/2, cy-h/2, cx+w/2, cy+h/2])
            rows.append(dict(file=image.name, kind=records[image.name]['kind'], truth=truth,
                             boxes=result.boxes.xyxyn.cpu().tolist(), scores=result.boxes.conf.cpu().tolist()))
        if len(rows) != len(records) or len({r['file'] for r in rows}) != len(records):
            raise ValueError('Validation coverage changed')
        (output/f'{name}_{args.imgsz}_validation_predictions.json').write_text(json.dumps(rows)+'\n')
        points = [counts(rows, step/100) for step in range(10, 81, 5)]
        eligible = [p for p in points if p['recall'] >= .35]
        selected = max(eligible, key=lambda p: (p['selection_score'], p['recall'])) if eligible else None
        metrics = model.val(data=str(data), split='val', imgsz=args.imgsz, rect=False, batch=2,
                            device=args.device, workers=0, conf=.001, plots=False,
                            project=str(output/'validation_runs'), name=f'{name}_{args.imgsz}', verbose=False)
        item = dict(checkpoint=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    map50=float(metrics.box.map50), map50_95=float(metrics.box.map),
                    operating_points=points, selected=selected)
        if selected:
            item['by_source_at_selected_threshold'] = {
                kind: counts([r for r in rows if r['kind'] == kind], selected['threshold'])
                for kind in sorted({r['kind'] for r in rows})}
        report['candidates'][name] = item
        (output/f'selection_{args.imgsz}.json').write_text(json.dumps(report, indent=2)+'\n')
        print(name, json.dumps({k: item[k] for k in ('map50', 'map50_95', 'selected')}), flush=True)
    eligible = [(name, r) for name, r in report['candidates'].items() if r['selected']]
    if not eligible:
        raise RuntimeError('No checkpoint meets validation recall requirement; no model selected')
    report['selected_candidate'] = max(eligible, key=lambda pair: (pair[1]['selected']['selection_score'], pair[1]['selected']['recall']))[0]
    report['limitation'] = 'Object localization on development images; not throwing accuracy. MIVIA ROI annotations are a small assistant-reviewed pilot. Historical legacy weights have test exposure.'
    (output/f'selection_{args.imgsz}.json').write_text(json.dumps(report, indent=2)+'\n')
    print('Selected:', report['selected_candidate'], flush=True)


if __name__ == '__main__':
    main()
