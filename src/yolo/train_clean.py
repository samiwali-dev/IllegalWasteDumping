"""Compare fresh COCO transfer and random initialization on development data only."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training_protocol import require_development_video


def validate_data(manifest, data):
    """Verify the files actually consumed by YOLO against their provenance."""
    import yaml
    config = yaml.safe_load(data.read_text())
    if 'test' in config or config.get('names') != {0: 'litter_object'}:
        raise ValueError('Development YAML must contain one object class and no test split')
    base = Path(config['path'])
    records = {str((ROOT / row['file']).resolve()): row for row in manifest['records']}
    if len(records) != len(manifest['records']):
        raise ValueError('Duplicate object image paths')
    image_hashes = {'fit': set(), 'validation': set()}
    for split, role in [('train', 'fit'), ('val', 'validation')]:
        files = (base / config[split]).read_text().splitlines()
        if len(files) != manifest['exposures'][split]:
            raise ValueError('Object sampling list changed')
        expected = {key for key, row in records.items() if row['role'] == role}
        if {str(Path(name).resolve()) for name in files} != expected:
            raise ValueError('Image list disagrees with its declared development split')
        for name in expected:
            row = records[name]
            if row.get('source_video'):
                require_development_video(row['source_video'], role)
            image = Path(name)
            label = base / 'labels' / split / (image.stem + '.txt')
            if hashlib.sha256(image.read_bytes()).hexdigest() != row['sha256']:
                raise ValueError(f'Object image changed: {image.name}')
            if hashlib.sha256(label.read_bytes()).hexdigest() != row['label_sha256']:
                raise ValueError(f'Object label changed: {image.name}')
            image_hashes[role].add(row['sha256'])
    if image_hashes['fit'] & image_hashes['validation']:
        raise ValueError('Object image content overlaps development splits')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initialization', choices=['transfer', 'scratch'], required=True)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()
    output = ROOT/'results/retraining_v2/yolo'
    manifest = json.loads((output/'dataset.json').read_text())
    for row in manifest['records']:
        if row.get('source_video'):
            require_development_video(row['source_video'], row['role'])
        if row['role'] not in ('fit', 'validation'):
            raise ValueError('Held-out data in development dataset')
    data = ROOT/'datasets/release_v2/objects/data.yaml'
    validate_data(manifest, data)
    run = output/args.initialization
    if run.exists():
        raise FileExistsError(run)
    source = ROOT/'src/yolo/actors.pt'
    if hashlib.sha256(source.read_bytes()).hexdigest() != 'f59b3d833e2ff32e194b5bb8e08d211dc7c5bdf144b90d2c8412c47ccfc83b36':
        raise ValueError('Official COCO source checkpoint changed')
    configuration = dict(initialization=args.initialization, epochs=args.epochs, imgsz=640, batch=8,
                         optimizer='AdamW', lr=.001, patience=6, seed=42, device='mps',
                         data_manifest_sha256=hashlib.sha256((output/'dataset.json').read_bytes()).hexdigest(),
                         test_used_for_training=False, test_used_for_selection=False)
    other = output / ('scratch_configuration.json' if args.initialization == 'transfer' else 'transfer_configuration.json')
    if other.exists() and json.loads(other.read_text())['data_manifest_sha256'] != configuration['data_manifest_sha256']:
        raise ValueError('Initialization comparison must use the same frozen dataset')
    (output/f'{args.initialization}_configuration.json').write_text(json.dumps(configuration, indent=2)+'\n')
    model = YOLO(str(source) if args.initialization == 'transfer' else 'yolov8n.yaml')
    started = time.monotonic()
    model.train(data=str(data), epochs=args.epochs, imgsz=640, batch=8, workers=0, device='mps',
                optimizer='AdamW', lr0=.001, patience=6, seed=42, pretrained=args.initialization == 'transfer',
                project=str(output), name=args.initialization, exist_ok=False, amp=False,
                cache=False, close_mosaic=5, plots=False, save_period=5)
    configuration.update(training_seconds=time.monotonic()-started, best_checkpoint=str(model.trainer.best))
    (output/f'{args.initialization}_configuration.json').write_text(json.dumps(configuration, indent=2)+'\n')


if __name__ == '__main__':
    main()
