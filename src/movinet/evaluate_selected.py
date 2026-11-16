"""One report-only official-test evaluation after freezing the action model.

This script cannot select or train weights. Historical test exposure remains a
limitation even though this round's training and selection exclude test videos.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data import load_clip, load_manifest, split_records
from evaluate import evaluate_predictions
from src.movinet.main import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--final-test', action='store_true', required=True)
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    output = ROOT/'results/retraining_v2/movinet'
    selection_path = output/'frozen_selection.json'
    selection = json.loads(selection_path.read_text())
    checkpoint = ROOT/selection['checkpoint']
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != selection['sha256'] or selection['threshold'] != .5:
        raise ValueError('Frozen action weights or threshold changed')
    target = output/'final_test.json'
    predictions = output/'final_test_predictions.csv'
    if target.exists() or predictions.exists():
        raise FileExistsError('Final evaluation already exists; do not rerun or tune against it')
    _, _, records = split_records(load_manifest())
    saved = torch.load(checkpoint, map_location='cpu', weights_only=True)
    architecture = saved['architecture']
    model = build_model(tf_like=architecture['tf_like']).to(args.device)
    model.load_state_dict(saved['state_dict']); model.eval()
    torch.set_num_threads(4)
    rows = []
    with torch.inference_mode():
        for record in records:
            clip = load_clip(record)
            tensor = torch.from_numpy(clip).permute(0, 3, 1, 2).float().div_(255).unsqueeze(0).to(args.device)
            started = time.monotonic()
            score = float(torch.sigmoid(model(tensor)).reshape(-1)[0].cpu())
            rows.append(dict(video_id=record['video_id'], label=record['label'], prediction=int(score >= .5),
                             probability=score, inference_seconds=time.monotonic()-started))
            if len(rows) % 20 == 0:
                print('Final action evaluation', len(rows), '/', len(records), flush=True)
    with predictions.open('x', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    report = dict(evaluation_split='official_test', videos=len(rows),
                  frozen_selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),
                  checkpoint_sha256=selection['sha256'], threshold=.5, device=args.device,
                  metrics=evaluate_predictions([r['label'] for r in rows], [r['prediction'] for r in rows]),
                  inference_total_seconds=sum(r['inference_seconds'] for r in rows),
                  test_used_for_training=False, test_used_for_selection=False,
                  limitation='Report-only evaluation after model selection. Official test clips were exposed during historical project development, so this is not a pristine independent benchmark. No indoor live-camera accuracy is established.')
    target.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
