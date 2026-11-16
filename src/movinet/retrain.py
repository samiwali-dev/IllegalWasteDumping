"""Fresh initialization comparison using only official-train fit/validation clips.

Never runs test inference or replaces the active model.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys
import time

import cv2
import numpy as np
import torch
from torch import nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data import load_clip
from training_protocol import development_records, require_development_video

DATA = ROOT/'datasets/release_v2'
RESULTS = ROOT/'results/retraining_v2/movinet'


def build_model(initialization, source=None):
    from movinets import MoViNet
    from movinets.config import _C

    class ActionModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.network = MoViNet(_C.MODEL.MoViNetA0, causal=False, pretrained=False,
                                   num_classes=1, conv_type='3d', tf_like=True)

        def forward(self, clips):
            return self.network(clips.permute(0, 2, 1, 3, 4).contiguous()).reshape(-1)

    model = ActionModel()
    if initialization == 'transfer':
        state = torch.load(source, map_location='cpu', weights_only=True)
        target = model.network.state_dict()
        compatible = {k: value for k, value in state.items() if k in target and value.shape == target[k].shape}
        missing, extra = model.network.load_state_dict(compatible, strict=False)
        if extra or any(not key.startswith('classifier.3.') for key in missing):
            raise ValueError(f'Unexpected pretrained checkpoint keys: {missing}, {extra}')
        print('Pretrained backbone tensors:', len(compatible), '; new binary head:', missing, flush=True)
    return model


def cache_clips():
    fit, validation = development_records()
    for role, rows in [('fit', fit), ('validation', validation)]:
        for row in rows:
            require_development_video(row['video_id'], role)
    ids = {role: [r['video_id'] for r in rows] for role, rows in [('fit', fit), ('validation', validation)]}
    path = DATA/'action_train_validation.npz'
    if path.is_file():
        cache = np.load(path, allow_pickle=False)
        if cache['fit_ids'].tolist() != ids['fit'] or cache['validation_ids'].tolist() != ids['validation']:
            raise ValueError('Cached split identity changed')
        return cache['fit'], cache['validation'], np.array([r['label'] for r in fit]), np.array([r['label'] for r in validation]), ids
    arrays = {}
    cv2.setNumThreads(1)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for role, rows in [('fit', fit), ('validation', validation)]:
            images = []
            for i, clip in enumerate(pool.map(load_clip, rows), 1):
                images.append(clip)
                if i % 40 == 0:
                    print('Cached', role, i, '/', len(rows), flush=True)
            arrays[role] = np.stack(images)
    np.savez(path, **arrays, fit_ids=np.array(ids['fit']), validation_ids=np.array(ids['validation']))
    return arrays['fit'], arrays['validation'], np.array([r['label'] for r in fit]), np.array([r['label'] for r in validation]), ids


def metrics(labels, scores, threshold=.5):
    predicted = np.array(scores) >= threshold
    return dict(accuracy=float(accuracy_score(labels, predicted)),
                macro_f1=float(f1_score(labels, predicted, average='macro', zero_division=0)),
                precision=float(precision_score(labels, predicted, zero_division=0)),
                recall=float(recall_score(labels, predicted, zero_division=0)),
                confusion_matrix=confusion_matrix(labels, predicted, labels=[0, 1]).tolist())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--initialization', choices=['scratch', 'transfer'], required=True)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--patience', type=int, default=4)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--device', default='mps')
    args = parser.parse_args()
    output = RESULTS/args.initialization
    if output.exists():
        raise FileExistsError(f'Run already exists: {output}')
    output.mkdir(parents=True)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    torch.set_num_threads(4)
    x_train, x_validation, y_train, y_validation, ids = cache_clips()
    source = DATA/'source/movinet_a0_kinetics600.pt'
    if args.initialization == 'transfer':
        if not source.is_file():
            source.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run(['curl', '--fail', '--location', '--silent', '--show-error', '--max-time', '120',
                            'https://raw.githubusercontent.com/Atze00/MoViNet-pytorch/c2d1edf48fc6c5259707f9d833f22171b4f63493/weights/modelA0_statedict_v3',
                            '--output', str(source)], check=True)
        if hashlib.sha256(source.read_bytes()).hexdigest() != '7fa4d3b5a1c3970d581476782f8892c24d46c10433b7bd8ea4fc4ccdc36224e1':
            raise ValueError('Pretrained checkpoint checksum changed')
    model = build_model(args.initialization, source).to(args.device)
    lr = .0002 if args.initialization == 'transfer' else .0008
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.001)
    loss_fn = nn.BCEWithLogitsLoss()
    report = dict(initialization=args.initialization, source_sha256=hashlib.sha256(source.read_bytes()).hexdigest() if args.initialization == 'transfer' else None,
                  protocol='fit320_validation80_only_v2', split_ids=ids,
                  architecture=dict(name='MoViNet A0', tf_like=True, causal=False, num_classes=1),
                  hyperparameters=dict(epochs=args.epochs, patience=args.patience, batch=args.batch, lr=lr,
                                       weight_decay=.001, input_size=112, frames=24, seed=42,
                                       augmentations=['whole-frame horizontal flip', 'brightness gain 0.9..1.1'],
                                       transfer_batchnorm='frozen pretrained statistics', transfer_head_warmup_epochs=2),
                  test_used_for_training=False, test_used_for_selection=False, history=[])
    best_key = (-1, -1); stale = 0; started = time.monotonic()
    for epoch in range(1, args.epochs+1):
        model.train()
        if args.initialization == 'transfer':
            for name, parameter in model.named_parameters():
                parameter.requires_grad_(epoch > 2 or '.classifier.' in name)
            for layer in model.modules():
                if isinstance(layer, nn.BatchNorm3d):
                    layer.eval()
        order = np.random.default_rng(42+epoch).permutation(len(x_train))
        total_loss = 0
        for start in range(0, len(order), args.batch):
            indices = order[start:start+args.batch]
            clips = torch.from_numpy(x_train[indices].copy()).permute(0, 1, 4, 2, 3).float().div_(255).to(args.device)
            if random.random() < .5:
                clips = clips.flip(-1)
            clips = (clips * random.uniform(.9, 1.1)).clamp(0, 1)
            labels = torch.tensor(y_train[indices], dtype=torch.float32, device=args.device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(clips), labels)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step()
            total_loss += loss.item()*len(indices)
            if (start//args.batch+1) % 20 == 0:
                print(f'epoch {epoch} batch {start//args.batch+1}/{len(order)//args.batch}', flush=True)
        model.eval(); scores = []
        with torch.inference_mode():
            for start in range(0, len(x_validation), args.batch):
                clips = torch.from_numpy(x_validation[start:start+args.batch].copy()).permute(0, 1, 4, 2, 3).float().div_(255).to(args.device)
                scores.extend(torch.sigmoid(model(clips)).cpu().tolist())
        validation = metrics(y_validation, scores)
        key = (validation['macro_f1'], validation['precision'])
        row = dict(epoch=epoch, train_loss=total_loss/len(x_train), validation=validation,
                   elapsed_seconds=round(time.monotonic()-started, 2))
        report['history'].append(row)
        if key > best_key:
            best_key = key; stale = 0
            state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(dict(state_dict=state, architecture=report['architecture'], protocol=report['protocol']), output/'best.pt')
            report.update(best_epoch=epoch, validation=validation, validation_scores=scores)
        else:
            stale += 1
        (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')
        print(json.dumps(row), flush=True)
        if stale >= args.patience:
            break
    report['training_seconds'] = time.monotonic()-started
    (output/'report.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
