"""MoViNet-A0: train, evaluate, or predict from one entry point.

Uses the existing project's prepared frames, split manifest and MoViNet-pytorch
library (Atze00, commit c2d1edf48fc6c5259707f9d833f22171b4f63493).
  python movinet_a0_short.py evaluate
  python movinet_a0_short.py predict video.mp4
  python movinet_a0_short.py train --epochs 50
Pass --artifacts for a relocated project. Training saves compact_trained.pt.
"""
import argparse
import json
from pathlib import Path
import random
import sys

import cv2
import numpy as np
import torch
from sklearn import metrics
from torch import nn
from torch.utils.data import DataLoader, Dataset

CLASSES = ('No Dumping', 'Dumping')
DEFAULT_ARTIFACTS = Path.home() / 'Documents/Codex/2026-11-01/referenced-chatgpt-conversation-this-is-an/outputs/movinet_project/artifacts'

def model():
    # Same vendored A0 implementation used to train best.pt.
    from movinets import MoViNet
    from movinets.config import _C

    return MoViNet(_C.MODEL.MoViNetA0, causal=False, pretrained=False, num_classes=2)

def tensor(images):
    return torch.from_numpy(np.ascontiguousarray(images.transpose(3, 0, 1, 2))).float().div_(255)

def video_frames(path, sequential=False):
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 24 or cap.get(cv2.CAP_PROP_FPS) <= 0:
            raise ValueError(f'Invalid video: {path}')
        if sequential:
            actual = 0
            while cap.grab():
                actual += 1
            if actual < 24 or abs(actual - count) > max(2, count * .01):
                raise ValueError(f'Frame count mismatch: {path}')
            count = actual
            cap.release()
            cap.open(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
        images, position = [], -1
        for index in np.linspace(0, count - 1, 24).round().astype(int):
            if sequential:
                for position in range(position + 1, index + 1):
                    if not cap.grab():
                        raise ValueError(f'Cannot decode frame {position}: {path}')
                ok, image = cap.retrieve()
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, image = cap.read()
            if not ok:
                if sequential:
                    raise ValueError(f'Cannot decode frame {index}: {path}')
                cap.release()
                return video_frames(path, sequential=True)
            h, w = image.shape[:2]
            scale = 172 / max(h, w)
            image = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
            h, w = image.shape[:2]
            top, left = (172 - h) // 2, (172 - w) // 2
            image = cv2.copyMakeBorder(image, top, 172 - h - top, left, 172 - w - left, cv2.BORDER_CONSTANT)
            images.append(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return np.stack(images)
    finally:
        cap.release()

class Clips(Dataset):
    def __init__(self, rows, split):
        self.rows = [r for r in rows if r['split'] == split]
        self.augment = split == 'train'

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        images = np.load(row['cache'])
        if self.augment:
            if random.random() < .5:
                images = images[:, :, ::-1, :]
            images = np.clip(images.astype(np.float32) * random.uniform(.85, 1.15), 0, 255).astype(np.uint8)
        return tensor(images), row['label']

def evaluate(net, loader, criterion, device):
    net.eval()
    labels, probabilities, loss = [], [], 0
    with torch.no_grad(), torch.amp.autocast(device.type, enabled=device.type == 'cuda'):
        for x, y in loader:
            logits = net(x.to(device))
            loss += criterion(logits, y.to(device)).item() * len(y)
            labels.extend(y.tolist())
            probabilities.extend(logits.float().softmax(1).cpu().tolist())
    return labels, np.asarray(probabilities), loss / len(labels)

def train(rows, artifacts, device, epochs):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)
    torch.backends.cudnn.deterministic = True
    train_data = Clips(rows, 'train')
    train_loader = DataLoader(train_data, batch_size=4, shuffle=True, pin_memory=True)
    valid_loader = DataLoader(Clips(rows, 'valid'), batch_size=4, pin_memory=True)
    net = model().to(device)
    counts = np.bincount([r['label'] for r in train_data.rows], minlength=2)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32, device=device))
    optimizer = torch.optim.AdamW(net.parameters(), lr=.001, weight_decay=.01)
    schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = torch.amp.GradScaler('cuda')
    best, stale = (-1, float('-inf')), 0
    checkpoint = artifacts / 'compact_trained.pt'
    for epoch in range(1, epochs + 1):
        net.train()
        for x, y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                loss = criterion(net(x.to(device)), y.to(device))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(net.parameters(), 5)
            scaler.step(optimizer)
            scaler.update()
        y, p, loss = evaluate(net, valid_loader, criterion, device)
        score = (metrics.f1_score(y, p.argmax(1), average='macro', labels=[0, 1]), -loss)
        print(f'Epoch {epoch}: Validation Macro F1 {score[0]:.1%}', flush=True)
        if score > best:
            best, stale = score, 0
            torch.save({'model': net.state_dict(), 'epoch': epoch}, checkpoint)
        else:
            stale += 1
        schedule.step()
        if stale == 12:
            break
    return checkpoint

def main():
    parser = argparse.ArgumentParser(description='Train, evaluate or predict with MoViNet-A0.')
    parser.add_argument('mode', choices=('train', 'evaluate', 'predict'))
    parser.add_argument('video', nargs='?')
    parser.add_argument('--artifacts', type=Path, default=DEFAULT_ARTIFACTS)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--epochs', type=int, default=50)
    args = parser.parse_args()
    if (args.artifacts.parent / 'movinets').is_dir():
        sys.path.insert(0, str(args.artifacts.parent.resolve()))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = args.checkpoint or args.artifacts / 'best.pt'
    if args.mode == 'train':
        if device.type != 'cuda' or args.epochs < 1:
            parser.error('Training requires CUDA and a positive number of epochs.')
        rows = json.loads((args.artifacts / 'manifest.json').read_text(encoding='utf-8'))
        checkpoint = train(rows, args.artifacts, device, args.epochs)
    net = model().to(device).eval()
    net.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)['model'])
    if args.mode == 'predict':
        if not args.video:
            parser.error('Provide an MP4 path after predict.')
        with torch.no_grad():
            p = net(tensor(video_frames(args.video))[None].to(device)).softmax(1)[0]
        print(json.dumps({'prediction': CLASSES[int(p.argmax())], 'p_dumping': float(p[1])}))
        return
    rows = json.loads((args.artifacts / 'manifest.json').read_text(encoding='utf-8'))
    loader = DataLoader(Clips(rows, 'test'), batch_size=4)
    y, p, _ = evaluate(net, loader, nn.CrossEntropyLoss(), device)
    predicted = p.argmax(1)
    report = metrics.classification_report(y, predicted, labels=[0, 1], target_names=CLASSES,
                                           output_dict=True, zero_division=0)
    print(f'Accuracy: {report["accuracy"]:.1%} | Macro F1: {report["macro avg"]["f1-score"]:.1%}')
    for label in CLASSES:
        scores = report[label]
        print(f'{label}: Precision {scores["precision"]:.1%}, Recall {scores["recall"]:.1%}, F1 {scores["f1-score"]:.1%}')
    print(f'ROC-AUC: {metrics.roc_auc_score(y, p[:, 1]):.1%} | PR-AUC: {metrics.average_precision_score(y, p[:, 1]):.1%}')
    print('Confusion Matrix:', metrics.confusion_matrix(y, predicted, labels=[0, 1]).tolist())

if __name__ == '__main__':
    main()
