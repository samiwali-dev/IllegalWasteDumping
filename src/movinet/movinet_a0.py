"""MoViNet-A0 video classification.

Architecture adapted from Atze00/MoViNet-pytorch, commit
c2d1edf48fc6c5259707f9d833f22171b4f63493.

MIT License

Copyright (c) 2021 Atze00

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
import argparse
import csv
import hashlib
import json
import random
import time
from pathlib import Path
import cv2
import numpy as np

from collections import OrderedDict
import torch
from torch import nn
from torch.nn import functional as F


# Only the non-streaming A0 architecture used by the saved checkpoint.
# Each row: input, output, expansion, kernel, stride, convolution padding, pool padding.
A0_BLOCKS = (
    ((8, 8, 24, (1, 5, 5), (1, 2, 2), (0, 2, 2), (0, 1, 1)),),
    (
        (8, 32, 80, (3, 3, 3), (1, 2, 2), (1, 0, 0), (0, 0, 0)),
        *((32, 32, 80, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1)),) * 2,
    ),
    (
        (32, 56, 184, (5, 3, 3), (1, 2, 2), (2, 0, 0), (0, 0, 0)),
        (56, 56, 112, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1)),
        (56, 56, 184, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1)),
    ),
    (
        (56, 56, 184, (5, 3, 3), (1, 1, 1), (2, 1, 1), (0, 1, 1)),
        *((56, 56, 184, (3, 3, 3), (1, 1, 1), (1, 1, 1), (0, 1, 1)),) * 3,
    ),
    (
        (56, 104, 384, (5, 3, 3), (1, 2, 2), (2, 1, 1), (0, 1, 1)),
        *((104, 104, 280, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1)),) * 2,
        (104, 104, 344, (1, 5, 5), (1, 1, 1), (0, 2, 2), (0, 1, 1)),
    ),
)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class ConvBlock(nn.Module):
    def __init__(self, channels_in, channels_out, kernel=1, stride=1,
                 padding=0, groups=1, bias=False, norm=True, activation=True):
        super().__init__()
        # Keep the checkpoint's original parameter names.
        self.conv_1 = nn.Sequential(OrderedDict([
            ('conv3d', nn.Conv3d(channels_in, channels_out, kernel, stride, padding,
                                 groups=groups, bias=bias)),
            ('norm', nn.BatchNorm3d(channels_out, eps=.001) if norm else nn.Identity()),
            ('act', Swish() if activation else nn.Identity()),
        ]))

    def forward(self, x):
        return self.conv_1(x)


class SqueezeExcitation(nn.Module):
    def __init__(self, channels):
        super().__init__()
        target = channels // 4
        hidden = max(8, (target + 4) // 8 * 8)
        if hidden < .9 * target:
            hidden += 8
        self.fc1 = ConvBlock(channels, hidden, bias=True, norm=False, activation=False)
        self.activation_1 = Swish()
        self.activation_2 = nn.Sigmoid()
        self.fc2 = ConvBlock(hidden, channels, bias=True, norm=False, activation=False)

    def forward(self, x):
        scale = F.adaptive_avg_pool3d(x, 1)
        scale = self.fc2(self.activation_1(self.fc1(scale)))
        return self.activation_2(scale) * x


class Bottleneck(nn.Module):
    def __init__(self, channels_in, channels_out, expanded, kernel, stride, padding, pool_padding):
        super().__init__()
        self.expand = ConvBlock(channels_in, expanded)
        self.deep = ConvBlock(expanded, expanded, kernel, stride, padding, groups=expanded)
        self.se = SqueezeExcitation(expanded)
        self.project = ConvBlock(expanded, channels_out, activation=False)
        residual = []
        if stride != (1, 1, 1):
            residual.append(nn.AvgPool3d((1, 3, 3), stride=stride, padding=pool_padding))
        if residual or channels_in != channels_out:
            residual.append(ConvBlock(channels_in, channels_out, activation=False))
        self.res = nn.Sequential(*residual)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        residual = self.res(x)
        features = self.project(self.se(self.deep(self.expand(x))))
        return residual + self.alpha * features


class MoViNetA0(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = ConvBlock(3, 8, (1, 3, 3), (1, 2, 2), (0, 1, 1))
        self.blocks = nn.Sequential(OrderedDict(
            (f'b{stage}_l{layer}', Bottleneck(*spec))
            for stage, blocks in enumerate(A0_BLOCKS)
            for layer, spec in enumerate(blocks)
        ))
        self.conv7 = ConvBlock(104, 480)
        self.classifier = nn.Sequential(
            ConvBlock(480, 2048, bias=True, norm=False, activation=False),
            Swish(),
            nn.Dropout(p=.2, inplace=True),
            ConvBlock(2048, 2, bias=True, norm=False, activation=False),
        )
        self.apply(self.initialize_weights)

    @staticmethod
    def initialize_weights(layer):
        if isinstance(layer, nn.Conv3d):
            nn.init.kaiming_normal_(layer.weight, mode='fan_out')
            if layer.bias is not None:
                nn.init.zeros_(layer.bias)
        elif isinstance(layer, nn.BatchNorm3d):
            nn.init.ones_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x):
        x = self.conv7(self.blocks(self.conv1(x)))
        return self.classifier(F.adaptive_avg_pool3d(x, 1)).flatten(1)



HERE = Path(__file__).resolve().parent
CLASSES = ['No Dumping', 'Dumping']
SEED = 42

def save(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding='utf-8')

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''):
            h.update(b)
    return h.hexdigest()

def frames(path, sequential=False):
    """24 uniform, chronological RGB frames, whole clip; no label timestamps."""
    cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if n < 24 or fps <= 0:
            raise ValueError(f'Invalid/too short video {path}: {n} frames, {fps} fps')
        if sequential:
            declared=n
            n=0
            while cap.grab(): n+=1
            if n<24 or abs(n-declared)>max(2,declared*.01):
                raise ValueError(f'Decode count mismatch {path}: declared {declared}, decoded {n}')
            cap.release()
            cap=cv2.VideoCapture(str(path),cv2.CAP_FFMPEG,[cv2.CAP_PROP_N_THREADS,2])
        ids = np.linspace(0, n-1, 24).round().astype(int)
        result = []
        position=-1
        for idx in ids:
            if sequential:
                while position<int(idx):
                    ok=cap.grab(); position+=1
                    if not ok: raise ValueError(f'Sequential decode failed {path}: {position}')
                ok,img=cap.retrieve()
            else:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, img = cap.read()
            if not ok:
                if not sequential:
                    cap.release()
                    return frames(path,sequential=True)
                raise ValueError(f'Cannot decode {path}, frame {idx}')
            # Preserve aspect ratio using a black letterbox; identical at inference.
            h,w = img.shape[:2]
            scale = 172 / max(h,w)
            resized = cv2.resize(img, (max(1,round(w*scale)),max(1,round(h*scale))))
            canvas = np.zeros((172,172,3), np.uint8)
            rh,rw = resized.shape[:2]
            y,x = (172-rh)//2, (172-rw)//2
            canvas[y:y+rh,x:x+rw] = cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
            result.append(canvas)
        return np.stack(result), ids.tolist(), n, float(fps)
    finally:
        cap.release()

def signature(a):
    # Multi-time visual signature for conservative near-duplicate/scene screening.
    gray = np.stack([cv2.resize(cv2.cvtColor(x,cv2.COLOR_RGB2GRAY),(17,16)) for x in a[::4]])
    return (gray[:,:,1:] > gray[:,:,:-1]).reshape(-1)

def prepare(args):
    from sklearn.model_selection import StratifiedGroupKFold
    root = Path(args.data).resolve()
    out = HERE/'artifacts'
    cache = out/'frames'
    cache.mkdir(parents=True, exist_ok=True)
    report = {'data_root':str(root),'classes':CLASSES,'seed':SEED,'errors':[],
              'sampling':'24 uniform chronological whole-video frames; RGB uint8 172x172 letterbox',
              'limitations':['Visual duplicate screening cannot prove camera/person independence without source/session identities.'],
              'near_pairs':[], 'quarantine':[]}
    rows, sigs = [], []
    for split in ('train','valid','test'):
        videos = {p.stem:p for p in (root/split/'videos').glob('*.mp4')}
        labels = {p.stem:p for p in (root/split/'labels').glob('*.json')}
        if not videos and not labels:
            continue
        for stem in sorted(videos.keys() ^ labels.keys()):
            report['errors'].append(f'{split}/{stem}: missing video or label')
        for stem in sorted(videos.keys() & labels.keys()):
            try:
                label = json.loads(labels[stem].read_text(encoding='utf-8-sig'))
                raw = str(label['label']).lower()
                if raw not in ('true','false'):
                    raise ValueError(f'Unknown label {raw}')
                if label.get('metadata',{}).get('video_id') != videos[stem].name:
                    raise ValueError('metadata.video_id mismatch')
                digest = sha(videos[stem])
                target = cache/f'{split}_{stem}.npy'
                meta = target.with_suffix('.json')
                if target.exists() and meta.exists() and json.loads(meta.read_text())['sha256'] == digest:
                    a = np.load(target)
                    info = json.loads(meta.read_text())
                else:
                    a, ids, n, fps = frames(videos[stem])
                    np.save(target,a)
                    info = {'sha256':digest,'indices':ids,'frame_count':n,'fps':fps}
                    save(meta,info)
                assert a.shape == (24,172,172,3) and a.dtype == np.uint8
                rows.append({'id':stem,'original_split':split,'split':split,'label':int(raw=='true'),
                             'video':str(videos[stem]),'cache':str(target),'sha256':digest,
                             'decoded_sha256':hashlib.sha256(a.tobytes()).hexdigest(),**info})
                sigs.append(signature(a))
                if len(rows)%25 == 0:
                    print(f'Prepared {len(rows)} videos',flush=True)
            except Exception as exc:
                report['errors'].append(f'{split}/{stem}: {exc}')
    # Union exact duplicates and visually near-identical clips; validation is group-held-out.
    parent = list(range(len(rows)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(i,j):
        parent[find(j)] = find(i)
    for i in range(len(rows)):
        for j in range(i):
            exact = rows[i]['sha256']==rows[j]['sha256'] or rows[i]['decoded_sha256']==rows[j]['decoded_sha256']
            distance = float(np.mean(sigs[i]!=sigs[j]))
            if exact or distance <= .08:
                union(i,j)
                report['near_pairs'].append({'a':rows[j]['video'],'b':rows[i]['video'],'exact':exact,'distance':distance})
                if exact and rows[i]['label'] != rows[j]['label']:
                    report['errors'].append(f'Conflicting exact duplicate labels: {rows[i]["id"]}/{rows[j]["id"]}')
    for i,r in enumerate(rows):
        r['group'] = find(i)
    heldout_groups = {r['group'] for r in rows if r['split']=='test'}
    valid_groups = {r['group'] for r in rows if r['split']=='valid'}
    # Preserve original test; quarantine related training/validation clips instead.
    seen_exact = set()
    for r in rows:
        reason = None
        if r['split'] != 'test' and r['group'] in heldout_groups:
            reason = 'visual/exact group overlaps original test'
        elif r['split']=='train' and r['group'] in valid_groups:
            reason = 'group overlaps original validation'
        elif (r['original_split'],r['decoded_sha256']) in seen_exact:
            reason = 'duplicate within original split'
        if reason:
            r['split']='quarantine'
            report['quarantine'].append({'video':r['video'],'reason':reason})
        seen_exact.add((r['original_split'],r['decoded_sha256']))
    if not any(r['split']=='valid' for r in rows):
        train = [r for r in rows if r['split']=='train']
        folds = StratifiedGroupKFold(n_splits=5,shuffle=True,random_state=SEED)
        _, val = next(folds.split(np.zeros(len(train)),[r['label'] for r in train],[r['group'] for r in train]))
        for i in val:
            train[i]['split']='valid'
        report['validation_source']='seeded ~20% stratified group split from original train; no original valid directory'
    report['counts'] = {s:{CLASSES[y]:sum(r['split']==s and r['label']==y for r in rows) for y in (0,1)} for s in ('train','valid','test','quarantine')}
    for s in ('train','valid','test'):
        if min(report['counts'][s].values()) == 0:
            report['errors'].append(f'{s} lacks both classes')
    for a,b in (('train','valid'),('train','test'),('valid','test')):
        if {r['group'] for r in rows if r['split']==a} & {r['group'] for r in rows if r['split']==b}:
            report['errors'].append(f'Group leakage: {a}/{b}')
    report['ready'] = not report['errors']
    save(out/'manifest.json',rows)
    save(out/'audit.json',report)
    print(json.dumps({k:report[k] for k in ('counts','ready','errors')},indent=2),flush=True)
    if not report['ready']:
        raise RuntimeError('Data audit failed; see artifacts/audit.json')

def tensor(a):
    import torch
    return torch.from_numpy(np.ascontiguousarray(a.transpose(3,0,1,2))).float().div_(255)

def metrics(y,p):
    from sklearn.metrics import accuracy_score,balanced_accuracy_score,precision_recall_fscore_support,roc_auc_score,average_precision_score,confusion_matrix
    pred = np.asarray(p).argmax(1)
    pr,re,f1,_ = precision_recall_fscore_support(y,pred,labels=[0,1],zero_division=0)
    return {'accuracy':float(accuracy_score(y,pred)),'balanced_accuracy':float(balanced_accuracy_score(y,pred)),
            'macro_f1':float(np.mean(f1)),'per_class':{CLASSES[i]:{'precision':float(pr[i]),'recall':float(re[i]),'f1':float(f1[i])} for i in (0,1)},
            'roc_auc':float(roc_auc_score(y,np.asarray(p)[:,1])) if len(set(y))==2 else None,
            'pr_auc':float(average_precision_score(y,np.asarray(p)[:,1])) if len(set(y))==2 else None,
            'confusion_matrix_true_rows_predicted_columns':confusion_matrix(y,pred,labels=[0,1]).tolist()}

def train(args):
    import torch
    from torch.utils.data import Dataset,DataLoader
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    out = HERE/'artifacts'
    audit=json.loads((out/'audit.json').read_text(encoding='utf-8'))
    if not audit['ready']:
        raise RuntimeError('Audit not ready')
    rows=json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; refusing silent CPU training')
    device='cuda'
    class Clips(Dataset):
        def __init__(self,split): self.rows=[r for r in rows if r['split']==split]; self.augment=split=='train'
        def __len__(self): return len(self.rows)
        def __getitem__(self,i):
            r=self.rows[i]; a=np.load(r['cache'])
            if self.augment:
                if random.random()<.5: a=a[:,:,::-1,:]
                a=np.clip(a.astype(np.float32)*random.uniform(.85,1.15),0,255).astype(np.uint8)
            return tensor(a),r['label']
    loaders={s:DataLoader(Clips(s),batch_size=args.batch_size,shuffle=s=='train',num_workers=0,pin_memory=True) for s in ('train','valid','test')}
    net=MoViNetA0().to(device)
    counts=np.bincount([r['label'] for r in rows if r['split']=='train'],minlength=2)
    weights=torch.tensor(counts.sum()/(2*counts),dtype=torch.float32,device=device)
    criterion=torch.nn.CrossEntropyLoss(weight=weights)
    opt=torch.optim.AdamW(net.parameters(),lr=args.lr,weight_decay=.01)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=args.epochs)
    scaler=torch.amp.GradScaler('cuda')
    def evaluate(split):
        net.eval(); ys=[]; ps=[]; loss=0
        with torch.no_grad():
            for x,y in loaders[split]:
                x=x.to(device); y=y.to(device)
                with torch.amp.autocast('cuda'):
                    z=net(x); batch_loss=criterion(z,y)
                loss+=batch_loss.item()*len(y)
                ys.extend(y.cpu().tolist()); ps.extend(z.float().softmax(1).cpu().tolist())
        return ys,ps,loss/len(ys)
    config={'architecture':'MoViNet-A0 base','pretrained':False,'classes':CLASSES,'frames':24,'size':172,'normalization':'RGB /255',
            'seed':SEED,'epochs_limit':args.epochs,'batch_size':args.batch_size,'lr':args.lr,'optimizer':'AdamW','weight_decay':.01,
            'patience':args.patience,'selection':'validation macro F1, tie lower validation loss','threshold':.5,
            'torch':torch.__version__,'gpu':torch.cuda.get_device_name(0),'upstream_commit':'c2d1edf48fc6c5259707f9d833f22171b4f63493'}
    save(out/'config.json',config)
    best=(-1,float('-inf')); stale=0; history=[]; start=time.time()
    for epoch in range(1,args.epochs+1):
        net.train(); total=0
        for x,y in loaders['train']:
            x=x.to(device); y=y.to(device); opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda'):
                z=net(x); loss=criterion(z,y)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(net.parameters(),5)
            scaler.step(opt); scaler.update(); total+=loss.item()*len(y)
        ys,ps,vl=evaluate('valid'); m=metrics(ys,ps)
        entry={'epoch':epoch,'train_loss':total/len(loaders['train'].dataset),'val_loss':vl,'val_macro_f1':m['macro_f1'],'val_accuracy':m['accuracy'],'elapsed_seconds':time.time()-start}
        history.append(entry); save(out/'history.json',history); print(json.dumps(entry),flush=True)
        score=(m['macro_f1'],-vl)
        if score>best:
            best=score; stale=0
            torch.save({'model':net.state_dict(),'config':config,'epoch':epoch,'validation':m},out/'best.pt')
        else: stale+=1
        torch.save({'model':net.state_dict(),'optimizer':opt.state_dict(),'epoch':epoch,'config':config},out/'last.pt')
        scheduler.step()
        if stale>=args.patience: break
    checkpoint=torch.load(out/'best.pt',map_location=device,weights_only=False)
    net.load_state_dict(checkpoint['model'])
    ys,ps,_=evaluate('test')
    result=metrics(ys,ps); result['best_epoch']=checkpoint['epoch']; result['test_n']=len(ys)
    result['limitations']=audit['limitations']+['Single seeded run on a small dataset; no claim of deployment reliability.','Test used once after validation model selection.']
    save(out/'test_metrics.json',result)
    with open(out/'test_predictions.csv','w',newline='',encoding='utf-8') as f:
        w=csv.writer(f); w.writerow(['id','true','predicted','p_dumping'])
        for r,y,p in zip(loaders['test'].dataset.rows,ys,ps): w.writerow([r['id'],CLASSES[y],CLASSES[int(np.argmax(p))],p[1]])
    print(json.dumps(result,indent=2),flush=True)

def predict(args):
    import torch
    device='cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint=torch.load(args.checkpoint,map_location=device,weights_only=False)
    net=MoViNetA0().to(device).eval(); net.load_state_dict(checkpoint['model'])
    a,ids,n,fps=frames(args.video)
    with torch.no_grad(): p=net(tensor(a).unsqueeze(0).to(device)).softmax(1)[0].cpu().tolist()
    result={'video':str(Path(args.video).resolve()),'prediction':CLASSES[int(np.argmax(p))],'p_dumping':p[1],'frame_indices':ids}
    print(json.dumps(result,indent=2))
    if args.output: save(args.output,result)

if __name__=='__main__':
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('prepare'); a.add_argument('--data',default=str(HERE/'data'))
    a=sub.add_parser('train'); a.add_argument('--epochs',type=int,default=50); a.add_argument('--patience',type=int,default=12); a.add_argument('--batch-size',type=int,default=4); a.add_argument('--lr',type=float,default=.001)
    a=sub.add_parser('predict'); a.add_argument('video'); a.add_argument('--checkpoint',default=str(HERE/'artifacts/best.pt')); a.add_argument('--output')
    args=p.parse_args(); {'prepare':prepare,'train':train,'predict':predict}[args.command](args)
