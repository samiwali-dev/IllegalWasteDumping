"""Train short-window live classifiers on official-train videos only.

Static sequences are explicit no-release examples. The user's camera screenshot
and all official test videos are excluded from fitting and selection.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import random
import time

import cv2
import joblib
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from data import _frame_paths
from training_protocol import development_records, require_development_video
from src.movinet.main import build_model
from src.movinet.retrain import metrics

ROOT = Path(__file__).resolve().parent
DATA = ROOT/'datasets/live_v1'
OUT = ROOT/'results/live_v1'
WINDOW = 1.4


def letterbox(image):
    h, w = image.shape[:2]; scale = min(112/w, 112/h)
    small = cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))))
    canvas = np.zeros((112, 112, 3), np.uint8)
    y, x = (112-small.shape[0])//2, (112-small.shape[1])//2
    canvas[y:y+small.shape[0], x:x+small.shape[1]] = small[:, :, ::-1]
    return canvas


def prepare():
    DATA.mkdir(parents=True, exist_ok=True); OUT.mkdir(parents=True, exist_ok=True)
    if (DATA/'windows.npz').exists():
        return
    cv2.setNumThreads(1)
    arrays = {}; records = {}; excluded = []
    fit, validation = development_records()
    def read(row):
        fps = row['source_fps']; duration = (row['source_frame_count']-1)/fps
        event = row.get('event_timestamp_seconds_diagnostic_only')
        if row['label'] and (event is None or not 0 <= event <= duration):
            return None
        center = float(event) if row['label'] else duration/2
        start = max(0, min(center-WINDOW/2, duration-WINDOW))
        indices = np.rint(np.linspace(start*fps, min(duration, start+WINDOW)*fps, 24)).astype(int)
        cap = cv2.VideoCapture(str(ROOT/'datasets/MIVIA-IWDD 500/train/videos'/row['video_id']))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(indices[0])); current = int(indices[0])-1
        frames = []; image = None
        try:
            for index in indices:
                while current < index:
                    ok, image = cap.read(); current += 1
                    if not ok: raise ValueError(f"Decode failed: {row['video_id']}")
                frames.append(letterbox(image))
        finally:
            cap.release()
        return np.stack(frames)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for role, rows in [('fit', fit), ('validation', validation)]:
            images = []; labels = []; info = []
            for row, clip in zip(rows, pool.map(read, rows)):
                require_development_video(row['video_id'], role)
                if clip is None:
                    excluded.append(dict(video=row['video_id'], reason='Invalid event timestamp')); continue
                images.append(clip); labels.append(row['label'])
                info.append(dict(video=row['video_id'], kind='event_window' if row['label'] else 'no_dumping_window'))
            real_count = len(images)
            # A frozen image contains no release, including when its source
            # scene contains a person, a bag, or existing litter.
            for row in rows:
                if role == 'fit' and not row['label']: continue
                frame = cv2.imread(str(_frame_paths(row)[12])); assert frame is not None
                images.append(np.repeat(letterbox(frame)[None], 24, axis=0)); labels.append(0)
                info.append(dict(video=row['video_id'], kind='static_no_release'))
            arrays[role] = np.stack(images); arrays[role+'_labels'] = np.array(labels)
            records[role] = dict(real_windows=real_count, samples=info)
            print('Prepared', role, len(images), flush=True)
    np.savez(DATA/'windows.npz', **arrays)
    (OUT/'dataset.json').write_text(json.dumps(dict(window_seconds=WINDOW, frames=24, records=records,
        excluded=excluded, test_used=False, screenshot_used_for_training=False,
        note='Positive windows centered on existing training annotation times; timing precision is limited. Static negatives are constructed without temporal release.'), indent=2)+'\n')


def assess(labels, scores, real):
    a = metrics(labels[:real], scores[:real]); false_static = float(np.mean(np.array(scores[real:]) >= .5))
    return dict(real_windows=a, static_false_positive_rate=false_static,
                selection_score=a['macro_f1']*(1-false_static),
                eligible=a['recall'] >= .5 and false_static <= .1)


def small_models(cache, manifest):
    from src.neural_network.main import extract_nn_features, build_mlp
    cv2.setNumThreads(1); torch.set_num_threads(2)
    feature_file = DATA/'features.npz'
    if not feature_file.exists():
        with ThreadPoolExecutor(max_workers=4) as pool:
            features = {role: np.stack(list(pool.map(extract_nn_features, cache[role]))) for role in ('fit','validation')}
        np.savez(feature_file, **features)
    features = np.load(feature_file); x, v = features['fit'], features['validation']
    y, z = cache['fit_labels'], cache['validation_labels']; real = manifest['records']['validation']['real_windows']
    candidates = []; best = None
    for kernel, c in [('linear', .01), ('linear', .1), ('linear', 1), ('rbf', .1), ('rbf', 1), ('rbf', 10)]:
        model = make_pipeline(StandardScaler(), SVC(kernel=kernel, C=c, class_weight='balanced'))
        model.fit(x[:, :320], y); raw = model.decision_function(v[:, :320]); result = assess(z, (raw >= 0).astype(float), real)
        candidates.append(dict(kernel=kernel, C=c, **result))
        if best is None or (result['eligible'], result['selection_score']) > (best[0]['eligible'], best[0]['selection_score']): best = (result, model)
    joblib.dump(dict(model=best[1], frame_count=24, window_seconds=WINDOW), OUT/'svm.joblib', compress=3)
    (OUT/'svm.json').write_text(json.dumps(dict(selected=best[0], candidates=candidates), indent=2)+'\n')
    print('SVM', best[0], flush=True)
    mean = x.mean(0); std = np.maximum(x.std(0), 1e-3)
    train = torch.tensor((x-mean)/std, dtype=torch.float32); val = torch.tensor((v-mean)/std, dtype=torch.float32)
    model = build_mlp((64,32), .55); optimizer = torch.optim.AdamW(model.parameters(), lr=.0008, weight_decay=.05)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((y==0).sum()/max(1,(y==1).sum()), dtype=torch.float32))
    best = None; stale = 0; history = []
    for epoch in range(1,41):
        model.train(); order = np.random.default_rng(42+epoch).permutation(len(y))
        for start in range(0,len(y),32):
            ids=order[start:start+32]
            if len(ids)<2: continue
            optimizer.zero_grad(); loss=loss_fn(model(train[ids]).reshape(-1), torch.tensor(y[ids], dtype=torch.float32)); loss.backward(); optimizer.step()
        model.eval()
        with torch.inference_mode(): scores=torch.sigmoid(model(val)).reshape(-1).tolist()
        result=assess(z,scores,real); history.append(dict(epoch=epoch,**result))
        if best is None or (result['eligible'],result['selection_score']) > (best['eligible'],best['selection_score']):
            best=result; stale=0
            torch.save(dict(state_dict=model.state_dict(), hidden_layers=[64,32], dropout=.55,input_dim=528,mean=mean,std=std), OUT/'mlp.pt')
        else: stale+=1
        if stale>=8: break
    (OUT/'mlp.json').write_text(json.dumps(dict(selected=best,history=history),indent=2)+'\n');print('MLP',best,flush=True)


def movinet(cache, manifest):
    random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.set_num_threads(4)
    source=ROOT/'src/movinet/model.pt'; saved=torch.load(source,map_location='cpu',weights_only=True)
    model=build_model(tf_like=True);model.load_state_dict(saved['state_dict']);model.to('mps')
    optimizer=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.001)
    y,z=cache['fit_labels'],cache['validation_labels'];real=manifest['records']['validation']['real_windows']
    loss_fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor((y==0).sum()/max(1,(y==1).sum()),dtype=torch.float32,device='mps'))
    best=None;history=[];started=time.monotonic()
    for epoch in range(1,5):
        if epoch>1 and time.monotonic()-started>780:break
        model.train()
        for name,p in model.named_parameters():p.requires_grad_(epoch>1 or '.classifier.' in name)
        for layer in model.modules():
            if isinstance(layer,torch.nn.BatchNorm3d):layer.eval()
        order=np.random.default_rng(42+epoch).permutation(len(y))
        for start in range(0,len(y),8):
            ids=order[start:start+8];a=torch.from_numpy(cache['fit'][ids].copy()).permute(0,1,4,2,3).float().div_(255).to('mps')
            if random.random()<.5:a=a.flip(-1)
            a=(a*random.uniform(.95,1.05)).clamp(0,1)
            optimizer.zero_grad(set_to_none=True);loss=loss_fn(model(a).reshape(-1),torch.tensor(y[ids],dtype=torch.float32,device='mps'));loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1);optimizer.step()
        model.eval();scores=[]
        with torch.inference_mode():
            for start in range(0,len(z),8):
                a=torch.from_numpy(cache['validation'][start:start+8].copy()).permute(0,1,4,2,3).float().div_(255).to('mps');scores.extend(torch.sigmoid(model(a)).reshape(-1).cpu().tolist())
        result=assess(z,scores,real);history.append(dict(epoch=epoch,seconds=time.monotonic()-started,**result))
        if best is None or (result['eligible'],result['selection_score']) > (best['eligible'],best['selection_score']):
            best=result;torch.save(dict(state_dict={k:v.cpu().clone() for k,v in model.state_dict().items()},architecture=saved['architecture'],window_seconds=WINDOW),OUT/'movinet.pt')
        report=dict(selected=best,history=history,initial_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),window_seconds=WINDOW,test_used=False)
        (OUT/'movinet.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(history[-1]),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('kind',choices=['prepare','small','movinet']);args=p.parse_args()
    prepare()
    if args.kind!='prepare':
        cache=np.load(DATA/'windows.npz');manifest=json.loads((OUT/'dataset.json').read_text())
        (small_models if args.kind=='small' else movinet)(cache,manifest)
