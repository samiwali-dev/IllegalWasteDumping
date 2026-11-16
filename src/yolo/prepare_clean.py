"""Build an object-training experiment without official-test sources."""
import hashlib
import json
from pathlib import Path
import shutil
import sys

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from training_protocol import require_development_video

DATA = ROOT/'datasets/release_v2/objects'
RESULTS = ROOT/'results/retraining_v2/yolo'


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    records = []
    lists = {'train': [], 'val': []}

    def export(stem, role, image, label_text, provenance):
        split = 'train' if role == 'fit' else 'val'
        destination = DATA/'images'/split/f'{stem}.jpg'
        label_path = DATA/'labels'/split/f'{stem}.txt'
        destination.parent.mkdir(parents=True, exist_ok=True)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(image, Path):
            if not destination.exists():
                destination.hardlink_to(image)
        elif not cv2.imwrite(str(destination), image):
            raise OSError(destination)
        label_path.write_text(label_text)
        records.append(dict(file=str(destination.relative_to(ROOT)), role=role,
                            sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                            label_sha256=hashlib.sha256(label_text.encode()).hexdigest(), **provenance))
        weight = 8 if split == 'train' and provenance['kind'] == 'reviewed_object_crop' else 3 if split == 'train' and provenance['kind'] == 'reviewed_background' else 1
        lists[split].extend([str(destination)] * weight)

    source = ROOT/'datasets/litter_objects_v1'
    taco = json.loads((ROOT/'results/yolo/object_v1/dataset.json').read_text())
    for row in taco['records']:
        if row['split'] not in ('train', 'val'):
            continue
        stem = f"taco_{row['id']:04d}"
        export(stem, 'fit' if row['split'] == 'train' else 'validation',
               source/'images'/row['split']/f'{stem}.jpg',
               (source/'labels'/row['split']/f'{stem}.txt').read_text(),
               dict(kind='official_taco', source_id=row['id'], source_split=row['split']))
    review = json.loads((RESULTS/'object_annotation_review.json').read_text())
    for annotation_index, row in enumerate(review):
        if row['decision'] != 'accept_object_boxes':
            continue
        require_development_video(row['video_id'], row['role'] if 'role' in row else ('fit' if Path(row['file']).name.startswith('fit_') else 'validation'))
        role = 'fit' if Path(row['file']).name.startswith('fit_') else 'validation'
        source_image = ROOT / row['file']
        if hashlib.sha256(source_image.read_bytes()).hexdigest() != row['source_image_sha256']:
            raise ValueError('Reviewed annotation frame changed')
        image = cv2.imread(str(source_image))
        h, w = image.shape[:2]
        a, b, c, d = row['crop_display_xyxy']
        x1, y1, x2, y2 = round(a*w/480), round(b*h/270), round(c*w/480), round(d*h/270)
        crop = image[y1:y2, x1:x2]
        labels = []
        for a, b, c, d in row['object_boxes_display_xyxy']:
            left, top, right, bottom = a*w/480-x1, b*h/270-y1, c*w/480-x1, d*h/270-y1
            if not 0 <= left < right <= crop.shape[1]+1 or not 0 <= top < bottom <= crop.shape[0]+1:
                raise ValueError(f'Object outside reviewed crop: {row}')
            labels.append(f'0 {(left+right)/2/crop.shape[1]:.8f} {(top+bottom)/2/crop.shape[0]:.8f} {(right-left)/crop.shape[1]:.8f} {(bottom-top)/crop.shape[0]:.8f}')
        export(f"mivia_{Path(row['video_id']).stem}_{row['frame_index']:06d}_{annotation_index:03d}", role, crop, '\n'.join(labels)+'\n',
               dict(kind='reviewed_object_crop', source_video=row['video_id'], source_frame=row['frame_index'],
                    crop_display_xyxy=row['crop_display_xyxy'], boxes_display_xyxy=row['object_boxes_display_xyxy'],
                    annotation_provenance='Assistant visual review; pilot labels, not independently human verified'))
    negatives = json.loads((ROOT/'results/yolo/object_v1/hard_negative_review_decisions.json').read_text())
    for row in negatives['records']:
        if row['decision'] != 'accept_background' or row['role'] not in ('train', 'val'):
            continue
        role = 'fit' if row['role'] == 'train' else 'validation'
        require_development_video(row['source_video'], role)
        path = source/'hard_negative_review'/f"{row['id']}.jpg"
        if not path.exists():
            video = ROOT / row['source_relative_video']
            expected = negatives['source_video_sha256'][row['source_relative_video']]
            if hashlib.sha256(video.read_bytes()).hexdigest() != expected:
                raise ValueError('Reviewed background source video changed')
            capture = cv2.VideoCapture(str(video))
            capture.set(cv2.CAP_PROP_POS_FRAMES, row['frame_index'])
            ok, frame = capture.read()
            capture.release()
            if not ok:
                raise ValueError(f'Could not decode reviewed background: {video}')
            if row['scale'] < 1:
                h, w = frame.shape[:2]
                frame = cv2.resize(frame, (round(w*row['scale']), round(h*row['scale'])))
            x1, y1, x2, y2 = row['crop_xyxy']
            path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(path), frame[y1:y2, x1:x2]):
                raise OSError(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('Reviewed background content changed')
        export(f"background_{row['id']}", role, path, '',
               dict(kind='reviewed_background', source_video=row['source_video'], source_frame=row['frame_index']))
    hashes = {role: {r['sha256'] for r in records if r['role'] == role} for role in ('fit', 'validation')}
    if hashes['fit'] & hashes['validation']:
        raise ValueError('Image content leakage across development splits')
    for split, files in lists.items():
        (DATA/f'{split}.txt').write_text('\n'.join(files)+'\n')
    (DATA/'data.yaml').write_text(f'path: {json.dumps(str(DATA))}\ntrain: train.txt\nval: val.txt\nnames:\n  0: litter_object\n')
    report = dict(protocol='official_train_development_only_v2', records=records,
                  counts={role: sum(r['role'] == role for r in records) for role in ('fit', 'validation')},
                  exposures={split: len(files) for split, files in lists.items()},
                  test_used_for_training=False, test_used_for_selection=False,
                  excluded_sources=['All official MIVIA test clips', 'TACO held-out test images', 'Legacy Roboflow crops of unknown video provenance'],
                  note='A small ROI-label pilot supplements official TACO. Video classification uses all 320 fit and 80 validation clips; these object crops are a smaller, different dataset.')
    (RESULTS/'dataset.json').write_text(json.dumps(report, indent=2)+'\n')
    print(report['counts'], report['exposures'])


if __name__ == '__main__':
    main()
