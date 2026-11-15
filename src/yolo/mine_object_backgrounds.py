"""Propose hard background examples from existing videos; never auto-label them."""
from __future__ import annotations

import json
import argparse
import hashlib
from pathlib import Path
import random
import sys

import cv2
import numpy as np
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data import load_manifest, split_records

DATA = ROOT/"datasets/litter_objects_v1"
RESULTS = ROOT/"results/yolo/object_v1"


def export_reviewed():
    """Recreate only the saved, visually reviewed crops and their explicit labels."""
    from src.yolo.train_objects import add_reviewed_negatives
    add_reviewed_negatives()
    review = json.loads((RESULTS/"hard_negative_review_decisions.json").read_text())
    paths = {role:[p for p in (DATA/f"{role}.txt").read_text().splitlines() if not Path(p).name.startswith("hard_negative_")]
             for role in ("train","val","test")}
    verified = set()
    for record in review["records"]:
        if record["decision"] != "accept_background":
            continue
        source = ROOT/record["source_relative_video"]
        crop_path = DATA/"hard_negative_review"/f"{record['id']}.jpg"
        if not crop_path.exists():
            if str(source) not in verified:
                if hashlib.sha256(source.read_bytes()).hexdigest() != review["source_video_sha256"][record["source_relative_video"]]:
                    raise ValueError(f"Reviewed source video changed: {source}")
                verified.add(str(source))
            cap=cv2.VideoCapture(str(source));cap.set(cv2.CAP_PROP_POS_FRAMES,record["frame_index"]);ok,image=cap.read();cap.release()
            if not ok: raise ValueError(f"Could not recreate reviewed frame: {source}")
            if record["scale"] < 1:
                h,w=image.shape[:2];image=cv2.resize(image,(round(w*record["scale"]),round(h*record["scale"])))
            x1,y1,x2,y2=record["crop_xyxy"];crop_path.parent.mkdir(exist_ok=True);cv2.imwrite(str(crop_path),image[y1:y2,x1:x2])
        if hashlib.sha256(crop_path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError(f"Reviewed crop checksum mismatch: {crop_path}")
        role=record["role"];target=DATA/"images"/role/f"hard_negative_{record['id']}.jpg"
        if not target.exists(): target.hardlink_to(crop_path)
        (DATA/"labels"/role/f"hard_negative_{record['id']}.txt").write_text("")
        paths[role].append(str(target))
    for role,files in paths.items():
        if role == "train":
            files=([p for p in files if Path(p).name.startswith("taco_")]
                   +[p for p in files if Path(p).name.startswith("negative_")]*10
                   +[p for p in files if Path(p).name.startswith("hard_negative_")]*4)
        (DATA/f"{role}_v2.txt").write_text("\n".join(files)+"\n")
    (DATA/"data_v2.yaml").write_text(f"path: {json.dumps(str(DATA))}\ntrain: train_v2.txt\nval: val_v2.txt\ntest: test_v2.txt\nnames:\n  0: litter_object\n")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--export-reviewed", action="store_true")
    if parser.parse_args().export_reviewed:
        export_reviewed()
        return
    fit, validation, test = split_records(load_manifest())
    already_reviewed = {"vid0012.mp4", "vid0023.mp4", "vid0044.mp4", "vid0069.mp4", "vid0075.mp4", "vid0084.mp4",
                        "vid0005.mp4", "vid0018.mp4", "vid0021.mp4", "vid0038.mp4", "vid0040.mp4", "vid0043.mp4"}
    selections = {}
    for role, pool, count in (("train", fit, 20), ("val", validation, 6), ("test", test, 8)):
        candidates = [row for row in pool if row["label"] == 0 and row["video_id"] not in already_reviewed]
        random.Random(624).shuffle(candidates)
        selections[role] = candidates[:count]
    output = DATA/"hard_negative_review"
    output.mkdir(exist_ok=True)
    (RESULTS/"hard_negative_video_selection.json").write_text(json.dumps({k:[r["video_id"] for r in v] for k,v in selections.items()}, indent=2)+"\n")
    model = YOLO(str(RESULTS/"training/objects_transfer_v1/weights/best.pt"))
    rows = []
    for role, selected in selections.items():
        for record in selected:
            source = ROOT/"datasets/MIVIA-IWDD 500"/record["split"]/"videos"/record["video_id"]
            cap = cv2.VideoCapture(str(source))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for ordinal, fraction in enumerate((.33, .67)):
                index = round((total-1)*fraction)
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, original = cap.read()
                if not ok:
                    continue
                h, w = original.shape[:2]
                scale = min(1, 960/max(w,h))
                image = cv2.resize(original,(round(w*scale),round(h*scale))) if scale < 1 else original
                h,w = image.shape[:2]
                result = model.predict(image, imgsz=640, conf=.25, device="mps", verbose=False)[0]
                for box, confidence in list(zip(result.boxes.xyxy.cpu().tolist(),result.boxes.conf.cpu().tolist()))[:3]:
                    cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
                    bw,bh=max(100,(box[2]-box[0])*1.8),max(100,(box[3]-box[1])*1.8)
                    x1,y1=max(0,int(cx-bw/2)),max(0,int(cy-bh/2));x2,y2=min(w,int(cx+bw/2)),min(h,int(cy+bh/2))
                    if (x2-x1)*(y2-y1)>w*h*.6:
                        continue
                    key=f'H{len(rows):03d}';path=output/f'{key}.jpg'
                    cv2.imwrite(str(path),image[y1:y2,x1:x2])
                    rows.append({"id":key,"role":role,"file":str(path),"source_video":record["video_id"],
                                 "source_relative_video":str(source.relative_to(ROOT)),"frame_index":index,"scale":scale,
                                 "crop_xyxy":[x1,y1,x2,y2],"proposal_confidence":confidence,
                                 "decision":"pending","reason":""})
            cap.release()
            print(role,record["video_id"],"proposals",len(rows),flush=True)
    (output/"review.json").write_text(json.dumps(rows,indent=2)+"\n")
    for start in range(0,len(rows),48):
        subset=rows[start:start+48]
        sheet=np.full((int(np.ceil(len(subset)/6))*205,6*210,3),245,np.uint8)
        for index,row in enumerate(subset):
            image=cv2.imread(row["file"]);h,w=image.shape[:2];scale=min(200/w,175/h)
            tile=cv2.resize(image,(round(w*scale),round(h*scale)));th,tw=tile.shape[:2];x=index%6*210;y=index//6*205
            sheet[y:y+th,x:x+tw]=tile
            cv2.putText(sheet,f'{row["id"]} / {row["role"]}',(x+3,y+195),cv2.FONT_HERSHEY_SIMPLEX,.45,(20,20,20),1)
        cv2.imwrite(str(RESULTS/f'hard_negative_{start//48:02d}_review.jpg'),sheet)
    print('Proposals awaiting visual review:',len(rows),flush=True)


if __name__ == "__main__":
    main()
