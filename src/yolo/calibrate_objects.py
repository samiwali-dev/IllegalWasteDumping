"""Select a model/score operating point using validation data only.

Scores are not converted to calibrated probabilities. This selects a threshold.
"""
from pathlib import Path
import json
import sys

from ultralytics import YOLO

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.yolo.train_objects import DATA, RESULTS, iou, split_paths


def counts(rows, threshold):
    tp=fp=fn=negative_images=flagged_background_images=0
    for row in rows:
        truth=row["truth"];matched=set();detected=0
        for box,score in zip(row["boxes"],row["scores"]):
            if score < threshold: continue
            detected+=1
            options=[(iou(box,target),index) for index,target in enumerate(truth) if index not in matched]
            overlap,index=max(options,default=(0,-1))
            if overlap>=.5: matched.add(index);tp+=1
            else: fp+=1
        fn+=len(truth)-len(matched)
        if not truth:
            negative_images+=1;flagged_background_images+=int(detected>0)
    precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn)
    f05=1.25*precision*recall/max(1e-12,.25*precision+recall)
    rejection=1-flagged_background_images/max(1,negative_images)
    return {"threshold":threshold,"tp":tp,"fp":fp,"fn":fn,"precision":precision,"recall":recall,
            "negative_images":negative_images,"negative_images_with_detection":flagged_background_images,
            "f05":f05,"background_rejection":rejection,"selection_score":f05*rejection}


def main():
    data=DATA/"data_v2.yaml"
    validation_paths=split_paths(data,"val")
    candidates=[]
    for checkpoint in (RESULTS/"training/objects_transfer_v2/weights/best.pt",RESULTS/"training/objects_transfer_v2/weights/last.pt"):
        model=YOLO(str(checkpoint));rows=[]
        for result in model.predict(source=str(DATA/"val_v2.txt"),imgsz=640,conf=.05,device="mps",batch=8,rect=False,stream=True,verbose=False):
            path=Path(result.path);truth=[]
            for line in (DATA/"labels/val"/f"{path.stem}.txt").read_text().splitlines():
                _,cx,cy,w,h=map(float,line.split());truth.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
            rows.append({"file":path.name,"boxes":result.boxes.xyxyn.cpu().tolist(),"scores":result.boxes.conf.cpu().tolist(),"truth":truth})
        assert len(rows)==len(validation_paths)
        for step in range(10,81,5):
            point=counts(rows,step/100);point["checkpoint"]=str(checkpoint);candidates.append(point)
    eligible=[p for p in candidates if p["recall"]>=.35]
    if not eligible:raise RuntimeError("No validation operating point retains the required recall")
    selected=max(eligible,key=lambda p:(p["selection_score"],p["recall"]))
    report={"policy":"Validation F0.5 multiplied by background-image rejection, with object recall >= 0.35. Predefined threshold grid 0.10..0.80 in steps of 0.05; test data not used for this selection.",
            "score_note":"Operating threshold selection, not probability calibration. Final test benchmark has been reused during development.",
            "validation_images":len(validation_paths),"selected":selected,"validation_operating_points":candidates}
    (RESULTS/"operating_point_selection.json").write_text(json.dumps(report,indent=2)+"\n")
    print('SELECTED',json.dumps(selected),flush=True)
    model=YOLO(selected["checkpoint"])
    # The deployment point is measured separately from the fixed-0.25 comparison.
    paths=split_paths(data,"test");rows=[]
    for result in model.predict(source=str(DATA/"test_v2.txt"),imgsz=640,conf=.05,device="mps",batch=8,rect=False,stream=True,verbose=False):
        path=Path(result.path);truth=[]
        for line in (DATA/"labels/test"/f"{path.stem}.txt").read_text().splitlines():
            _,cx,cy,w,h=map(float,line.split());truth.append([cx-w/2,cy-h/2,cx+w/2,cy+h/2])
        rows.append({"file":path.name,"boxes":result.boxes.xyxyn.cpu().tolist(),"scores":result.boxes.conf.cpu().tolist(),"truth":truth})
    report["test_at_selected_threshold"]=counts(rows,selected["threshold"])
    (RESULTS/"selected_test_predictions.json").write_text(json.dumps(rows,indent=2)+"\n")
    (RESULTS/"operating_point_selection.json").write_text(json.dumps(report,indent=2)+"\n")
    print('TEST',json.dumps(report["test_at_selected_threshold"]),flush=True)


if __name__=="__main__":main()
