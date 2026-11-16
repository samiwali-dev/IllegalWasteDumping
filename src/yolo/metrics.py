"""Object-box matching and validation operating-point metrics."""


def iou(a, b):
    width = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    height = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    intersection = width * height
    union = max(0, a[2]-a[0])*max(0, a[3]-a[1]) + max(0, b[2]-b[0])*max(0, b[3]-b[1]) - intersection
    return intersection/union if union else 0


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
