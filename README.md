# كشف رمي النفايات في فيديوهات المراقبة

يحلّل التطبيق فيديوهات MP4 باستخدام نموذجين مدرّبين من الصفر. يصنّف **MoViNet A0** الفيديو كاملًا إلى `Dumping` أو `No Dumping` من 24 إطارًا مرتّبًا. يفحص **YOLOv8n** 12 إطارًا موزّعًا على الفيديو ويحدّد مناطق `littering` بمربعات. يعرض التطبيق نتيجة كل نموذج في جدول مستقل.

## تشغيل التطبيق

من جذر المستودع، باستخدام Python 3.14 على macOS بمعالج Apple Silicon:

```sh
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python app.py
```

الأوزان المدربة والتقييم المحفوظ موجودان في المستودع. في تبويب `Video Analysis` اختر فيديو ثم `MoViNet A0` أو `YOLOv8n` أو `Both models`، واضغط `Analyze Video`. تُحفظ صور YOLO المشروحة في `results/yolo/video_predictions/`. يعرض تبويب `Evaluation Results` المقاييس المحفوظة ومصفوفة الالتباس الخاصة بـ MoViNet.

## الملفات الأساسية

| الملف | الوظيفة |
| --- | --- |
| `app.py` | واجهة تحليل الفيديو وعرض التقييم |
| `data.py` | استخراج الإطارات وتجهيز مقاطع MIVIA وتقسيمها |
| `clean.py` | تدقيق MIVIA وتنظيف وسوم Littering وتصديرها بصيغة YOLO |
| `common.py` | تدريب MoViNet وتحليل الفيديو بأوزانه المحفوظة |
| `src/movinet/main.py` | بناء MoViNet A0 وتدريبه واستخدامه |
| `src/yolo/main.py` | تدريب YOLOv8n وكشف المناطق في الفيديو والصور |
| `evaluate.py` | تجميع نتائج التصنيف والكشف في تقريري JSON وCSV |
| `test_project.py` | اختبارات تجهيز البيانات والاستدلال والتقييم والواجهة |

## البيانات والتدريب

لتجهيز بيانات التدريب محليًا، ضع البيانات الأصلية ووسومها في المسارات التالية. تشغيل التطبيق بأوزانه المحفوظة يتطلب فيديو MP4 فقط.

- `datasets/MIVIA-IWDD 500/` مع مجلدَي `train` و`test`، ويحتوي كل منهما على `videos` و`labels`. [MIVIA-IWDD-500](https://github.com/MiviaLab/IllegalWasteDumping)
- `datasets/Littering.v8-yolotiny.coco/` بتصدير COCO، مع مجلدات `train` و`valid` و`test`. [Roboflow Littering v8](https://universe.roboflow.com/thesis-kztn8/littering-whlsk/dataset/8)

```sh
.venv/bin/python data.py
.venv/bin/python clean.py
```

يستخرج `data.py` 24 إطارًا لكل فيديو من MIVIA. يقسّم MoViNet مجموعة التدريب الرسمية إلى 320 فيديو للتعلّم و80 للتحقق باستخدام البذرة 42، ويترك 100 فيديو للاختبار الرسمي. تبقى إطارات الفيديو الواحد في مجموعة واحدة. يراجع `clean.py` ملفات MIVIA ويجهز `datasets/Littering-YOLO-clean/` لتدريب YOLO. تضم النسخة 11,882 صورة تدريب و1,168 صورة تحقق. مجموعة الاختبار الموردة لـ Littering تتضمن 1,230 صورة بلا وسوم.

للتدريب إلى مجلدات منفصلة، استخدم:

```sh
.venv/bin/python src/movinet/main.py train --models-dir models/movinet --results-dir runs/movinet
.venv/bin/python src/yolo/main.py train --models-dir models/yolo --results-dir runs/yolo
```

للتنبؤ من الطرفية بالأوزان المرفقة:

```sh
.venv/bin/python src/movinet/main.py predict "path/to/video.mp4"
.venv/bin/python src/yolo/main.py predict "path/to/video.mp4"
.venv/bin/python src/yolo/main.py image "path/to/image.jpg"
```

## النتائج المحفوظة

| النموذج | مجموعة التقييم | المقاييس |
| --- | --- | --- |
| MoViNet A0 | 100 فيديو من الاختبار الرسمي لـ MIVIA | Accuracy 72.00%، Precision 71.15%، Recall 74.00%، F1 72.55% |
| YOLOv8n | 1,168 صورة تحقق من Roboflow Littering | Precision 51.08%، Recall 72.24%، mAP@50 54.29%، mAP@50:95 24.50% |

مقاييس النموذجين تخص مهمتين مختلفتين. نتائج MoViNet استكشافية لأن مجموعة الاختبار استُخدمت سابقًا قبل تعديل معيار اختيار النموذج. تدرّبت أوزان YOLO المرفقة على التصدير السابق `Littering-YOLO`، ويستخدم أمر التدريب الحالي النسخة المنظّفة افتراضيًا. تُراجع نتيجة الفيديو مع التسجيل الأصلي قبل الاعتماد عليها.

بعد تجهيز إطارات MIVIA، أعد بناء تقرير المقارنة وشغّل الاختبارات:

```sh
.venv/bin/python evaluate.py \
  --prediction movinet_a0=results/movinet/movinet_a0_test_predictions.csv \
  --metadata movinet_a0=results/movinet/movinet_a0_run.json \
  --yolo-report results/yolo/yolov8n_run.json

.venv/bin/python -m unittest test_project -v
```
