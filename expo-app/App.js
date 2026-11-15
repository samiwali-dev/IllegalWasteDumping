import React, { useEffect, useRef, useState } from 'react';
import {
  ActivityIndicator,
  Image,
  Platform,
  Pressable,
  SafeAreaView,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { StatusBar } from 'expo-status-bar';
import { analyzeLiveFramesOnEdge } from './liveEdgeAi';

const FALLBACK_EVALUATION = {
  dataset: 'MIVIA-IWDD 500',
  evaluation_split: 'official_test',
  split_counts: { fit: 320, validation: 80, test: 100 },
  models: {
    neural_network: {
      task: 'video_classification',
      dataset: 'MIVIA-IWDD 500',
      evaluation_split: 'official_test',
      metrics: {
        accuracy: 0.74,
        precision: 0.74,
        recall: 0.74,
        f1: 0.74,
        confusion_matrix: [
          [37, 13],
          [13, 37],
        ],
        true_negative: 37,
        false_positive: 13,
        false_negative: 13,
        true_positive: 37,
      },
      training_seconds: 21.53,
      inference_mean_ms_per_video: 279.88,
      model_size_mib: 0.15,
    },
    movinet_a0: {
      task: 'video_classification',
      dataset: 'MIVIA-IWDD 500',
      evaluation_split: 'official_test',
      metrics: {
        accuracy: 0.72,
        precision: 0.711538,
        recall: 0.74,
        f1: 0.72549,
        confusion_matrix: [
          [35, 15],
          [13, 37],
        ],
        true_negative: 35,
        false_positive: 15,
        false_negative: 13,
        true_positive: 37,
      },
      training_seconds: 1899.78,
      inference_mean_ms_per_video: 337.22,
      model_size_mib: 7.38,
    },
    optical_flow_svm: {
      task: 'video_classification',
      dataset: 'MIVIA-IWDD 500',
      evaluation_split: 'official_test',
      metrics: {
        accuracy: 0.64,
        precision: 0.62963,
        recall: 0.68,
        f1: 0.653846,
        confusion_matrix: [
          [30, 20],
          [16, 34],
        ],
        true_negative: 30,
        false_positive: 20,
        false_negative: 16,
        true_positive: 34,
      },
      training_seconds: 119.17,
      inference_mean_ms_per_video: 272.71,
      model_size_mib: 0.38,
    },
    yolov8n: {
      task: 'object_detection',
      dataset: 'Roboflow Littering',
      evaluation_split: 'Roboflow validation (1,168 images)',
      metrics: {
        precision: 0.510825,
        recall: 0.722365,
        map50: 0.542863,
        map50_95: 0.24498,
      },
      training_seconds: 821.94,
      validation_seconds: 21.85,
      model_size_mib: 5.93,
    },
  },
};

const MODEL_META = {
  neural_network: {
    title: 'Neural Network (MLP)',
    subtitle: '528 Spatiotemporal Motion + Scene-Change Features',
    badge: 'الأعلى دقة (74%)',
    color: '#10B981',
  },
  movinet_a0: {
    title: 'MoViNet A0',
    subtitle: '3D Video Convolutional Network (24 RGB Frames)',
    badge: '72% Accuracy',
    color: '#3B82F6',
  },
  optical_flow_svm: {
    title: 'Optical Flow + SVM',
    subtitle: '320 Farneback Motion Grid Features + RBF SVM',
    badge: '64% Accuracy',
    color: '#F59E0B',
  },
  yolov8n: {
    title: 'YOLOv8n Detector',
    subtitle: 'Bounding-Box Littering Region Detection (12 Sampled Frames)',
    badge: 'mAP@50: 54.3%',
    color: '#8B5CF6',
  },
};

const FILTER_CHOICES = [
  { id: 'all', label: 'جميع الخوارزميات الـ 4 (All 4 Models)' },
  { id: 'neural_network', label: 'Neural Network (MLP)' },
  { id: 'movinet_a0', label: 'MoViNet A0' },
  { id: 'optical_flow_svm', label: 'Optical Flow + SVM' },
  { id: 'yolov8n', label: 'YOLOv8n Detector' },
];

export default function App() {
  const [activeTab, setActiveTab] = useState('analyze');
  const [apiBase, setApiBase] = useState('http://localhost:8000');
  const [selectedChoice, setSelectedChoice] = useState('all');
  const [pickedFile, setPickedFile] = useState(null);
  const [selectedSample, setSelectedSample] = useState(null);
  const [samples, setSamples] = useState([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [statusText, setStatusText] = useState('اختر فيديو MP4 من جهازك أو من عينات الاختبار للبدء.');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [evaluationData, setEvaluationData] = useState(FALLBACK_EVALUATION);
  const [selectedEvalModel, setSelectedEvalModel] = useState('neural_network');
  const [apiConnected, setApiConnected] = useState(false);

  // Live Camera state
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraFacing, setCameraFacing] = useState('back');
  const [cameraReady, setCameraReady] = useState(false);
  const [liveMonitoring, setLiveMonitoring] = useState(false);
  const [liveAnalyzing, setLiveAnalyzing] = useState(false);
  const [liveStatus, setLiveStatus] = useState('الكاميرا جاهزة للبث والتحليل المباشر.');
  const [liveResult, setLiveResult] = useState(null);
  const [bufferedCount, setBufferedCount] = useState(0);

  const cameraRef = useRef(null);
  const baselineFrameRef = useRef(null);
  const frameBufferRef = useRef([]);
  const liveTimerRef = useRef(null);
  const busyLockRef = useRef(false);

  const cleanBaseUrl = apiBase.replace(/\/+$/, '');

  const fetchBackendData = async () => {
    try {
      const [evalRes, sampRes] = await Promise.all([
        fetch(`${cleanBaseUrl}/api/evaluation`),
        fetch(`${cleanBaseUrl}/api/samples`),
      ]);
      if (evalRes.ok) {
        const evalJson = await evalRes.json();
        if (evalJson && evalJson.models) {
          setEvaluationData(evalJson);
          setApiConnected(true);
        }
      }
      if (sampRes.ok) {
        const sampJson = await sampRes.json();
        if (sampJson && Array.isArray(sampJson.samples)) {
          setSamples(sampJson.samples);
        }
      }
    } catch (_err) {
      setApiConnected(false);
    }
  };

  useEffect(() => {
    fetchBackendData();
  }, [cleanBaseUrl]);

  // Stop live interval if leaving live tab
  useEffect(() => {
    if (activeTab !== 'live' && liveMonitoring) {
      stopLiveMonitoring();
    }
    return () => {
      if (liveTimerRef.current) clearInterval(liveTimerRef.current);
    };
  }, [activeTab]);

  const getSelectedModelsList = () =>
    selectedChoice === 'all'
      ? ['neural_network', 'movinet_a0', 'optical_flow_svm', 'yolov8n']
      : [selectedChoice];

  const captureSingleBase64Frame = async () => {
    if (!cameraRef.current || !cameraReady) return null;
    try {
      const photo = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: 0.55,
        skipProcessing: true,
      });
      if (!photo) return null;
      if (photo.base64) {
        return `data:image/jpeg;base64,${photo.base64}`;
      }
      if (photo.uri && photo.uri.startsWith('data:')) {
        return photo.uri;
      }
      return null;
    } catch (_e) {
      return null;
    }
  };

  const runFramesAnalysis = async (framesToAnalyze) => {
    const modelsParam = getSelectedModelsList();
    const anchoredFrames = baselineFrameRef.current
      ? [baselineFrameRef.current, ...framesToAnalyze]
      : framesToAnalyze;

    // Try Python Backend first if reachable; otherwise run Edge/Browser AI (for Vercel & Offline Mobile)
    try {
      const controller = new AbortController();
      const timeoutId = setTimeout(() => controller.abort(), 5000);
      const res = await fetch(`${cleanBaseUrl}/api/analyze-live`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          frames: anchoredFrames,
          models: modelsParam,
        }),
        signal: controller.signal,
      });
      clearTimeout(timeoutId);
      if (res.ok) {
        const data = await res.json();
        setApiConnected(true);
        return { ...data, mode: 'python_backend' };
      }
    } catch (_err) {
      setApiConnected(false);
    }
    // Fallback to Edge AI (uses exported PyTorch Neural Network weights + calibrated motion & scene change)
    return await analyzeLiveFramesOnEdge(framesToAnalyze, modelsParam, baselineFrameRef.current);
  };

  const handleResetBaseline = async () => {
    if (!cameraReady) return;
    const base = await captureSingleBase64Frame();
    if (base) {
      baselineFrameRef.current = base;
      frameBufferRef.current = [base];
      setBufferedCount(1);
      setLiveResult(null);
      setLiveStatus('✅ تم حفظ خلفية المشهد النظيفة! الآن ارمِ أو ضع أي غرض أمام الكاميرا ليتم كشفه.');
    }
  };

  const handleCaptureBurstAndAnalyze = async () => {
    if (!cameraReady || liveAnalyzing) return;
    setLiveAnalyzing(true);
    try {
      if (!baselineFrameRef.current) {
        const initialBase = await captureSingleBase64Frame();
        if (initialBase) baselineFrameRef.current = initialBase;
      }
      const burst = [];
      const totalSteps = 8;
      for (let i = 1; i <= totalSteps; i++) {
        setLiveStatus(`📸 جاري رصد حركة الرمي (${i}/${totalSteps})... ارمِ أو حرّك الغرض أمام الكاميرا الآن!`);
        const frame = await captureSingleBase64Frame();
        if (frame) burst.push(frame);
        await new Promise((r) => setTimeout(r, 270));
      }
      if (burst.length === 0) {
        setLiveStatus('تعذر التقاط إطارات من الكاميرا. تأكد من السماح بصلاحية الكاميرا.');
        return;
      }
      frameBufferRef.current = burst;
      setBufferedCount(burst.length);
      setLiveStatus('🧠 جاري تحليل الحركة وتغير المشهد بالخوارزميات الأربع...');
      const res = await runFramesAnalysis(burst);
      setLiveResult(res);
      const anyDumping = res.actions && res.actions.some((a) => a.prediction === 1);
      setLiveStatus(
        anyDumping
          ? '⚠️ تم رصد عملية رمي (DUMPING DETECTED) وتحديد موقع الغرض في المشهد!'
          : '✅ المشهد سليم (NO DUMPING) — اضغط "تثبيت خلفية المشهد" قبل الرمي لرصد الفارق بدقة.'
      );
    } catch (err) {
      setLiveStatus(`خطأ أثناء تحليل الكاميرا: ${err.message}`);
    } finally {
      setLiveAnalyzing(false);
    }
  };

  const startLiveMonitoring = async () => {
    if (!cameraReady) {
      setLiveStatus('يرجى الانتظار حتى تصبح الكاميرا جاهزة.');
      return;
    }
    if (!baselineFrameRef.current) {
      const base = await captureSingleBase64Frame();
      if (base) baselineFrameRef.current = base;
    }
    setLiveMonitoring(true);
    setLiveStatus('🔴 المراقبة المباشرة نشطة: أي غرض ترميه أو تتركه أمام الكاميرا سيُكشف فوراً...');
    liveTimerRef.current = setInterval(async () => {
      if (busyLockRef.current) return;
      busyLockRef.current = true;
      try {
        const frame = await captureSingleBase64Frame();
        if (frame) {
          frameBufferRef.current = [...frameBufferRef.current, frame].slice(-7);
          setBufferedCount(frameBufferRef.current.length);
          if (frameBufferRef.current.length >= 2) {
            const res = await runFramesAnalysis(frameBufferRef.current);
            setLiveResult(res);
          }
        }
      } catch (_e) {
        // Ignore transient frame errors during live stream
      } finally {
        busyLockRef.current = false;
      }
    }, 1400);
  };

  const stopLiveMonitoring = () => {
    if (liveTimerRef.current) {
      clearInterval(liveTimerRef.current);
      liveTimerRef.current = null;
    }
    busyLockRef.current = false;
    setLiveMonitoring(false);
    setLiveStatus('تم إيقاف المراقبة التلقائية.');
  };

  const handlePickVideo = async () => {
    try {
      const res = await DocumentPicker.getDocumentAsync({
        type: ['video/mp4', 'video/*'],
        copyToCacheDirectory: true,
      });
      if (!res.canceled && res.assets && res.assets.length > 0) {
        const asset = res.assets[0];
        setPickedFile(asset);
        setSelectedSample(null);
        setAnalysisResult(null);
        setStatusText(`تم اختيار الملف: ${asset.name}`);
      }
    } catch (err) {
      setStatusText(`تعذر قراءة الملف: ${err.message}`);
    }
  };

  const handleSelectSample = (sample) => {
    setSelectedSample(sample);
    setPickedFile(null);
    setAnalysisResult(null);
    setStatusText(
      `تم اختيار عينة الاختبار: ${sample.video_id} (${
        sample.label === 1 ? 'رمي فعلي / Dumping' : 'سليم / No Dumping'
      })`
    );
  };

  const handleRunAnalysis = async () => {
    if (!pickedFile && !selectedSample) {
      setStatusText('يرجى اختيار فيديو MP4 أو عينة اختبار أولاً.');
      return;
    }

    setAnalyzing(true);
    setAnalysisResult(null);
    const modelsParam = getSelectedModelsList();
    setStatusText('جاري استخراج الـ 24 إطاراً وتشغيل الخوارزميات المختارة...');

    try {
      let response;
      if (selectedSample) {
        response = await fetch(`${cleanBaseUrl}/api/analyze`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            sample_video_id: selectedSample.video_id,
            models: modelsParam,
          }),
        });
      } else {
        const formData = new FormData();
        formData.append('models', modelsParam.join(','));
        if (Platform.OS === 'web' && pickedFile.file) {
          formData.append('video', pickedFile.file, pickedFile.name || 'video.mp4');
        } else {
          formData.append('video', {
            uri: pickedFile.uri,
            name: pickedFile.name || 'video.mp4',
            type: pickedFile.mimeType || 'video/mp4',
          });
        }
        response = await fetch(`${cleanBaseUrl}/api/analyze`, {
          method: 'POST',
          body: formData,
        });
      }

      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error || 'فشل التحليل من الخادم');
      }
      setAnalysisResult(data);
      setApiConnected(true);
      setStatusText('اكتمل التحليل بنجاح!');
    } catch (err) {
      setStatusText(
        `خطأ في الاتصال بالخادم (${cleanBaseUrl}): لتحليل ملفات MP4 الكاملة شغّل (.venv/bin/python api.py)، أو استخدم تبويب "كاميرا لايف مباشر" الذي يعمل مباشرة على Vercel وبدون خادم!`
      );
    } finally {
      setAnalyzing(false);
    }
  };

  const renderHeader = () => (
    <View style={styles.header}>
      <View style={styles.headerTop}>
        <View style={{ flex: 1 }}>
          <Text style={styles.appTitle}>كشف رمي النفايات غير القانوني بالذكاء الاصطناعي</Text>
          <Text style={styles.appSubtitle}>
            Illegal Waste Dumping AI • Web, Mobile & Live Camera (4 Algorithms)
          </Text>
        </View>
        <View
          style={[
            styles.statusPill,
            { backgroundColor: apiConnected ? 'rgba(16, 185, 129, 0.18)' : 'rgba(56, 189, 248, 0.18)' },
          ]}
        >
          <View
            style={[
              styles.statusDot,
              { backgroundColor: apiConnected ? '#10B981' : '#38BDF8' },
            ]}
          />
          <Text style={styles.statusPillText}>
            {apiConnected ? 'متصل بخادم Python (4 Models)' : 'Edge AI جاهز (Vercel / Browser)'}
          </Text>
        </View>
      </View>

      <View style={styles.tabBar}>
        <Pressable
          style={[styles.tabButton, activeTab === 'analyze' && styles.tabButtonActive]}
          onPress={() => setActiveTab('analyze')}
        >
          <Text style={[styles.tabButtonText, activeTab === 'analyze' && styles.tabButtonTextActive]}>
            🎥 تحليل فيديو (MP4)
          </Text>
        </Pressable>
        <Pressable
          style={[styles.tabButton, activeTab === 'live' && styles.tabButtonActive]}
          onPress={() => setActiveTab('live')}
        >
          <Text style={[styles.tabButtonText, activeTab === 'live' && styles.tabButtonTextActive]}>
            📷 كاميرا لايف (Live AI)
          </Text>
        </Pressable>
        <Pressable
          style={[styles.tabButton, activeTab === 'evaluation' && styles.tabButtonActive]}
          onPress={() => setActiveTab('evaluation')}
        >
          <Text style={[styles.tabButtonText, activeTab === 'evaluation' && styles.tabButtonTextActive]}>
            📊 مقارنة الـ 4 خوارزميات
          </Text>
        </Pressable>
      </View>
    </View>
  );

  const renderAlgorithmChips = () => (
    <View style={styles.filterGrid}>
      {FILTER_CHOICES.map((item) => {
        const active = selectedChoice === item.id;
        return (
          <Pressable
            key={item.id}
            onPress={() => setSelectedChoice(item.id)}
            style={[styles.filterChip, active && styles.filterChipActive]}
          >
            <Text style={[styles.filterChipText, active && styles.filterChipTextActive]}>
              {item.label}
            </Text>
          </Pressable>
        );
      })}
    </View>
  );

  const renderPredictionsList = (resultObj) => {
    if (!resultObj) return null;
    return (
      <View style={styles.card}>
        <View style={styles.resultHeaderRow}>
          <Text style={styles.cardTitle}>
            نتائج التصنيف — {resultObj.video_name}
          </Text>
          {resultObj.ground_truth_verdict && (
            <View style={styles.gtBadge}>
              <Text style={styles.gtBadgeText}>
                الوسم الحقيقي: {resultObj.ground_truth_verdict}
              </Text>
            </View>
          )}
        </View>

        {resultObj.actions && resultObj.actions.length > 0 ? (
          resultObj.actions.map((row) => {
            const isDumping = row.prediction === 1;
            const meta = MODEL_META[row.model] || { title: row.label_title || row.model, color: '#38BDF8' };
            return (
              <View key={row.model} style={styles.predictionRow}>
                <View style={{ flex: 1 }}>
                  <Text style={styles.predictionModelName}>{row.label_title || meta.title}</Text>
                  <Text style={styles.predictionKindText}>{row.score_kind}</Text>
                </View>
                <View style={styles.predictionStats}>
                  <View
                    style={[
                      styles.verdictBadge,
                      {
                        backgroundColor: isDumping
                          ? 'rgba(239, 68, 68, 0.2)'
                          : 'rgba(16, 185, 129, 0.2)',
                        borderColor: isDumping ? '#EF4444' : '#10B981',
                      },
                    ]}
                  >
                    <Text
                      style={[
                        styles.verdictText,
                        { color: isDumping ? '#FCA5A5' : '#6EE7B7' },
                      ]}
                    >
                      {isDumping ? '⚠️ DUMPING (رمي نفايات)' : '✅ NO DUMPING (سليم)'}
                    </Text>
                  </View>
                  <Text style={styles.scoreText}>
                    Score: {Number(row.score).toFixed(4)} • {(Number(row.seconds) * 1000).toFixed(0)} ms
                  </Text>
                </View>
              </View>
            );
          })
        ) : (
          <Text style={styles.helperText}>لم يتم اختيار مصنفات الفيديو في هذه العملية.</Text>
        )}
      </View>
    );
  };

  const renderLiveCameraTab = () => {
    if (!permission) {
      return (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>جاري التحقق من صلاحيات الكاميرا...</Text>
        </View>
      );
    }

    if (!permission.granted) {
      return (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>📷 تفعيل الكاميرا المباشرة (Live Camera AI)</Text>
          <Text style={styles.helperText}>
            يتطلب البث المباشر صلاحية الوصول إلى الكاميرا في المتصفح أو الجوال لرصد رمي النفايات بالوقت الفعلي.
          </Text>
          <Pressable style={[styles.primaryButton, { marginTop: 14, alignSelf: 'flex-start' }]} onPress={requestPermission}>
            <Text style={styles.primaryButtonText}>السماح بتشغيل الكاميرا (Grant Camera Access)</Text>
          </Pressable>
        </View>
      );
    }

    return (
      <View style={styles.sectionContainer}>
        <View style={styles.card}>
          <View style={styles.resultHeaderRow}>
            <View>
              <Text style={styles.cardTitle}>📷 البث المباشر للكاميرا (Live Camera Stream)</Text>
              <Text style={styles.helperText}>
                يعمل مباشرة على الويب (Vercel) والجوال! يلتقط إطارات متتابعة ويصنف الحركة وتغير المشهد بالشبكة العصبية.
              </Text>
            </View>
            <View style={styles.gtBadge}>
              <Text style={styles.gtBadgeText}>الإطارات المخزنة: {bufferedCount}</Text>
            </View>
          </View>

          <View style={styles.cameraSplitRow}>
            {/* Live Camera Preview */}
            <View style={styles.cameraContainer}>
              <CameraView
                ref={cameraRef}
                style={styles.cameraView}
                facing={cameraFacing}
                onCameraReady={() => setCameraReady(true)}
              />
              <View style={styles.cameraOverlayBar}>
                <View style={styles.liveIndicator}>
                  <View
                    style={[
                      styles.statusDot,
                      { backgroundColor: liveMonitoring ? '#EF4444' : '#10B981' },
                    ]}
                  />
                  <Text style={styles.liveIndicatorText}>
                    {liveMonitoring ? 'LIVE AUTO-MONITORING' : cameraReady ? 'CAMERA READY' : 'INITIALIZING...'}
                  </Text>
                </View>
                <View style={{ flexDirection: 'row', gap: 8 }}>
                  <Pressable
                    style={styles.flipCameraBtn}
                    onPress={handleResetBaseline}
                  >
                    <Text style={styles.flipCameraText}>🎯 تثبيت خلفية المشهد</Text>
                  </Pressable>
                  <Pressable
                    style={styles.flipCameraBtn}
                    onPress={() => {
                      baselineFrameRef.current = null;
                      setCameraFacing((prev) => (prev === 'back' ? 'front' : 'back'));
                    }}
                  >
                    <Text style={styles.flipCameraText}>🔄 قلب الكاميرا</Text>
                  </Pressable>
                </View>
              </View>
            </View>

            {/* Annotated YOLOv8n / Live Snapshot Box */}
            {liveResult && liveResult.yolo_live && liveResult.yolo_live.annotated_data_uri && (
              <View style={styles.liveSnapshotContainer}>
                <Image
                  source={{ uri: liveResult.yolo_live.annotated_data_uri }}
                  style={styles.cameraView}
                  resizeMode="cover"
                />
                <View style={styles.cameraOverlayBar}>
                  <Text style={styles.liveIndicatorText}>
                    🎯 كشف المناطق: {liveResult.yolo_live.total_detections} منطقة
                  </Text>
                </View>
              </View>
            )}
          </View>

          <Text style={[styles.subHeaderTitle, { marginTop: 14 }]}>اختر الخوارزميات للبث المباشر:</Text>
          {renderAlgorithmChips()}

          <View style={styles.statusBanner}>
            {(liveAnalyzing || liveMonitoring) && (
              <ActivityIndicator color="#38BDF8" style={{ marginRight: 10 }} />
            )}
            <Text style={styles.statusBannerText}>{liveStatus}</Text>
          </View>

          <View style={styles.liveButtonsRow}>
            <Pressable
              style={[
                styles.analyzeCtaButton,
                { flex: 1, marginTop: 0 },
                (!cameraReady || liveAnalyzing) && styles.buttonDisabled,
              ]}
              disabled={!cameraReady || liveAnalyzing}
              onPress={handleCaptureBurstAndAnalyze}
            >
              <Text style={styles.analyzeCtaText}>
                {liveAnalyzing ? 'جاري الالتقاط والتحليل...' : '⚡ التقاط وتحليل فوري (Capture & Analyze)'}
              </Text>
            </Pressable>

            <Pressable
              style={[
                styles.liveToggleBtn,
                liveMonitoring ? styles.liveStopBtn : styles.liveStartBtn,
              ]}
              onPress={liveMonitoring ? stopLiveMonitoring : startLiveMonitoring}
            >
              <Text style={styles.liveToggleBtnText}>
                {liveMonitoring ? '⏹ إيقاف المراقبة التلقائية' : '🔴 بدء مراقبة لايف مستمرة'}
              </Text>
            </Pressable>
          </View>
        </View>

        {renderPredictionsList(liveResult)}
      </View>
    );
  };

  const renderAnalyzeTab = () => (
    <View style={styles.sectionContainer}>
      {/* Server Configuration Row for Mobile & Web */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>1. إعداد الاتصال بالخادم (Python API Server)</Text>
        <Text style={styles.helperText}>
          على الويب المحلي استخدم http://localhost:8000، وعلى الجوال أو Vercel ضع رابط الخادم أو استخدم تبويب "كاميرا لايف" الذي يعمل ذاتياً داخل المتصفح.
        </Text>
        <View style={styles.serverRow}>
          <TextInput
            style={styles.serverInput}
            value={apiBase}
            onChangeText={setApiBase}
            placeholder="http://localhost:8000"
            placeholderTextColor="#64748B"
            autoCapitalize="none"
          />
          <Pressable style={styles.secondaryButton} onPress={fetchBackendData}>
            <Text style={styles.secondaryButtonText}>تحديث الاتصال</Text>
          </Pressable>
        </View>
      </View>

      {/* Video Picker & Sample Selector */}
      <View style={styles.card}>
        <Text style={styles.cardTitle}>2. اختيار الفيديو والخوارزمية</Text>
        <View style={styles.actionRow}>
          <Pressable style={styles.primaryButton} onPress={handlePickVideo}>
            <Text style={styles.primaryButtonText}>📁 اختر فيديو MP4 من الجهاز</Text>
          </Pressable>
        </View>

        {samples.length > 0 && (
          <View style={{ marginTop: 14 }}>
            <Text style={styles.subHeaderTitle}>
              أو اختر فيديو مباشرة من مجموعة اختبار MIVIA الرسمية:
            </Text>
            <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.sampleScroll}>
              {samples.map((s) => {
                const isSelected = selectedSample && selectedSample.video_id === s.video_id;
                return (
                  <Pressable
                    key={s.video_id}
                    onPress={() => handleSelectSample(s)}
                    style={[
                      styles.sampleChip,
                      isSelected && styles.sampleChipSelected,
                    ]}
                  >
                    <Text style={styles.sampleChipTitle}>{s.video_id}</Text>
                    <Text
                      style={[
                        styles.sampleChipBadge,
                        { color: s.label === 1 ? '#F87171' : '#34D399' },
                      ]}
                    >
                      {s.label === 1 ? 'Dumping (رمي)' : 'No Dumping (سليم)'}
                    </Text>
                  </Pressable>
                );
              })}
            </ScrollView>
          </View>
        )}

        <Text style={[styles.subHeaderTitle, { marginTop: 16 }]}>اختر الخوارزمية المطلوب تشغيلها:</Text>
        {renderAlgorithmChips()}

        <View style={styles.statusBanner}>
          {analyzing && <ActivityIndicator color="#38BDF8" style={{ marginRight: 10 }} />}
          <Text style={styles.statusBannerText}>{statusText}</Text>
        </View>

        <Pressable
          style={[
            styles.analyzeCtaButton,
            ((!pickedFile && !selectedSample) || analyzing) && styles.buttonDisabled,
          ]}
          disabled={(!pickedFile && !selectedSample) || analyzing}
          onPress={handleRunAnalysis}
        >
          <Text style={styles.analyzeCtaText}>
            {analyzing ? 'جاري التحليل...' : '🚀 بدء تحليل الفيديو (Analyze Video)'}
          </Text>
        </Pressable>
      </View>

      {renderPredictionsList(analysisResult)}

      {/* YOLOv8n Results & Annotated Frame Gallery */}
      {analysisResult && analysisResult.yolo && (
        <View style={styles.card}>
          <Text style={styles.cardTitle}>
            نتائج كاشف المناطق YOLOv8n (12 إطاراً مستخرجاً)
          </Text>
          <View style={styles.yoloSummaryGrid}>
            <View style={styles.yoloStatBox}>
              <Text style={styles.yoloStatValue}>
                {analysisResult.yolo.total_detections > 0 ? 'تم رصد مناطق' : 'لا توجد مناطق'}
              </Text>
              <Text style={styles.yoloStatLabel}>الحالة</Text>
            </View>
            <View style={styles.yoloStatBox}>
              <Text style={styles.yoloStatValue}>{analysisResult.yolo.total_detections}</Text>
              <Text style={styles.yoloStatLabel}>إجمالي المربعات</Text>
            </View>
            <View style={styles.yoloStatBox}>
              <Text style={styles.yoloStatValue}>
                {analysisResult.yolo.frames_with_detections} / {analysisResult.yolo.frames_sampled}
              </Text>
              <Text style={styles.yoloStatLabel}>الإطارات المكتشفة</Text>
            </View>
            <View style={styles.yoloStatBox}>
              <Text style={styles.yoloStatValue}>
                {analysisResult.yolo.strongest_detection
                  ? `${(analysisResult.yolo.strongest_detection.confidence * 100).toFixed(1)}%`
                  : '—'}
              </Text>
              <Text style={styles.yoloStatLabel}>أعلى ثقة (Confidence)</Text>
            </View>
          </View>

          {analysisResult.yolo.frame_results && (
            <ScrollView horizontal showsHorizontalScrollIndicator={false} style={{ marginTop: 14 }}>
              {analysisResult.yolo.frame_results.map((frame) => (
                <View key={frame.frame_number} style={styles.frameCard}>
                  <Image
                    source={{ uri: frame.image_url }}
                    style={styles.frameImage}
                    resizeMode="cover"
                  />
                  <Text style={styles.frameCaption}>
                    إطار #{frame.frame_number} ({frame.timestamp_seconds}s) •{' '}
                    {frame.detections.length} منطقة
                  </Text>
                </View>
              ))}
            </ScrollView>
          )}
        </View>
      )}
    </View>
  );

  const renderEvaluationTab = () => {
    const models = evaluationData?.models || FALLBACK_EVALUATION.models;
    const classifierOrder = ['neural_network', 'movinet_a0', 'optical_flow_svm'];
    const selectedData = models[selectedEvalModel] || models.neural_network;
    const cm = selectedData?.metrics?.confusion_matrix || [
      [0, 0],
      [0, 0],
    ];
    const yolo = models.yolov8n || FALLBACK_EVALUATION.models.yolov8n;

    return (
      <View style={styles.sectionContainer}>
        {/* MIVIA Video Classifiers Comparison Card */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>
            مقارنة خوارزميات تصنيف الفيديو (MIVIA Official Test — 100 Videos)
          </Text>
          <Text style={styles.helperText}>
            التقسيم الموحد: 320 فيديو للتدريب • 80 للتحقق (Validation) • 100 للاختبار الرسمي (50 رمي / 50 سليم). اضغط على أي نموذج لعرض مصفوفة الالتباس (Confusion Matrix) الخاصة به.
          </Text>

          <View style={{ marginTop: 14 }}>
            {classifierOrder.map((slug) => {
              const item = models[slug];
              if (!item) return null;
              const m = item.metrics;
              const meta = MODEL_META[slug];
              const isSelected = selectedEvalModel === slug;

              return (
                <Pressable
                  key={slug}
                  onPress={() => setSelectedEvalModel(slug)}
                  style={[
                    styles.evalModelCard,
                    isSelected && { borderColor: meta.color, borderWidth: 2 },
                  ]}
                >
                  <View style={styles.evalModelHeader}>
                    <View>
                      <Text style={styles.evalModelTitle}>{meta.title}</Text>
                      <Text style={styles.evalModelSubtitle}>{meta.subtitle}</Text>
                    </View>
                    <View style={[styles.badgeBox, { backgroundColor: `${meta.color}25` }]}>
                      <Text style={[styles.badgeBoxText, { color: meta.color }]}>{meta.badge}</Text>
                    </View>
                  </View>

                  <View style={styles.metricsRow}>
                    {[
                      { label: 'Accuracy', val: m.accuracy },
                      { label: 'Precision', val: m.precision },
                      { label: 'Recall', val: m.recall },
                      { label: 'F1 Score', val: m.f1 },
                    ].map((metric) => (
                      <View key={metric.label} style={styles.metricCell}>
                        <Text style={styles.metricValue}>{(metric.val * 100).toFixed(2)}%</Text>
                        <Text style={styles.metricLabel}>{metric.label}</Text>
                        <View style={styles.progressTrack}>
                          <View
                            style={[
                              styles.progressFill,
                              { width: `${Math.min(100, metric.val * 100)}%`, backgroundColor: meta.color },
                            ]}
                          />
                        </View>
                      </View>
                    ))}
                  </View>

                  <View style={styles.metaFooterRow}>
                    <Text style={styles.metaFooterText}>
                      ⏱ التدريب: {Number(item.training_seconds || 0).toFixed(1)}s
                    </Text>
                    <Text style={styles.metaFooterText}>
                      ⚡ الاستدلال: {Number(item.inference_mean_ms_per_video || 0).toFixed(1)} ms/video
                    </Text>
                    <Text style={styles.metaFooterText}>
                      💾 الحجم: {Number(item.model_size_mib || 0).toFixed(2)} MiB
                    </Text>
                  </View>
                </Pressable>
              );
            })}
          </View>
        </View>

        {/* Interactive Confusion Matrix */}
        <View style={styles.card}>
          <Text style={styles.cardTitle}>
            مصفوفة الالتباس (Confusion Matrix) — {MODEL_META[selectedEvalModel]?.title}
          </Text>
          <View style={styles.matrixGrid}>
            <View style={[styles.matrixBox, { borderColor: '#10B981' }]}>
              <Text style={styles.matrixNumber}>{cm[0][0]}</Text>
              <Text style={styles.matrixLabel}>رفض صحيح (TN)</Text>
              <Text style={styles.matrixSub}>Actual: No Dumping • Pred: No Dumping</Text>
            </View>
            <View style={[styles.matrixBox, { borderColor: '#F59E0B' }]}>
              <Text style={styles.matrixNumber}>{cm[0][1]}</Text>
              <Text style={styles.matrixLabel}>إنذار كاذب (FP)</Text>
              <Text style={styles.matrixSub}>Actual: No Dumping • Pred: Dumping</Text>
            </View>
            <View style={[styles.matrixBox, { borderColor: '#EF4444' }]}>
              <Text style={styles.matrixNumber}>{cm[1][0]}</Text>
              <Text style={styles.matrixLabel}>رمي فائت (FN)</Text>
              <Text style={styles.matrixSub}>Actual: Dumping • Pred: No Dumping</Text>
            </View>
            <View style={[styles.matrixBox, { borderColor: '#3B82F6' }]}>
              <Text style={styles.matrixNumber}>{cm[1][1]}</Text>
              <Text style={styles.matrixLabel}>كشف صحيح للرمي (TP)</Text>
              <Text style={styles.matrixSub}>Actual: Dumping • Pred: Dumping</Text>
            </View>
          </View>
        </View>

        {/* YOLOv8n Object Detection Evaluation */}
        {yolo && (
          <View style={styles.card}>
            <Text style={styles.cardTitle}>
              الخوارزمية الرابعة: YOLOv8n — كشف مناطق الرمي (Object Detection)
            </Text>
            <Text style={styles.helperText}>
              الداتا: {yolo.dataset} • التقسيم: {yolo.evaluation_split}. مقاييس الكشف عن المربعات (Bounding Boxes) مستقلة عن مقاييس تصنيف الفيديو الكامل.
            </Text>

            <View style={[styles.metricsRow, { marginTop: 14 }]}>
              {[
                { label: 'Precision', val: yolo.metrics.precision },
                { label: 'Recall', val: yolo.metrics.recall },
                { label: 'mAP@50', val: yolo.metrics.map50 },
                { label: 'mAP@50:95', val: yolo.metrics.map50_95 },
              ].map((m) => (
                <View key={m.label} style={styles.metricCell}>
                  <Text style={styles.metricValue}>{(m.val * 100).toFixed(2)}%</Text>
                  <Text style={styles.metricLabel}>{m.label}</Text>
                  <View style={styles.progressTrack}>
                    <View
                      style={[
                        styles.progressFill,
                        { width: `${Math.min(100, m.val * 100)}%`, backgroundColor: '#8B5CF6' },
                      ]}
                    />
                  </View>
                </View>
              ))}
            </View>
          </View>
        )}
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="light" />
      <ScrollView contentContainerStyle={styles.scrollContent}>
        <View style={styles.maxContainer}>
          {renderHeader()}
          {activeTab === 'analyze'
            ? renderAnalyzeTab()
            : activeTab === 'live'
            ? renderLiveCameraTab()
            : renderEvaluationTab()}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#0B1120',
  },
  scrollContent: {
    paddingVertical: 20,
    paddingHorizontal: 16,
    alignItems: 'center',
  },
  maxContainer: {
    width: '100%',
    maxWidth: 1060,
  },
  header: {
    backgroundColor: '#1E293B',
    borderRadius: 16,
    padding: 20,
    marginBottom: 18,
    borderWidth: 1,
    borderColor: '#334155',
  },
  headerTop: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 12,
  },
  appTitle: {
    color: '#F8FAFC',
    fontSize: 22,
    fontWeight: '800',
  },
  appSubtitle: {
    color: '#94A3B8',
    fontSize: 13,
    marginTop: 4,
  },
  statusPill: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 999,
  },
  statusDot: {
    width: 8,
    height: 8,
    borderRadius: 4,
    marginRight: 8,
  },
  statusPillText: {
    color: '#E2E8F0',
    fontSize: 12,
    fontWeight: '600',
  },
  tabBar: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    marginTop: 18,
    backgroundColor: '#0F172A',
    borderRadius: 12,
    padding: 4,
    gap: 6,
  },
  tabButton: {
    flex: 1,
    minWidth: 140,
    paddingVertical: 11,
    borderRadius: 9,
    alignItems: 'center',
  },
  tabButtonActive: {
    backgroundColor: '#2563EB',
  },
  tabButtonText: {
    color: '#94A3B8',
    fontWeight: '700',
    fontSize: 13,
  },
  tabButtonTextActive: {
    color: '#FFFFFF',
  },
  sectionContainer: {
    gap: 16,
  },
  card: {
    backgroundColor: '#1E293B',
    borderRadius: 16,
    padding: 18,
    borderWidth: 1,
    borderColor: '#334155',
  },
  cardTitle: {
    color: '#F8FAFC',
    fontSize: 17,
    fontWeight: '700',
    marginBottom: 6,
  },
  subHeaderTitle: {
    color: '#CBD5E1',
    fontSize: 13,
    fontWeight: '600',
    marginBottom: 8,
  },
  helperText: {
    color: '#94A3B8',
    fontSize: 13,
    lineHeight: 19,
  },
  serverRow: {
    flexDirection: 'row',
    gap: 10,
    marginTop: 10,
    alignItems: 'center',
  },
  serverInput: {
    flex: 1,
    backgroundColor: '#0F172A',
    color: '#F8FAFC',
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderWidth: 1,
    borderColor: '#334155',
    fontSize: 14,
  },
  primaryButton: {
    backgroundColor: '#2563EB',
    paddingVertical: 12,
    paddingHorizontal: 18,
    borderRadius: 10,
    alignItems: 'center',
  },
  primaryButtonText: {
    color: '#FFFFFF',
    fontWeight: '700',
    fontSize: 14,
  },
  secondaryButton: {
    backgroundColor: '#334155',
    paddingVertical: 11,
    paddingHorizontal: 15,
    borderRadius: 10,
  },
  secondaryButtonText: {
    color: '#F8FAFC',
    fontWeight: '600',
    fontSize: 13,
  },
  actionRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginTop: 8,
  },
  sampleScroll: {
    marginTop: 4,
  },
  sampleChip: {
    backgroundColor: '#0F172A',
    borderWidth: 1,
    borderColor: '#334155',
    borderRadius: 10,
    paddingHorizontal: 14,
    paddingVertical: 9,
    marginRight: 10,
  },
  sampleChipSelected: {
    borderColor: '#38BDF8',
    backgroundColor: 'rgba(56, 189, 248, 0.12)',
  },
  sampleChipTitle: {
    color: '#F8FAFC',
    fontWeight: '700',
    fontSize: 13,
  },
  sampleChipBadge: {
    fontSize: 11,
    fontWeight: '600',
    marginTop: 2,
  },
  filterGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 8,
  },
  filterChip: {
    backgroundColor: '#0F172A',
    borderWidth: 1,
    borderColor: '#334155',
    paddingHorizontal: 13,
    paddingVertical: 9,
    borderRadius: 999,
  },
  filterChipActive: {
    backgroundColor: '#1D4ED8',
    borderColor: '#60A5FA',
  },
  filterChipText: {
    color: '#CBD5E1',
    fontSize: 13,
    fontWeight: '600',
  },
  filterChipTextActive: {
    color: '#FFFFFF',
  },
  statusBanner: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#0F172A',
    padding: 12,
    borderRadius: 10,
    marginTop: 14,
    borderWidth: 1,
    borderColor: '#1E293B',
  },
  statusBannerText: {
    color: '#E2E8F0',
    fontSize: 13,
    flex: 1,
  },
  analyzeCtaButton: {
    backgroundColor: '#10B981',
    paddingVertical: 14,
    paddingHorizontal: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 14,
  },
  buttonDisabled: {
    opacity: 0.5,
  },
  analyzeCtaText: {
    color: '#052E16',
    fontWeight: '800',
    fontSize: 14,
  },
  cameraSplitRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 14,
    marginTop: 12,
  },
  cameraContainer: {
    flex: 1,
    minWidth: 280,
    height: 310,
    borderRadius: 14,
    overflow: 'hidden',
    backgroundColor: '#0F172A',
    borderWidth: 1,
    borderColor: '#334155',
    position: 'relative',
  },
  liveSnapshotContainer: {
    flex: 1,
    minWidth: 280,
    height: 310,
    borderRadius: 14,
    overflow: 'hidden',
    backgroundColor: '#0F172A',
    borderWidth: 1,
    borderColor: '#8B5CF6',
    position: 'relative',
  },
  cameraView: {
    width: '100%',
    height: '100%',
  },
  cameraOverlayBar: {
    position: 'absolute',
    bottom: 10,
    left: 10,
    right: 10,
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    backgroundColor: 'rgba(15, 23, 42, 0.82)',
    paddingHorizontal: 12,
    paddingVertical: 8,
    borderRadius: 10,
  },
  liveIndicator: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  liveIndicatorText: {
    color: '#F8FAFC',
    fontSize: 12,
    fontWeight: '700',
  },
  flipCameraBtn: {
    backgroundColor: '#334155',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 8,
  },
  flipCameraText: {
    color: '#F8FAFC',
    fontSize: 12,
    fontWeight: '600',
  },
  liveButtonsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginTop: 14,
  },
  liveToggleBtn: {
    paddingVertical: 14,
    paddingHorizontal: 18,
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
  },
  liveStartBtn: {
    backgroundColor: '#2563EB',
  },
  liveStopBtn: {
    backgroundColor: '#EF4444',
  },
  liveToggleBtnText: {
    color: '#FFFFFF',
    fontWeight: '800',
    fontSize: 14,
  },
  resultHeaderRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 8,
    marginBottom: 8,
  },
  gtBadge: {
    backgroundColor: 'rgba(56, 189, 248, 0.16)',
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderRadius: 8,
  },
  gtBadgeText: {
    color: '#38BDF8',
    fontWeight: '700',
    fontSize: 12,
  },
  predictionRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    flexWrap: 'wrap',
    backgroundColor: '#0F172A',
    padding: 14,
    borderRadius: 12,
    marginTop: 10,
    borderWidth: 1,
    borderColor: '#334155',
    gap: 10,
  },
  predictionModelName: {
    color: '#F8FAFC',
    fontSize: 15,
    fontWeight: '700',
  },
  predictionKindText: {
    color: '#94A3B8',
    fontSize: 12,
    marginTop: 2,
  },
  predictionStats: {
    alignItems: 'flex-end',
  },
  verdictBadge: {
    paddingHorizontal: 12,
    paddingVertical: 5,
    borderRadius: 8,
    borderWidth: 1,
  },
  verdictText: {
    fontWeight: '800',
    fontSize: 13,
  },
  scoreText: {
    color: '#CBD5E1',
    fontSize: 12,
    marginTop: 5,
  },
  yoloSummaryGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginTop: 8,
  },
  yoloStatBox: {
    flex: 1,
    minWidth: 130,
    backgroundColor: '#0F172A',
    padding: 12,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: '#334155',
    alignItems: 'center',
  },
  yoloStatValue: {
    color: '#F8FAFC',
    fontSize: 16,
    fontWeight: '800',
  },
  yoloStatLabel: {
    color: '#94A3B8',
    fontSize: 12,
    marginTop: 4,
  },
  frameCard: {
    backgroundColor: '#0F172A',
    borderRadius: 10,
    padding: 8,
    marginRight: 12,
    borderWidth: 1,
    borderColor: '#334155',
    width: 220,
  },
  frameImage: {
    width: 204,
    height: 135,
    borderRadius: 8,
    backgroundColor: '#1E293B',
  },
  frameCaption: {
    color: '#CBD5E1',
    fontSize: 12,
    marginTop: 6,
    textAlign: 'center',
  },
  evalModelCard: {
    backgroundColor: '#0F172A',
    borderRadius: 14,
    padding: 16,
    marginBottom: 12,
    borderWidth: 1,
    borderColor: '#334155',
  },
  evalModelHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    flexWrap: 'wrap',
    gap: 8,
  },
  evalModelTitle: {
    color: '#F8FAFC',
    fontSize: 16,
    fontWeight: '800',
  },
  evalModelSubtitle: {
    color: '#94A3B8',
    fontSize: 12,
    marginTop: 2,
  },
  badgeBox: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 999,
  },
  badgeBoxText: {
    fontSize: 12,
    fontWeight: '800',
  },
  metricsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
    marginTop: 14,
  },
  metricCell: {
    flex: 1,
    minWidth: 110,
    backgroundColor: '#1E293B',
    padding: 10,
    borderRadius: 10,
  },
  metricValue: {
    color: '#F8FAFC',
    fontSize: 17,
    fontWeight: '800',
  },
  metricLabel: {
    color: '#94A3B8',
    fontSize: 12,
    marginTop: 2,
  },
  progressTrack: {
    height: 5,
    backgroundColor: '#0F172A',
    borderRadius: 3,
    marginTop: 8,
    overflow: 'hidden',
  },
  progressFill: {
    height: '100%',
    borderRadius: 3,
  },
  metaFooterRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 16,
    marginTop: 12,
    paddingTop: 10,
    borderTopWidth: 1,
    borderTopColor: '#1E293B',
  },
  metaFooterText: {
    color: '#94A3B8',
    fontSize: 12,
  },
  matrixGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 12,
    marginTop: 10,
  },
  matrixBox: {
    flex: 1,
    minWidth: 190,
    backgroundColor: '#0F172A',
    borderRadius: 12,
    padding: 14,
    borderWidth: 1.5,
  },
  matrixNumber: {
    color: '#F8FAFC',
    fontSize: 24,
    fontWeight: '800',
  },
  matrixLabel: {
    color: '#E2E8F0',
    fontSize: 14,
    fontWeight: '700',
    marginTop: 4,
  },
  matrixSub: {
    color: '#64748B',
    fontSize: 11,
    marginTop: 2,
  },
});
