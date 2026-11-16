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
  useWindowDimensions,
} from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import { CameraView, useCameraPermissions } from 'expo-camera';
import { StatusBar } from 'expo-status-bar';
import { frameUri, captureFromResult, retainCaptures, captureDetectionsForFall } from './captureStore';
import MotionTrail from './MotionTrail';
import { yoloDetectionLabel } from './motionGeometry';
import {
  analyzeLiveFramesOnEdge,
  drawDetectionsOnFrame,
  grabWebVideoFrameFast,
  getWebCameraAspectRatio,
  isWebCameraMirrored,
} from './liveEdgeAi';

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
        accuracy: 0.81,
        precision: 0.829787,
        recall: 0.78,
        f1: 0.804124,
        confusion_matrix: [
          [42, 8],
          [11, 39],
        ],
        true_negative: 42,
        false_positive: 8,
        false_negative: 11,
        true_positive: 39,
      },
      training_seconds: 1148.70,
      inference_mean_ms_per_video: null,
      model_size_mib: 7.36,
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
      dataset: 'TACO + reviewed MIVIA crops',
      evaluation_split: 'Validation · 230 images (development)',
      metrics: {
        precision: 0.8338870432,
        recall: 0.4312714777,
        map50: 0.5404895240,
        map50_95: 0.3845741515,
      },
      confidence_threshold: 0.65,
      training_seconds: 2255.81,
      model_size_mib: 5.93,
    },
  },
};

const MODEL_META = {
  neural_network: {
    title: 'Neural Network (MLP)',
    shortTitle: 'MLP Neural Net',
    subtitle: '528 Spatiotemporal + Scene Diffs',
    badge: '74.0% Acc',
    color: '#10B981',
    icon: '🧠',
  },
  movinet_a0: {
    title: 'MoViNet A0',
    shortTitle: 'MoViNet A0',
    subtitle: '3D Video CNN (24 RGB Frames)',
    badge: '81.0% Acc',
    color: '#3B82F6',
    icon: '🎬',
  },
  optical_flow_svm: {
    title: 'Optical Flow + SVM',
    shortTitle: 'Optical Flow SVM',
    subtitle: '320 Farnebäck Grid + RBF SVM',
    badge: '64.0% Acc',
    color: '#F59E0B',
    icon: '🌊',
  },
  yolov8n: {
    title: 'YOLOv8n + tracking',
    shortTitle: 'YOLOv8n Box',
    subtitle: 'Tracked item candidates',
    badge: '54.0% mAP',
    color: '#8B5CF6',
    icon: '🎯',
  },
};

const NAV_ITEMS = [
  { id: 'monitor', label: 'Live monitor', icon: 'camera' },
  { id: 'analysis', label: 'Video analysis', icon: 'video' },
  { id: 'evaluation', label: 'Evaluation', icon: 'chart' },
  { id: 'settings', label: 'Connection', icon: 'settings' },
];

function Icon({ name = 'camera', size = 18, color = 'currentColor' }) {
  if (Platform.OS !== 'web') return <Text style={{ color, fontSize: size }}>•</Text>;
  const paths = {
    camera: 'M14 4l1.5 3H20a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V9a2 2 0 0 1 2-2h4.5L10 4z M16 13a4 4 0 1 1-8 0 4 4 0 0 1 8 0',
    video: 'M15 10l6-4v12l-6-4 M3 5h10a2 2 0 0 1 2 2v10a2 2 0 0 1-2 2H3z',
    chart: 'M4 4v16h17 M9 16v-5 M14 16V7 M19 16v-8',
    settings: 'M4 7h16 M4 17h16 M8 4v6 M16 14v6',
    download: 'M12 3v12 M7 10l5 5 5-5 M4 16v5h16v-5',
    play: 'M7 4l14 8-14 8z',
    stop: 'M6 6h12v12H6z',
    refresh: 'M20 7v5h-5 M4 17v-5h5 M6 7a7 7 0 0 1 12-2l2 2 M18 17a7 7 0 0 1-12 2l-2-2',
    upload: 'M12 16V3 M7 8l5-5 5 5 M4 16v5h16v-5',
    image: 'M3 3h18v18H3z M3 16l5-5 4 4 3-3 6 6 M8 7h.01',
  };
  return React.createElement('svg', { width: size, height: size, viewBox: '0 0 24 24', fill: 'none', stroke: color, strokeWidth: 1.7, strokeLinecap: 'round', strokeLinejoin: 'round', 'aria-hidden': true },
    React.createElement('path', { d: paths[name] || paths.camera }));
}

function Button({ children, onPress, icon, primary, danger, disabled, small }) {
  return (
    <Pressable accessibilityRole="button" onPress={onPress} disabled={disabled}
      style={({ hovered }) => [styles.button, small && styles.buttonSmall, primary && styles.buttonPrimary, danger && styles.buttonDanger, hovered && !disabled && styles.buttonHover, disabled && styles.disabled]}>
      {icon && <Icon name={icon} size={16} color={primary ? '#FFFFFF' : danger ? '#C64545' : '#435169'} />}
      <Text style={[styles.buttonText, primary && styles.buttonTextPrimary, danger && styles.buttonTextDanger]}>{children}</Text>
    </Pressable>
  );
}

function readCaptures() {
  if (Platform.OS !== 'web' || typeof sessionStorage === 'undefined') return [];
  try {
    const saved = JSON.parse(sessionStorage.getItem('wastewatch.captures') || '[]');
    return Array.isArray(saved) ? saved.filter(item => item?.uri && item?.capturedAt).slice(0, 6) : [];
  } catch { return []; }
}

export default function App() {
  const { width: windowWidth } = useWindowDimensions();
  const isDesktop = windowWidth >= 1080;
  const isCompact = windowWidth < 600;

  // Primary mode: 'live' (camera stage) or 'analyze' (MP4 stage)
  const [stageMode, setStageMode] = useState('live');
  const [page, setPage] = useState('monitor');
  const [captures, setCaptures] = useState(readCaptures);
  const [selectedCaptureId, setSelectedCaptureId] = useState(null);
  const [captureBusy, setCaptureBusy] = useState(false);
  const [frameTotal, setFrameTotal] = useState(0);
  const lastAutoCaptureRef = useRef(0);
  const lastModelCaptureRef = useRef(0);
  const lastTrackedFallRef = useRef(null);
  const cameraSessionRef = useRef('');

  const [apiBase, setApiBase] = useState('http://localhost:8000');
  const [selectedChoice, setSelectedChoice] = useState('all');
  const [pickedFile, setPickedFile] = useState(null);
  const [selectedSample, setSelectedSample] = useState(null);
  const [samples, setSamples] = useState([]);
  const [sampleQuery, setSampleQuery] = useState('');
  const [sampleLabel, setSampleLabel] = useState('all');
  const [analyzing, setAnalyzing] = useState(false);
  const [statusText, setStatusText] = useState('Select an MP4 file or an official MIVIA test sample.');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [selectedFramePreview, setSelectedFramePreview] = useState(null);
  const [evaluationData, setEvaluationData] = useState(FALLBACK_EVALUATION);
  const [selectedEvalModel, setSelectedEvalModel] = useState('neural_network');
  const [apiConnected, setApiConnected] = useState(false);
  const [apiLiveResult, setApiLiveResult] = useState(null);
  const [apiLiveBusy, setApiLiveBusy] = useState(false);
  const [apiLiveError, setApiLiveError] = useState('');

  // Live Camera state
  const [permission, requestPermission] = useCameraPermissions();
  const [cameraFacing, setCameraFacing] = useState('back');
  const [cameraReady, setCameraReady] = useState(false);
  const [liveMonitoring, setLiveMonitoring] = useState(false);
  const [liveAnalyzing, setLiveAnalyzing] = useState(false);
  const [liveStatus, setLiveStatus] = useState('Camera ready for live monitoring.');
  const [liveResult, setLiveResult] = useState(null);
  const [bufferedCount, setBufferedCount] = useState(0);
  const [liveFps, setLiveFps] = useState(0);
  const [cameraAspect, setCameraAspect] = useState(16 / 9);
  const [cameraMirrored, setCameraMirrored] = useState(false);

  const cameraRef = useRef(null);
  const scrollRef = useRef(null);
  const frameBufferRef = useRef([]);
  const liveTimerRef = useRef(null);
  const busyLockRef = useRef(false);
  const detectionHoldUntilRef = useRef(0);
  const fpsCounterRef = useRef({ count: 0, lastTs: Date.now() });

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
          if (!selectedSample && sampJson.samples.length > 0) {
            setSelectedSample(sampJson.samples[0]);
          }
        }
      }
    } catch (_err) {
      setApiConnected(false);
    }
  };

  useEffect(() => {
    fetchBackendData();
  }, [cleanBaseUrl]);

  useEffect(() => {
    if (stageMode !== 'live' && liveMonitoring) {
      stopLiveMonitoring();
    }
    return () => {
      if (liveTimerRef.current) clearInterval(liveTimerRef.current);
    };
  }, [stageMode]);

  useEffect(() => {
    if (Platform.OS !== 'web' || stageMode !== 'live') return;
    const timer = setInterval(() => {
      const aspect = getWebCameraAspectRatio();
      if (aspect) {
        setCameraAspect(aspect);
        setCameraMirrored(isWebCameraMirrored());
        clearInterval(timer);
      }
    }, 200);
    return () => clearInterval(timer);
  }, [stageMode, cameraFacing, cameraReady]);

  const getSelectedModelsList = () =>
    selectedChoice === 'all'
      ? ['neural_network', 'movinet_a0', 'optical_flow_svm', 'yolov8n']
      : [selectedChoice];

  const captureSingleFastFrame = async (withSnapshot = true) => {
    if (Platform.OS === 'web') {
      const fastFrame = grabWebVideoFrameFast(withSnapshot);
      if (fastFrame) {
        setCameraMirrored(isWebCameraMirrored());
        return fastFrame;
      }
    }
    if (!cameraRef.current || !cameraReady) return null;
    try {
      const photo = await cameraRef.current.takePictureAsync({
        base64: true,
        quality: 0.5,
        skipProcessing: true,
      });
      if (!photo) return null;
      if (photo.base64) return { uri: `data:image/jpeg;base64,${photo.base64}`, ts: Date.now() };
      if (photo.uri && photo.uri.startsWith('data:')) return { uri: photo.uri, ts: Date.now() };
      return null;
    } catch (_e) {
      return null;
    }
  };

  const runFramesAnalysis = async (framesToAnalyze) => {
    return await analyzeLiveFramesOnEdge(framesToAnalyze, ['neural_network']);
  };

  useEffect(() => {
    if (Platform.OS === 'web' && typeof sessionStorage !== 'undefined') {
      try { sessionStorage.setItem('wastewatch.captures', JSON.stringify(captures)); } catch { /* Keep the in-memory copy if storage is full. */ }
    }
  }, [captures]);

  const addCapture = capture => {
    if (!capture?.uri) return;
    setCaptures(previous => retainCaptures(previous, capture));
    setSelectedCaptureId(capture.id);
  };

  // Keep trained-model requests separate so a slow API cannot freeze camera capture.
  useEffect(() => {
    if (!liveMonitoring || !apiConnected) return;
    const controller = new AbortController();
    let active = true;
    let pending = false;
    const analyzeModels = async () => {
      if (pending || frameBufferRef.current.length < 6) return;
      pending = true;
      setApiLiveBusy(true);
      const frames = frameBufferRef.current.slice(-24).map(frameUri).filter(Boolean);
      const frameTimes = frameBufferRef.current.slice(-24).filter(frame => frameUri(frame)).map(frame => frame.ts);
      const motionAtCapture = lastTrackedFallRef.current;
      const capturedAt = Date.now();
      try {
        const response = await fetch(`${cleanBaseUrl}/api/analyze-live`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ frames, ...(frameTimes.every(Number.isFinite) ? { frame_timestamps: frameTimes, session_id: cameraSessionRef.current } : {}), models: Object.keys(MODEL_META) }),
          signal: controller.signal,
        });
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || 'Live model request failed.');
        if (!active) return;
        setApiLiveResult({ ...result, mode: 'python_api', receivedAt: Date.now() });
        setApiLiveError(result.errors?.join(' · ') || '');
        const now = Date.now();
        const associated = captureDetectionsForFall(result, motionAtCapture, capturedAt);
        if (associated.length && now - lastModelCaptureRef.current > 3000) {
          const annotated = result.yolo_live?.task === 'temporal_release_candidates'
            ? result.yolo_live.annotated_data_uri
            : await drawDetectionsOnFrame(frames[frames.length - 1], associated);
          if (!active) return;
          lastModelCaptureRef.current = now;
          const capture = captureFromResult({ ...result, mode: 'python_api', yolo_live: { ...result.yolo_live, detections: associated, total_detections: associated.length, annotated_data_uri: annotated } }, frames, capturedAt);
          if (capture) addCapture({ ...capture, kind: 'Release candidate' });
        }
      } catch (error) {
        if (active) setApiLiveError(`Live models: ${error.message}`);
      } finally {
        pending = false;
        if (active) setApiLiveBusy(false);
      }
    };
    const timer = setInterval(analyzeModels, 1000);
    return () => { active = false; controller.abort(); clearInterval(timer); setApiLiveBusy(false); };
  }, [liveMonitoring, apiConnected, cleanBaseUrl]);

  const acceptLiveResult = (result, frames) => {
    setLiveResult(result);
    if (result?.yolo_live?.total_detections > 0) {
      lastTrackedFallRef.current = { at: Date.now(), detections: result.yolo_live.detections };
    }
    const capture = captureFromResult(result, frames);
    if (capture && Date.now() - lastAutoCaptureRef.current > 3000) {
      lastAutoCaptureRef.current = Date.now();
      addCapture(capture);
    }
  };

  const handleCaptureSnapshot = async () => {
    if (!cameraReady || captureBusy) return;
    setCaptureBusy(true);
    try {
      const frame = await captureSingleFastFrame(true);
      const uri = frameUri(frame);
      if (!uri) throw new Error('No camera image available yet.');
      const now = Date.now();
      addCapture({ id: String(now), uri, capturedAt: now, kind: 'Manual capture', regions: 0 });
      setLiveStatus('Snapshot saved to this session.');
    } catch (error) { setLiveStatus(`Capture failed: ${error.message}`); }
    finally { setCaptureBusy(false); }
  };

  const navigate = nextPage => {
    if (nextPage !== 'monitor') {
      stopLiveMonitoring();
      setCameraReady(false);
    }
    setPage(nextPage);
    setStageMode(nextPage === 'monitor' ? 'live' : 'analyze');
  };

  const handleCaptureBurstAndAnalyze = async () => {
    if (!cameraReady || liveAnalyzing) return;
    setLiveAnalyzing(true);
    try {
      const burst = [];
      const totalSteps = Platform.OS === 'web' ? 24 : 10;
      const stepDelayMs = Platform.OS === 'web' ? 60 : 140;
      const tStart = Date.now();
      for (let i = 1; i <= totalSteps; i++) {
        if (i % 4 === 1 || i === totalSteps) {
          setLiveStatus(`Capturing camera frames (${i}/${totalSteps})...`);
        }
        const frame = await captureSingleFastFrame(true);
        if (frame) burst.push(frame);
        await new Promise((r) => setTimeout(r, stepDelayMs));
      }
      const elapsedSec = Math.max(0.1, (Date.now() - tStart) / 1000);
      setLiveFps(Math.round(burst.length / elapsedSec));

      if (burst.length === 0) {
        setLiveStatus('Could not capture camera frames.');
        return;
      }
      frameBufferRef.current = burst;
      setBufferedCount(burst.length);
      setLiveStatus('Analyzing the captured sequence...');
      const res = await runFramesAnalysis(burst);
      acceptLiveResult(res, burst);
      const anyDumping = res.actions && res.actions.some((a) => a.prediction === 1);
      const boxCount = res.yolo_live?.total_detections || 0;
      setLiveStatus(
        anyDumping
          ? `Motion alert: ${boxCount} tracked region${boxCount === 1 ? '' : 's'}. Snapshot saved.`
          : 'No motion alert in this sequence.'
      );
    } catch (err) {
      setLiveStatus(`Camera error: ${err.message}`);
    } finally {
      setLiveAnalyzing(false);
    }
  };

  const startLiveMonitoring = async () => {
    if (!cameraReady) {
      setLiveStatus('Waiting for camera sensor...');
      return;
    }
    if (liveTimerRef.current) clearInterval(liveTimerRef.current);
    frameBufferRef.current = [];
    cameraSessionRef.current = `${Date.now()}-${Math.random().toString(36).slice(2)}`;
    lastTrackedFallRef.current = null;
    setApiLiveResult(null);
    setApiLiveError('');
    setFrameTotal(0);
    fpsCounterRef.current = { count: 0, lastTs: Date.now() };
    detectionHoldUntilRef.current = 0;
    setLiveMonitoring(true);
    setLiveStatus('Live auto-monitoring active @ High-FPS...');

    const intervalMs = Platform.OS === 'web' ? 58 : 180;
    let tick = 0;

    liveTimerRef.current = setInterval(async () => {
      if (busyLockRef.current) return;
      busyLockRef.current = true;
      tick++;
      try {
        const withSnap = true; // Preserve every source frame for an event snapshot.
        const frame = await captureSingleFastFrame(withSnap);
        if (frame) {
          setFrameTotal(count => count + 1);
          frameBufferRef.current = [...frameBufferRef.current, frame].slice(-24);
          setBufferedCount(frameBufferRef.current.length);

          fpsCounterRef.current.count += 1;
          const now = Date.now();
          const dt = (now - fpsCounterRef.current.lastTs) / 1000;
          if (dt >= 0.8) {
            setLiveFps(Math.round(fpsCounterRef.current.count / dt));
            fpsCounterRef.current = { count: 0, lastTs: now };
          }

          if (tick % 4 === 0 && frameBufferRef.current.length >= 6) {
            const recentWindow = frameBufferRef.current.slice(-16);
            const res = await runFramesAnalysis(recentWindow);
            const isDump = res.actions && res.actions.some((a) => a.prediction === 1);

            if (isDump) {
              detectionHoldUntilRef.current = Date.now() + 3500;
              acceptLiveResult(res, recentWindow);
              setLiveStatus('Motion alert. A snapshot has been saved for review.');
            } else if (Date.now() > detectionHoldUntilRef.current) {
              setLiveResult(res);
              setLiveStatus('Monitoring camera frames. No current motion alert.');
            }
          }
        }
      } catch (error) {
        setLiveStatus(`Camera analysis error: ${error.message}`);
      } finally {
        busyLockRef.current = false;
      }
    }, intervalMs);
  };

  const stopLiveMonitoring = () => {
    if (liveTimerRef.current) {
      clearInterval(liveTimerRef.current);
      liveTimerRef.current = null;
    }
    busyLockRef.current = false;
    setLiveMonitoring(false);
    setLiveFps(0);
    setLiveResult(null);
    lastTrackedFallRef.current = null;
    setLiveStatus('Live monitoring paused.');
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
        setSelectedFramePreview(null);
        setStageMode('analyze');
        setPage('analysis');
        setStatusText(`Loaded MP4: ${asset.name}`);
      }
    } catch (err) {
      setStatusText(`File error: ${err.message}`);
    }
  };

  const handleSelectSample = (sample) => {
    setSelectedSample(sample);
    setPickedFile(null);
    setAnalysisResult(null);
    setSelectedFramePreview(null);
    setStageMode('analyze');
    setPage('analysis');
    scrollRef.current?.scrollTo({ y: 0, animated: true });
    setStatusText(
      `Selected MIVIA sample: ${sample.video_id} (${
        sample.label === 1 ? 'Ground Truth: Dumping' : 'Ground Truth: No Dumping'
      })`
    );
  };

  const handleRunAnalysis = async () => {
    if (!pickedFile && !selectedSample) {
      setStatusText('Select an MP4 file or a MIVIA sample first.');
      return;
    }

    setAnalyzing(true);
    setAnalysisResult(null);
    setSelectedFramePreview(null);
    const modelsParam = getSelectedModelsList();
    setStatusText('Extracting 24 frames & running selected AI models...');

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
        throw new Error(data.error || 'Server analysis failed');
      }
      setAnalysisResult(data);
      if (data.yolo?.frame_results?.length > 0) {
        const withDet = data.yolo.frame_results.find((f) => f.detections?.length > 0);
        setSelectedFramePreview(withDet || data.yolo.frame_results[0]);
      }
      setApiConnected(true);
      setStatusText(`Analysis complete for ${data.video_name}.`);
    } catch (error) {
      setStatusText(`Analysis failed: ${error.message}. Check the Python API connection.`);
    } finally {
      setAnalyzing(false);
    }
  };

  const modelsEval = evaluationData?.models || FALLBACK_EVALUATION.models;
  const stageAspect = isCompact ? 4 / 3 : 16 / 9;
  const cameraRegion = cameraAspect > stageAspect
    ? { width: '100%', height: `${100 * stageAspect / cameraAspect}%` }
    : { height: '100%', width: `${100 * cameraAspect / stageAspect}%` };
  const selectedCapture = captures.find(item => item.id === selectedCaptureId) || captures[0];
  const liveAlert = (liveResult?.yolo_live?.total_detections || 0) > 0;
  const yoloDetections = liveMonitoring && Date.now() - (apiLiveResult?.receivedAt || 0) < 1200
    ? apiLiveResult?.yolo_live?.detections || [] : [];
  const clock = value => new Date(value).toLocaleTimeString('en-GB', { hour: '2-digit', minute: '2-digit', second: '2-digit' });
  const percent = value => value == null ? '—' : `${(value * 100).toFixed(1)}%`;
  const videoUrl = pickedFile?.uri || (selectedSample ? `${cleanBaseUrl}/api/sample-video/${selectedSample.video_id}` : null);
  const visibleSamples = samples.filter(sample =>
    sample.video_id.toLowerCase().includes(sampleQuery.trim().toLowerCase()) &&
    (sampleLabel === 'all' || String(sample.label) === sampleLabel));

  const downloadCapture = () => {
    if (!selectedCapture || Platform.OS !== 'web') return;
    const link = document.createElement('a');
    link.href = selectedCapture.uri;
    link.download = `wastewatch-${new Date(selectedCapture.capturedAt).toISOString().replace(/[:.]/g, '-')}.jpg`;
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  const capturePanel = () => (
    <View style={[styles.evidenceColumn, isDesktop && { width: 348 }]}>
      <View style={styles.card}>
        <View style={styles.cardHeading}>
          <View><Text style={styles.eyebrow}>LIVE INFERENCE</Text><Text style={styles.cardTitle}>Live algorithms</Text></View>
          {apiLiveBusy && <ActivityIndicator size="small" color="#0B8163" />}
        </View>
        <View style={styles.liveModelGrid}>
          {Object.entries(MODEL_META).map(([id, model]) => {
            const apiAction = apiLiveResult?.actions?.find(action => action.model === id);
            const localAction = !apiConnected && id === 'neural_network' ? liveResult?.actions?.find(action => action.model === id) : null;
            const action = apiAction || localAction;
            const yolo = id === 'yolov8n' ? apiLiveResult?.yolo_live : null;
            const hasResult = Boolean(action || yolo);
            const positive = action?.prediction === 1;
            const status = yolo ? `${yolo.total_detections} CANDIDATES` : hasResult ? positive ? 'ALERT' : 'NO ALERT' : apiConnected || id === 'neural_network' ? liveMonitoring ? 'WAITING' : 'IDLE' : 'API REQUIRED';
            return <View key={id} style={styles.liveModelCard}>
              <Text style={[styles.liveModelBadge, positive && styles.liveModelAlert]}>{status}</Text>
              <Text style={styles.liveModelTitle}>{id === 'neural_network' ? 'Neural Network (MLP)' : model.title}</Text>
              <Text style={styles.liveModelSource}>{apiAction || yolo ? 'Python model' : id === 'neural_network' && !apiConnected ? 'Local MLP + motion' : 'Python API'}</Text>
              <Text style={styles.liveModelScore}>{action ? `Score ${Number(action.score).toFixed(3)} · ${Math.round(action.seconds * 1000)} ms` : yolo ? `${yolo.total_detections} tracked release candidates` : 'Awaiting frames'}</Text>
            </View>;
          })}
        </View>
        <View style={styles.liveModelNote}>
          <Text style={styles.mutedSmall}>{apiConnected ? apiLiveResult ? `Last model update ${clock(apiLiveResult.receivedAt)}. Results refresh while monitoring.` : 'All four trained models run through the connected API while monitoring.' : 'MLP + motion runs locally. Connect the Python API to run MoViNet, SVM and YOLO live.'}</Text>
          {apiLiveError ? <Text style={styles.errorText}>{apiLiveError}</Text> : null}
          {!apiConnected && <Pressable accessibilityRole="button" onPress={() => navigate('settings')}><Text style={styles.liveModelLink}>Connect all four models →</Text></Pressable>}
        </View>
      </View>
      <View style={styles.card}>
        <View style={styles.cardHeading}>
          <View>
            <Text style={styles.eyebrow}>EVIDENCE</Text>
            <Text style={styles.cardTitle}>Latest capture</Text>
          </View>
          <View style={styles.countPill}><Text style={styles.countText}>{String(captures.length).padStart(2, '0')}</Text></View>
        </View>
        <View style={styles.captureImage} testID="capture-preview">
          {selectedCapture ? (
            <Image testID="saved-capture-image" accessibilityLabel={`Saved ${selectedCapture.kind}`} source={{ uri: selectedCapture.uri }} style={styles.imageFill} resizeMode="contain" />
          ) : (
            <View style={styles.captureEmpty}>
              <View style={styles.emptyIcon}><Icon name="image" size={26} color="#7E8EA3" /></View>
              <Text style={styles.emptyTitle}>Your next capture appears here</Text>
              <Text style={styles.emptyCopy}>Alerts save a snapshot automatically. You can also capture a frame at any time.</Text>
            </View>
          )}
        </View>
        {selectedCapture ? (
          <View style={styles.captureMeta}>
            <View style={styles.inlineBetween}>
              <Text style={styles.captureKind}>{selectedCapture.kind}</Text>
              <Text style={styles.time}>{clock(selectedCapture.capturedAt)}</Text>
            </View>
            <Text style={styles.mutedSmall}>{selectedCapture.kind === 'Manual capture' ? 'A frame saved directly from your camera.' : 'A captured candidate for visual review.'}</Text>
            {Platform.OS === 'web' && <Button icon="download" onPress={downloadCapture}>Download image</Button>}
          </View>
        ) : <Text style={[styles.mutedSmall, { padding: 18 }]}>Captures stay visible when the live result changes.</Text>}
      </View>
      <View style={styles.card}>
        <View style={styles.cardHeading}>
          <Text style={styles.smallTitle}>Session captures</Text>
          {captures.length > 0 && <Pressable accessibilityRole="button" onPress={() => { setCaptures([]); setSelectedCaptureId(null); }}><Text style={styles.textLink}>Clear</Text></Pressable>}
        </View>
        {captures.length ? (
          <View style={styles.thumbnailGrid}>
            {captures.map(item => (
              <Pressable key={item.id} accessibilityRole="button" accessibilityLabel={`View ${item.kind} at ${clock(item.capturedAt)}`}
                onPress={() => setSelectedCaptureId(item.id)} style={[styles.thumbnail, selectedCapture?.id === item.id && styles.thumbnailSelected]}>
                <Image source={{ uri: item.uri }} style={styles.thumbnailImage} resizeMode="cover" />
                <Text style={styles.thumbnailTime}>{clock(item.capturedAt)}</Text>
              </Pressable>
            ))}
          </View>
        ) : <Text style={[styles.mutedSmall, { paddingHorizontal: 20, paddingBottom: 20 }]}>No captures yet. Start with “Capture frame”.</Text>}
        <Text style={styles.sessionNote}>Stored in this browser tab. Download an image to keep a copy.</Text>
      </View>
    </View>
  );

  const monitorPage = () => (
    <>
      <View style={styles.pageHeading}>
        <View><Text style={styles.eyebrow}>CAMERA WORKSPACE</Text><Text style={styles.pageTitle}>Live monitor</Text><Text style={styles.pageSubtitle}>Full camera view. Captures kept within reach.</Text></View>
        <View style={styles.modeBadge}><View style={[styles.modeDot, apiConnected && { backgroundColor: '#14A580' }]} /><Text style={styles.modeText}>{apiConnected ? '4 models connected' : 'Browser motion preview'}</Text></View>
      </View>
      <View style={[styles.mainColumns, !isDesktop && styles.stacked]}>
        <View style={styles.monitorColumn}>
          <View style={styles.card}>
            <View style={styles.feedHeader}>
              <View style={styles.inlineRow}>
                <View style={[styles.statusDot, { backgroundColor: liveMonitoring ? '#14A580' : '#9AA7B8' }]} />
                <Text style={styles.feedTitle}>{liveMonitoring ? 'LIVE CAMERA' : 'CAMERA PREVIEW'}</Text>
              </View>
              <Text style={styles.feedMeta}>{liveMonitoring ? `${liveFps} FPS` : cameraReady ? 'Ready when you are' : 'Camera inactive'}</Text>
            </View>
            <View style={[styles.cameraViewport, { aspectRatio: stageAspect }, !permission?.granted && { minHeight: 320 }]} testID="live-camera-frame">
              {Platform.OS === 'web' && React.createElement('style', null,
                '[data-testid="live-camera-frame"] video { object-fit: contain !important; width: 100% !important; height: 100% !important; }')}
              {permission?.granted ? (
                <View style={[styles.cameraRegion, cameraRegion]} testID="camera-image-region">
                  <CameraView ref={cameraRef} facing={cameraFacing} style={styles.imageFill}
                    onCameraReady={() => setCameraReady(true)}
                    onMountError={error => { setCameraReady(false); setLiveStatus(`Camera error: ${error.message}`); }} />
                  {(!yoloDetections.length ? liveResult?.yolo_live?.detections : [])?.map((detection, i) => {
                    const [x1, y1, x2, y2] = detection.norm_box || [0, 0, 0, 0];
                    const left = Platform.OS === 'web' && cameraMirrored ? 1 - x2 : x1;
                    return <React.Fragment key={i}>
                      <MotionTrail trail={detection.trail} mirrored={Platform.OS === 'web' && cameraMirrored} />
                      <View pointerEvents="none" style={[styles.detectionBox, { left: `${left * 100}%`, top: `${y1 * 100}%`, width: `${(x2 - x1) * 100}%`, height: `${(y2 - y1) * 100}%` }]}>
                        <Text style={styles.detectionLabel}>Motion candidate</Text>
                      </View>
                    </React.Fragment>;
                  })}
                  {yoloDetections.map((detection, i) => {
                    const [x1, y1, x2, y2] = detection.norm_box || [0, 0, 0, 0];
                    const left = Platform.OS === 'web' && cameraMirrored ? 1 - x2 : x1;
                    const confirmed = detection.source === 'yolov8n';
                    return <React.Fragment key={`yolo-${i}`}>
                      <MotionTrail trail={detection.trail} mirrored={Platform.OS === 'web' && cameraMirrored} />
                      <View testID="yolo-detection-box" pointerEvents="none" style={[styles.detectionBox, confirmed && styles.yoloBox, { left: `${left * 100}%`, top: `${y1 * 100}%`, width: `${(x2 - x1) * 100}%`, height: `${(y2 - y1) * 100}%` }]}>
                        <Text style={[styles.detectionLabel, confirmed && styles.yoloLabel]}>{yoloDetectionLabel(detection)}</Text>
                      </View>
                    </React.Fragment>;
                  })}
                  <View pointerEvents="none" style={styles.cameraCorner}><Text style={styles.cameraCornerText}>CAM 01</Text></View>
                </View>
              ) : (
                <View style={styles.permissionPanel}>
                  <View style={styles.cameraEmptyIcon}><Icon name="camera" size={35} color="#D4E0EC" /></View>
                  <Text style={styles.permissionTitle}>Your camera, front and center.</Text>
                  <Text style={styles.permissionCopy}>Enable your camera to start monitoring and save a snapshot.</Text>
                  <Button primary icon="camera" onPress={requestPermission}>Enable camera</Button>
                </View>
              )}
            </View>
            <View style={styles.feedControls}>
              <View style={styles.controlGroup}>
                <Button primary={!liveMonitoring} danger={liveMonitoring} icon={liveMonitoring ? 'stop' : 'play'}
                  disabled={!cameraReady || liveAnalyzing} onPress={liveMonitoring ? stopLiveMonitoring : startLiveMonitoring}>
                  {liveMonitoring ? 'Stop monitoring' : 'Start monitoring'}
                </Button>
                <Button icon="camera" disabled={!cameraReady || captureBusy} onPress={handleCaptureSnapshot}>{captureBusy ? 'Capturing…' : 'Capture frame'}</Button>
              </View>
              <Button small icon="refresh" disabled={!cameraReady} onPress={() => { stopLiveMonitoring(); setCameraReady(false); setLiveResult(null); setCameraFacing(value => value === 'back' ? 'front' : 'back'); }}>Switch camera</Button>
            </View>
            <View style={[styles.feedStatus, liveAlert && styles.feedStatusAlert]}>
              <View style={[styles.statusDot, { backgroundColor: liveAlert ? '#D79533' : '#8395A9' }]} />
              <Text style={styles.feedStatusText}>{permission?.granted ? liveStatus : 'Enable your camera to start monitoring.'}</Text>
              {(liveAnalyzing || captureBusy) && <ActivityIndicator size="small" color="#0B8163" />}
            </View>
          </View>
          <View style={styles.sessionBar}>
            <View><Text style={styles.statValue}>{frameTotal.toLocaleString('en-US')}</Text><Text style={styles.statLabel}>Frames captured</Text></View>
            <View style={styles.statDivider} />
            <View><Text style={styles.statValue}>{bufferedCount}<Text style={styles.statSuffix}> / 24</Text></Text><Text style={styles.statLabel}>Current buffer</Text></View>
            <View style={styles.statDivider} />
            <View><Text style={styles.statValue}>{captures.length}</Text><Text style={styles.statLabel}>Saved captures</Text></View>
          </View>
          <View style={styles.noteCard}>
            <View style={styles.inlineBetween}><Text style={styles.smallTitle}>Camera tools</Text><Text style={styles.noteBadge}>OPTIONAL</Text></View>
            <Text style={styles.mutedSmall}>Analyze a short camera sequence on demand. Saved images stay in the capture panel.</Text>
            <View style={styles.controlGroup}>
              <Button small disabled={!cameraReady || liveAnalyzing} onPress={handleCaptureBurstAndAnalyze}>{liveAnalyzing ? 'Analyzing sequence…' : 'Analyze 24-frame burst'}</Button>
            </View>
          </View>
          <Text style={styles.footnote}>Amber: tracked release candidate and motion path. Blue: a tracked candidate also recognized by YOLO. Percentages are object scores, not the probability of throwing.</Text>
        </View>
        {capturePanel()}
      </View>
    </>
  );

  const analysisPage = () => (
    <>
      <View style={styles.pageHeading}><View><Text style={styles.eyebrow}>SAVED MODEL INFERENCE</Text><Text style={styles.pageTitle}>Video analysis</Text><Text style={styles.pageSubtitle}>Upload a video or choose an official test sample.</Text></View>
        <View style={styles.modeBadge}><View style={[styles.modeDot, { backgroundColor: apiConnected ? '#14A580' : '#9AA7B8' }]} /><Text style={styles.modeText}>{apiConnected ? 'Python API connected' : 'Python API required'}</Text></View>
      </View>
      <View style={[styles.mainColumns, !isDesktop && styles.stacked]}>
        <View style={styles.monitorColumn}>
          <View style={styles.card}>
            <View style={styles.feedHeader}><Text style={styles.feedTitle}>{pickedFile?.name || selectedSample?.video_id || 'VIDEO WORKSPACE'}</Text><Text style={styles.feedMeta}>MP4</Text></View>
            <View style={[styles.cameraViewport, { aspectRatio: 16 / 9 }, !videoUrl && { minHeight: 300 }]}>
              {Platform.OS === 'web' && videoUrl ? React.createElement('video', { src: videoUrl, controls: true, playsInline: true, style: { width: '100%', height: '100%', objectFit: 'contain' } }) :
                <View style={styles.permissionPanel}><Icon name="video" size={38} color="#B8C9DB" /><Text style={styles.permissionTitle}>Choose a video to begin</Text><Text style={styles.permissionCopy}>Your results and annotated frames will appear alongside the video.</Text><Button primary icon="upload" onPress={handlePickVideo}>Choose MP4</Button></View>}
            </View>
            <View style={styles.feedControls}><View style={styles.controlGroup}><Button primary icon="play" disabled={(!pickedFile && !selectedSample) || analyzing} onPress={handleRunAnalysis}>{analyzing ? 'Analyzing…' : 'Analyze video'}</Button><Button icon="upload" onPress={handlePickVideo}>Choose MP4</Button></View></View>
            <View style={styles.feedStatus}><Text style={styles.feedStatusText}>{statusText}</Text>{analyzing && <ActivityIndicator size="small" color="#0B8163" />}</View>
          </View>
          <View style={styles.noteCard}><Text style={styles.smallTitle}>Models to run</Text><View style={styles.modelChoices}>
            {[{ id: 'all', title: 'All four models' }, ...Object.entries(MODEL_META).map(([id, item]) => ({ id, title: item.title }))].map(item => <Pressable accessibilityRole="button" key={item.id} onPress={() => setSelectedChoice(item.id)} style={[styles.choice, selectedChoice === item.id && styles.choiceActive]}><Text style={[styles.choiceText, selectedChoice === item.id && styles.choiceTextActive]}>{item.title}</Text></Pressable>)}
          </View></View>
          <View style={styles.card}><View style={styles.cardHeading}><Text style={styles.smallTitle}>Official MIVIA samples</Text><Text style={styles.mutedSmall}>{visibleSamples.length} of {samples.length} videos</Text></View>
            {samples.length ? <>
              <View style={styles.sampleFilters}>
                <TextInput value={sampleQuery} onChangeText={setSampleQuery} placeholder="Search by video name…" accessibilityLabel="Search sample videos" autoCapitalize="none" autoCorrect={false} style={[styles.input, { flexGrow: 1, minWidth: 180 }]} />
                <View style={styles.modelChoices}>{[{ id: 'all', title: 'All' }, { id: '1', title: 'Dumping' }, { id: '0', title: 'No dumping' }].map(filter => <Pressable key={filter.id} accessibilityRole="button" onPress={() => setSampleLabel(filter.id)} style={[styles.choice, sampleLabel === filter.id && styles.choiceActive]}><Text style={[styles.choiceText, sampleLabel === filter.id && styles.choiceTextActive]}>{filter.title}</Text></Pressable>)}</View>
              </View>
              <View style={styles.sampleGrid}>{visibleSamples.map(sample => <Pressable accessibilityRole="button" key={sample.video_id} onPress={() => handleSelectSample(sample)} style={[styles.sample, { width: isCompact ? '47%' : 150 }, selectedSample?.video_id === sample.video_id && styles.sampleSelected]}><Icon name="video" size={17} color={sample.label === 1 ? '#B4803E' : '#58977F'} /><Text style={styles.captureKind}>{sample.video_id}</Text><Text style={styles.mutedSmall}>{sample.label === 1 ? 'Dumping' : 'No dumping'}</Text></Pressable>)}</View>
              {!visibleSamples.length && <Text style={[styles.mutedSmall, { padding: 20 }]}>No videos match your search.</Text>}
            </> : <View style={styles.inner}><Text style={styles.mutedSmall}>Connect the Python API to browse the official test videos.</Text><Button small onPress={() => navigate('settings')}>Open connection settings</Button></View>}
          </View>
        </View>
        <View style={[styles.evidenceColumn, isDesktop && { width: 348 }]}>
          <View style={styles.card}><View style={styles.cardHeading}><Text style={styles.cardTitle}>Analysis results</Text></View>
            {analysisResult ? <View style={styles.inner}>
              {analysisResult.actions?.map(action => <View style={styles.resultRow} key={action.model}><Text style={styles.captureKind}>{MODEL_META[action.model]?.title || action.label_title}</Text><Text style={[styles.resultVerdict, action.prediction === 1 && { color: '#B4651A' }]}>{action.prediction === 1 ? 'Predicted dumping' : 'Predicted no dumping'}</Text><Text style={styles.mutedSmall}>Score {Number(action.score).toFixed(3)} · {Math.round(action.seconds * 1000)} ms</Text></View>)}
              {analysisResult.yolo && <Text style={styles.mutedSmall}>Release tracking: {analysisResult.yolo.total_detections} candidates · {analysisResult.yolo.frames_scanned || analysisResult.yolo.frames_sampled} frames checked. Percentages appear only when YOLO confirms the object.</Text>}
              {analysisResult.errors?.map((error, i) => <Text key={i} style={styles.errorText}>{error}</Text>)}
            </View> : <View style={styles.inner}><Text style={styles.mutedSmall}>Run an analysis to see the saved models’ predictions here.</Text></View>}
          </View>
          <View style={styles.card}><View style={styles.cardHeading}><Text style={styles.smallTitle}>Annotated frames</Text></View>
            {selectedFramePreview ? <><View style={styles.captureImage}><Image source={{ uri: selectedFramePreview.image_url }} style={styles.imageFill} resizeMode="contain" /></View><View style={styles.inner}><Text style={styles.mutedSmall}>Frame {selectedFramePreview.frame_number} · {selectedFramePreview.timestamp_seconds}s</Text></View></> : <View style={styles.inner}><Text style={styles.mutedSmall}>Detection frames appear after video analysis.</Text></View>}
            <ScrollView horizontal contentContainerStyle={styles.sampleStrip}>{analysisResult?.yolo?.frame_results?.map(frame => <Pressable key={frame.frame_number} onPress={() => setSelectedFramePreview(frame)}><Image source={{ uri: frame.image_url }} style={styles.frameThumbnail} resizeMode="cover" /></Pressable>)}</ScrollView>
          </View>
        </View>
      </View>
    </>
  );

  const evaluationPage = () => {
    const selected = modelsEval[selectedEvalModel] || FALLBACK_EVALUATION.models.neural_network;
    const matrix = selected?.metrics?.confusion_matrix || [[0, 0], [0, 0]];
    return <>
      <View style={styles.pageHeading}><View><Text style={styles.eyebrow}>RECORDED EXPERIMENTS</Text><Text style={styles.pageTitle}>Model evaluation</Text><Text style={styles.pageSubtitle}>Saved benchmarks, separate from the current camera session.</Text></View><Button icon="refresh" onPress={fetchBackendData}>Refresh results</Button></View>
      <View style={styles.card}><View style={styles.cardHeading}><View><Text style={styles.cardTitle}>Video classification</Text><Text style={styles.mutedSmall}>MIVIA · 100 official test videos · 50 dumping / 50 no dumping</Text></View></View>
        <ScrollView horizontal><View style={styles.metricTable}>
          <View style={[styles.metricTableRow, styles.metricTableHeader]}>{['Model', 'Accuracy', 'Precision', 'Recall', 'F1'].map((title, i) => <Text key={title} style={[styles.tableLabel, i === 0 ? styles.modelCell : styles.metricCell]}>{title}</Text>)}</View>
          {['neural_network', 'movinet_a0', 'optical_flow_svm'].map(slug => { const metrics = (modelsEval[slug] || FALLBACK_EVALUATION.models[slug]).metrics; return <Pressable key={slug} onPress={() => setSelectedEvalModel(slug)} style={[styles.metricTableRow, selectedEvalModel === slug && styles.selectedTableRow]}><Text style={[styles.modelCell, styles.tableModel]}>{MODEL_META[slug].title}</Text>{['accuracy', 'precision', 'recall', 'f1'].map(key => <Text style={styles.metricCell} key={key}>{percent(metrics[key])}</Text>)}</Pressable>; })}
        </View></ScrollView>
      </View>
      <View style={[styles.mainColumns, !isDesktop && styles.stacked, { marginTop: 24 }]}>
        <View style={[styles.card, { flex: 1 }]}><View style={styles.cardHeading}><View><Text style={styles.cardTitle}>Confusion matrix</Text><Text style={styles.mutedSmall}>{MODEL_META[selectedEvalModel].title} · select a model above</Text></View></View><View style={styles.matrixGrid}>{[{ label: 'True negative', value: matrix[0][0], tone: '#0B8163' }, { label: 'False positive', value: matrix[0][1], tone: '#B46D21' }, { label: 'False negative', value: matrix[1][0], tone: '#BF4D4D' }, { label: 'True positive', value: matrix[1][1], tone: '#367AA9' }].map(item => <View style={styles.matrixCell} key={item.label}><Text style={[styles.matrixNumber, { color: item.tone }]}>{item.value}</Text><Text style={styles.mutedSmall}>{item.label}</Text></View>)}</View></View>
        <View style={[styles.card, { flex: 1 }]}><View style={styles.cardHeading}><View><Text style={styles.cardTitle}>YOLOv8n detection</Text><Text style={styles.mutedSmall}>{(modelsEval.yolov8n || FALLBACK_EVALUATION.models.yolov8n).dataset} · {(modelsEval.yolov8n || FALLBACK_EVALUATION.models.yolov8n).evaluation_split}</Text></View></View><View style={styles.detectionMetrics}>{[['Precision', 'precision'], ['Recall', 'recall'], ['mAP@50', 'map50'], ['mAP@50:95', 'map50_95']].map(([label, key]) => <View style={styles.detectionMetric} key={key}><Text style={styles.statValue}>{percent((modelsEval.yolov8n || FALLBACK_EVALUATION.models.yolov8n).metrics[key])}</Text><Text style={styles.statLabel}>{label}</Text></View>)}</View><Text style={[styles.footnote, { padding: 20 }]}>Object detection is not proof of throwing. Precision and recall use the selected score threshold; mAP summarizes all score thresholds. These validation images were used during model selection.</Text></View>
      </View>
      <Text style={styles.footnote}>MIVIA results are exploratory because the official test set was reused during protocol development. These benchmarks are not live confidence scores.</Text>
    </>;
  };

  const settingsPage = () => (
    <>
      <View style={styles.pageHeading}><View><Text style={styles.eyebrow}>WORKSPACE SETTINGS</Text><Text style={styles.pageTitle}>Connection</Text><Text style={styles.pageSubtitle}>Connect the Python API for trained-model inference and test videos.</Text></View></View>
      <View style={[styles.card, { maxWidth: 760 }]}><View style={styles.cardHeading}><Text style={styles.cardTitle}>Python API</Text><View style={styles.modeBadge}><View style={[styles.modeDot, { backgroundColor: apiConnected ? '#14A580' : '#9AA7B8' }]} /><Text style={styles.modeText}>{apiConnected ? 'Connected' : 'Not connected'}</Text></View></View>
        <View style={styles.inner}><Text style={styles.fieldLabel}>SERVER ADDRESS</Text><TextInput value={apiBase} onChangeText={setApiBase} autoCapitalize="none" autoCorrect={false} placeholder="http://localhost:8000" style={styles.input} accessibilityLabel="Python API server address" />
          <View style={styles.controlGroup}><Button primary icon="refresh" onPress={fetchBackendData}>Connect API</Button><Button onPress={() => navigate('monitor')}>Back to camera</Button></View>
          <View style={styles.explanation}><Text style={styles.smallTitle}>Run locally on your Mac</Text><Text selectable style={styles.code}>.venv/bin/python api.py --port 8000</Text><Text style={styles.mutedSmall}>The API runs the saved MLP, MoViNet, SVM and YOLO checkpoints for video analysis. The browser camera preview remains available without this connection.</Text></View>
        </View>
      </View>
    </>
  );

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="dark" />
      {Platform.OS === 'web' && React.createElement('style', null,
        'html,body,#root{margin:0;min-height:100%;background:#F5F7FA}*{box-sizing:border-box}button,a,input{font:inherit}::selection{background:#D2EEE4}button:focus-visible,[role=button]:focus-visible{outline:3px solid #82CDB7;outline-offset:3px}')}
      <ScrollView ref={scrollRef} contentContainerStyle={styles.outerScroll}>
        <View style={[styles.header, isCompact && { paddingHorizontal: 18 }]}>
          <View style={styles.brand}><View style={styles.brandIcon}><Icon name="camera" size={21} color="#0A8063" /></View><View><Text style={styles.brandName}>wastewatch<Text style={styles.brandDot}>.</Text></Text><Text style={styles.brandSubtitle}>MONITORING WORKSPACE</Text></View></View>
          <View style={[styles.navigation, !isDesktop && { width: '100%' }]}>
            {NAV_ITEMS.map(item => <Pressable accessibilityRole="tab" accessibilityState={{ selected: page === item.id }} key={item.id} onPress={() => navigate(item.id)} style={[styles.navItem, page === item.id && styles.navItemActive]}><Icon name={item.icon} size={16} color={page === item.id ? '#08775A' : '#738096'} /><Text style={[styles.navText, page === item.id && styles.navTextActive]}>{item.label}</Text></Pressable>)}
          </View>
          {isDesktop && <View style={styles.headerSession}><View style={styles.sessionIndicator} /><Text style={styles.headerSessionText}>Local session</Text></View>}
        </View>
        <View style={[styles.page, isCompact && { paddingHorizontal: 18, paddingTop: 28 }]}>{page === 'monitor' ? monitorPage() : page === 'analysis' ? analysisPage() : page === 'evaluation' ? evaluationPage() : settingsPage()}</View>
        <View style={styles.footer}><Text style={styles.footerText}>Wastewatch / Illegal Waste Dumping Detection</Text><Text style={styles.footerText}>Capture. Review. Keep the evidence.</Text></View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: { flex: 1, backgroundColor: '#F5F7FA' },
  outerScroll: { flexGrow: 1 },
  header: { paddingHorizontal: 28, paddingVertical: 20, backgroundColor: '#FFFFFF', borderBottomWidth: 1, borderBottomColor: '#E4E9EF', flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 22, flexWrap: 'wrap' },
  brand: { flexDirection: 'row', alignItems: 'center', gap: 11 },
  brandIcon: { height: 40, width: 40, borderRadius: 12, backgroundColor: '#E7F4EE', alignItems: 'center', justifyContent: 'center' },
  brandName: { color: '#17283C', fontSize: 23, fontWeight: '800', letterSpacing: -1 },
  brandDot: { color: '#0A8063' },
  brandSubtitle: { fontSize: 8, fontWeight: '700', color: '#8492A4', letterSpacing: 1.9, marginTop: 2 },
  navigation: { flexDirection: 'row', flexWrap: 'wrap', gap: 4 },
  navItem: { flexDirection: 'row', alignItems: 'center', gap: 8, paddingHorizontal: 14, paddingVertical: 11, borderRadius: 9 },
  navItemActive: { backgroundColor: '#E8F4EE' },
  navText: { color: '#6B788D', fontSize: 13, fontWeight: '600' },
  navTextActive: { color: '#08775A' },
  headerSession: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  sessionIndicator: { width: 7, height: 7, borderRadius: 4, backgroundColor: '#4CAB8D' },
  headerSessionText: { color: '#7C899B', fontSize: 12 },
  page: { width: '100%', maxWidth: 1440, alignSelf: 'center', paddingHorizontal: 28, paddingTop: 36, paddingBottom: 45 },
  pageHeading: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', marginBottom: 25, gap: 16, flexWrap: 'wrap' },
  eyebrow: { fontSize: 9, fontWeight: '700', color: '#8B98A9', letterSpacing: 1.7, marginBottom: 8 },
  pageTitle: { fontSize: 31, fontWeight: '700', color: '#17283C', letterSpacing: -1.2 },
  pageSubtitle: { fontSize: 13, color: '#7E8B9E', marginTop: 8, lineHeight: 20 },
  modeBadge: { flexDirection: 'row', alignItems: 'center', gap: 7, paddingVertical: 8, paddingHorizontal: 11, borderRadius: 7, backgroundColor: '#ECF0F4' },
  modeDot: { width: 6, height: 6, borderRadius: 3, backgroundColor: '#7E90A4' },
  modeText: { color: '#64758A', fontSize: 10, fontWeight: '600' },
  mainColumns: { flexDirection: 'row', alignItems: 'flex-start', gap: 24 },
  stacked: { flexDirection: 'column', alignItems: 'stretch' },
  monitorColumn: { flex: 1, minWidth: 0, gap: 18 },
  evidenceColumn: { gap: 18, alignSelf: 'stretch' },
  liveModelGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10, paddingHorizontal: 16, paddingBottom: 16 },
  liveModelCard: { flexBasis: '46%', flexGrow: 1, padding: 12, borderRadius: 8, backgroundColor: '#F4F7FA', gap: 7 },
  liveModelBadge: { fontSize: 8, fontWeight: '700', letterSpacing: 0.5, color: '#738399' },
  liveModelAlert: { color: '#B46520' },
  liveModelTitle: { fontSize: 12, fontWeight: '700', color: '#2F425B', lineHeight: 17 },
  liveModelSource: { fontSize: 9, color: '#7C8EA3' },
  liveModelScore: { fontSize: 10, color: '#516982', fontVariant: ['tabular-nums'] },
  liveModelNote: { padding: 17, paddingTop: 0, gap: 8 },
  liveModelLink: { color: '#0A8063', fontSize: 11, fontWeight: '600' },
  card: { backgroundColor: '#FFFFFF', borderWidth: 1, borderColor: '#E2E7ED', borderRadius: 13, overflow: 'hidden' },
  cardHeading: { paddingHorizontal: 20, paddingVertical: 19, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 12 },
  cardTitle: { color: '#203248', fontSize: 17, fontWeight: '700', letterSpacing: -0.4 },
  smallTitle: { color: '#26384E', fontSize: 13, fontWeight: '700' },
  feedHeader: { paddingHorizontal: 21, paddingVertical: 16, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 12 },
  inlineRow: { flexDirection: 'row', alignItems: 'center', gap: 8 },
  inlineBetween: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between', gap: 8 },
  statusDot: { width: 6, height: 6, borderRadius: 3 },
  feedTitle: { color: '#52647B', fontSize: 10, letterSpacing: 1.1, fontWeight: '700' },
  feedMeta: { color: '#8B98A8', fontSize: 10 },
  cameraViewport: { width: '100%', backgroundColor: '#111D2C', position: 'relative', overflow: 'hidden', alignItems: 'center', justifyContent: 'center' },
  cameraRegion: { position: 'relative' },
  imageFill: { width: '100%', height: '100%' },
  cameraCorner: { position: 'absolute', top: 18, left: 18, paddingVertical: 5, paddingHorizontal: 8, borderRadius: 4, backgroundColor: 'rgba(8,19,30,0.50)' },
  cameraCornerText: { color: '#D4DFE9', fontSize: 9, fontWeight: '700', letterSpacing: 1 },
  permissionPanel: { flex: 1, justifyContent: 'center', alignItems: 'center', padding: 28, gap: 15 },
  cameraEmptyIcon: { height: 70, width: 70, borderRadius: 22, backgroundColor: '#213246', alignItems: 'center', justifyContent: 'center', marginBottom: 4 },
  permissionTitle: { color: '#F3F6FA', fontSize: 21, fontWeight: '600', textAlign: 'center', letterSpacing: -0.6 },
  permissionCopy: { color: '#8EA2B8', fontSize: 12, textAlign: 'center', lineHeight: 20, maxWidth: 320, marginBottom: 5 },
  detectionBox: { position: 'absolute', borderWidth: 2, borderColor: '#F2AE4E', backgroundColor: 'rgba(242,174,78,0.08)' },
  detectionLabel: { position: 'absolute', top: -22, left: -2, backgroundColor: '#F2AE4E', color: '#302414', paddingHorizontal: 6, paddingVertical: 3, fontSize: 9, fontWeight: '700' },
  yoloBox: { borderColor: '#60A5FA', backgroundColor: 'rgba(96,165,250,0.08)' },
  yoloLabel: { backgroundColor: '#60A5FA', color: '#10243F' },
  feedControls: { paddingHorizontal: 18, paddingVertical: 17, gap: 10, flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap' },
  controlGroup: { flexDirection: 'row', flexWrap: 'wrap', gap: 8, alignItems: 'center' },
  button: { minHeight: 39, paddingHorizontal: 14, paddingVertical: 11, borderWidth: 1, borderColor: '#DAE2E9', borderRadius: 7, flexDirection: 'row', gap: 8, alignItems: 'center', justifyContent: 'center', backgroundColor: '#FFFFFF' },
  buttonSmall: { minHeight: 32, paddingVertical: 8, paddingHorizontal: 10 },
  buttonPrimary: { backgroundColor: '#0A8063', borderColor: '#0A8063' },
  buttonDanger: { backgroundColor: '#FFF7F6', borderColor: '#EBC9C6' },
  buttonHover: { opacity: 0.84 },
  buttonText: { color: '#46566C', fontSize: 11, fontWeight: '600' },
  buttonTextPrimary: { color: '#FFFFFF' },
  buttonTextDanger: { color: '#BE4E48' },
  disabled: { opacity: 0.45 },
  feedStatus: { borderTopWidth: 1, borderTopColor: '#EDF0F4', paddingHorizontal: 19, paddingVertical: 13, backgroundColor: '#FBFCFD', flexDirection: 'row', alignItems: 'center', gap: 9 },
  feedStatusAlert: { backgroundColor: '#FFFBF1' },
  feedStatusText: { fontSize: 10, color: '#8190A4', flex: 1, lineHeight: 17 },
  countPill: { height: 26, minWidth: 30, alignItems: 'center', justifyContent: 'center', backgroundColor: '#EEF3F6', borderRadius: 6 },
  countText: { color: '#6D7F93', fontSize: 11, fontWeight: '600' },
  captureImage: { backgroundColor: '#EEF2F5', width: '100%', aspectRatio: 4 / 3, borderTopWidth: 1, borderBottomWidth: 1, borderColor: '#EDF0F4' },
  captureEmpty: { flex: 1, alignItems: 'center', justifyContent: 'center', padding: 25 },
  emptyIcon: { marginBottom: 15 },
  emptyTitle: { fontSize: 12, fontWeight: '600', color: '#6C7F94', textAlign: 'center', marginBottom: 8 },
  emptyCopy: { fontSize: 11, lineHeight: 19, color: '#96A3B3', textAlign: 'center', maxWidth: 240 },
  captureMeta: { padding: 17, gap: 12 },
  captureKind: { color: '#4B5E76', fontSize: 12, fontWeight: '600' },
  time: { color: '#97A3B2', fontSize: 10, fontVariant: ['tabular-nums'] },
  mutedSmall: { color: '#8B98A9', fontSize: 11, lineHeight: 19 },
  thumbnailGrid: { paddingHorizontal: 16, paddingBottom: 16, flexDirection: 'row', flexWrap: 'wrap', gap: 9 },
  thumbnail: { width: 92, borderRadius: 7, overflow: 'hidden', borderWidth: 2, borderColor: 'transparent', backgroundColor: '#F0F3F6' },
  thumbnailSelected: { borderColor: '#0A8063' },
  thumbnailImage: { width: '100%', height: 61 },
  thumbnailTime: { color: '#8B98A9', fontSize: 8, textAlign: 'center', paddingVertical: 5 },
  sessionNote: { borderTopWidth: 1, borderTopColor: '#EDF0F4', paddingHorizontal: 18, paddingVertical: 12, color: '#A1ACB9', fontSize: 9, lineHeight: 16 },
  textLink: { color: '#8997A8', fontSize: 10 },
  sessionBar: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-around', backgroundColor: '#ECF1F5', paddingHorizontal: 16, paddingVertical: 21, borderRadius: 10 },
  statValue: { color: '#2F425B', fontSize: 22, fontWeight: '600', letterSpacing: -0.5 },
  statSuffix: { color: '#9CA9B8', fontSize: 14, fontWeight: '400' },
  statLabel: { color: '#95A2B2', fontSize: 9, marginTop: 6 },
  statDivider: { height: 29, width: 1, backgroundColor: '#D8E1E9' },
  noteCard: { padding: 20, gap: 13, borderWidth: 1, borderColor: '#E3E8EE', borderRadius: 10, backgroundColor: '#F8FAFC' },
  noteBadge: { color: '#A4AFBC', fontSize: 8, letterSpacing: 1.2 },
  footnote: { color: '#9BA8B8', fontSize: 10, lineHeight: 18, marginTop: 8 },
  footer: { marginTop: 'auto', paddingHorizontal: 28, paddingVertical: 19, borderTopWidth: 1, borderTopColor: '#E6EBF1', flexDirection: 'row', justifyContent: 'space-between', gap: 12, flexWrap: 'wrap' },
  footerText: { color: '#A6B0BD', fontSize: 9 },
  modelChoices: { flexDirection: 'row', flexWrap: 'wrap', gap: 7 },
  choice: { borderWidth: 1, borderColor: '#DFE6ED', backgroundColor: '#FFFFFF', paddingHorizontal: 11, paddingVertical: 8, borderRadius: 6 },
  choiceActive: { borderColor: '#9FCFBE', backgroundColor: '#EAF5EF' },
  choiceText: { color: '#8C99AA', fontSize: 10 },
  choiceTextActive: { color: '#167958' },
  inner: { paddingHorizontal: 20, paddingBottom: 20, gap: 14 },
  sampleStrip: { paddingHorizontal: 18, paddingBottom: 18, gap: 9 },
  sampleFilters: { paddingHorizontal: 18, paddingBottom: 18, gap: 12, flexDirection: 'row', flexWrap: 'wrap', alignItems: 'center' },
  sampleGrid: { flexDirection: 'row', flexWrap: 'wrap', gap: 10, paddingHorizontal: 18, paddingBottom: 18 },
  sample: { borderWidth: 1, borderColor: '#E4E9EF', padding: 13, borderRadius: 8, width: 130, gap: 6 },
  sampleSelected: { borderColor: '#93C6B2', backgroundColor: '#F1F8F4' },
  resultRow: { paddingBottom: 14, gap: 6, borderBottomWidth: 1, borderBottomColor: '#EBF0F5' },
  resultVerdict: { fontSize: 11, color: '#167958' },
  errorText: { color: '#BA5650', fontSize: 11, lineHeight: 18 },
  frameThumbnail: { height: 62, width: 88, borderRadius: 5 },
  metricTable: { minWidth: 720, width: '100%' },
  metricTableRow: { flexDirection: 'row', borderTopWidth: 1, borderTopColor: '#E9EEF3', paddingHorizontal: 22, paddingVertical: 18 },
  metricTableHeader: { backgroundColor: '#F9FBFC' },
  tableLabel: { color: '#91A0B1', fontSize: 10, fontWeight: '600' },
  modelCell: { width: 260 },
  metricCell: { width: 110, color: '#60738B', fontSize: 12 },
  tableModel: { color: '#41566F', fontSize: 12, fontWeight: '600' },
  selectedTableRow: { backgroundColor: '#F0F8F3' },
  matrixGrid: { flexDirection: 'row', flexWrap: 'wrap', padding: 18, gap: 12 },
  matrixCell: { width: '46%', backgroundColor: '#F4F7FA', borderRadius: 8, padding: 20, gap: 6 },
  matrixNumber: { fontSize: 29, fontWeight: '600' },
  detectionMetrics: { flexDirection: 'row', flexWrap: 'wrap', paddingHorizontal: 21, paddingBottom: 10, gap: 20 },
  detectionMetric: { width: '42%', paddingVertical: 12 },
  fieldLabel: { color: '#8F9DAE', fontSize: 9, letterSpacing: 1.2, fontWeight: '600' },
  input: { backgroundColor: '#F7F9FB', borderWidth: 1, borderColor: '#DDE5ED', borderRadius: 7, padding: 14, color: '#4B627D', fontSize: 13 },
  explanation: { marginTop: 15, gap: 14, paddingTop: 22, borderTopWidth: 1, borderTopColor: '#E9EEF3' },
  code: { backgroundColor: '#F1F5F8', color: '#5D728A', padding: 13, borderRadius: 6, fontFamily: Platform.OS === 'web' ? 'ui-monospace, SFMono-Regular, Menlo, monospace' : 'monospace', fontSize: 11 },
});
