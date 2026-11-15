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
import {
  analyzeLiveFramesOnEdge,
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
    badge: '72.0% Acc',
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
    title: 'YOLOv8n Detector',
    shortTitle: 'YOLOv8n Box',
    subtitle: 'Littering Region Bounding Boxes',
    badge: '54.3% mAP',
    color: '#8B5CF6',
    icon: '🎯',
  },
};

const NAV_ITEMS = [
  { id: 'live', icon: '📷', label: 'Live Cam' },
  { id: 'analyze', icon: '🎞️', label: 'MP4 Video' },
  { id: 'samples', icon: '📂', label: 'MIVIA Set' },
  { id: 'frames', icon: '🖼️', label: 'YOLO Frames' },
  { id: 'matrix', icon: '📊', label: 'Confusion' },
  { id: 'settings', icon: '⚙️', label: 'API Config' },
];

export default function App() {
  const { width: windowWidth, height: windowHeight } = useWindowDimensions();
  const isDesktop = windowWidth >= 980;

  // Primary mode: 'live' (camera stage) or 'analyze' (MP4 stage)
  const [stageMode, setStageMode] = useState('analyze');
  // Bottom panel view: 'samples' | 'frames' | 'matrix' | 'settings'
  const [bottomPanel, setBottomPanel] = useState('samples');
  const [activeNav, setActiveNav] = useState('analyze');

  const [apiBase, setApiBase] = useState('http://localhost:8000');
  const [selectedChoice, setSelectedChoice] = useState('all');
  const [pickedFile, setPickedFile] = useState(null);
  const [selectedSample, setSelectedSample] = useState(null);
  const [samples, setSamples] = useState([]);
  const [analyzing, setAnalyzing] = useState(false);
  const [statusText, setStatusText] = useState('Select an MP4 file or an official MIVIA test sample.');
  const [analysisResult, setAnalysisResult] = useState(null);
  const [selectedFramePreview, setSelectedFramePreview] = useState(null);
  const [evaluationData, setEvaluationData] = useState(FALLBACK_EVALUATION);
  const [selectedEvalModel, setSelectedEvalModel] = useState('neural_network');
  const [apiConnected, setApiConnected] = useState(false);

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
  const baselineFrameRef = useRef(null);
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

  const handleNavClick = (id) => {
    setActiveNav(id);
    if (id === 'live') {
      setStageMode('live');
      setBottomPanel('frames');
    } else if (id === 'analyze') {
      setStageMode('analyze');
      setBottomPanel('samples');
    } else {
      setBottomPanel(id);
    }
  };

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
      if (photo.base64) return `data:image/jpeg;base64,${photo.base64}`;
      if (photo.uri && photo.uri.startsWith('data:')) return photo.uri;
      return null;
    } catch (_e) {
      return null;
    }
  };

  const runFramesAnalysis = async (framesToAnalyze) => {
    const modelsParam = getSelectedModelsList();
    return await analyzeLiveFramesOnEdge(framesToAnalyze, modelsParam);
  };

  const handleResetBaseline = async () => {
    if (!cameraReady) return;
    const base = await captureSingleFastFrame(true);
    if (base) {
      baselineFrameRef.current = base;
      frameBufferRef.current = [base];
      setBufferedCount(1);
      setLiveResult(null);
      setLiveStatus('Reference background locked.');
    }
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
      setLiveStatus('Running 4-model spatiotemporal analysis...');
      const res = await runFramesAnalysis(burst);
      setLiveResult(res);
      setBottomPanel('frames');
      const anyDumping = res.actions && res.actions.some((a) => a.prediction === 1);
      const boxCount = res.yolo_live?.total_detections || 0;
      setLiveStatus(
        anyDumping
          ? `DUMPING DETECTED (${boxCount} tracked region${boxCount === 1 ? '' : 's'}).`
          : 'NO DUMPING detected in captured sequence.'
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
    frameBufferRef.current = [];
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
        const withSnap = tick % 2 === 0;
        const frame = await captureSingleFastFrame(withSnap);
        if (frame) {
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
              setLiveResult(res);
              setLiveStatus('⚠️ DUMPING DETECTED — Trajectory locked.');
            } else if (Date.now() > detectionHoldUntilRef.current) {
              setLiveResult(res);
              setLiveStatus('Live monitoring active • Scene clear.');
            }
          }
        }
      } catch (_e) {
        // ignore transient frame error
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
        setBottomPanel('frames');
      }
      setApiConnected(true);
      setStatusText(`Analysis complete for ${data.video_name}.`);
    } catch (_err) {
      setStatusText(`API unreachable at ${cleanBaseUrl}. Ensure "python api.py" is running.`);
    } finally {
      setAnalyzing(false);
    }
  };

  // Active result object (either from Live Camera or MP4 Analysis)
  const activeResult = stageMode === 'live' ? liveResult : analysisResult;
  const modelsEval = evaluationData?.models || FALLBACK_EVALUATION.models;

  // Map latest action predictions by model slug
  const actionMap = {};
  if (activeResult?.actions) {
    for (const a of activeResult.actions) {
      actionMap[a.model] = a;
    }
  }

  const yoloDetCount =
    stageMode === 'live'
      ? liveResult?.yolo_live?.total_detections ?? null
      : analysisResult?.yolo?.total_detections ?? null;

  // Determine video source URL for the Top Stage when in MP4 mode on Web
  const stageVideoUrl =
    stageMode === 'analyze'
      ? pickedFile?.uri
        ? pickedFile.uri
        : selectedSample
        ? `${cleanBaseUrl}/api/sample-video/${selectedSample.video_id}`
        : null
      : null;

  // ============================================================================
  // 1. LEFT VERTICAL ICON SIDEBAR (Wireframe Left Rail)
  // ============================================================================
  const renderLeftSidebar = () => (
    <View style={[styles.leftRail, !isDesktop && styles.leftRailMobile]}>
      {/* Top Brand Icon */}
      <Pressable
        style={styles.brandSquare}
        onPress={() => {
          setSelectedChoice('all');
          fetchBackendData();
        }}
      >
        <Text style={styles.brandSquareIcon}>🛡️</Text>
        <View
          style={[
            styles.brandOnlineDot,
            { backgroundColor: apiConnected ? '#10B981' : '#F59E0B' },
          ]}
        />
      </Pressable>

      {/* Center 6 Navigation Squares */}
      <View style={[styles.navStack, !isDesktop && styles.navStackMobile]}>
        {NAV_ITEMS.map((item) => {
          const isActive =
            (item.id === 'live' && stageMode === 'live') ||
            (item.id === 'analyze' && stageMode === 'analyze' && bottomPanel === 'samples') ||
            (item.id !== 'live' && item.id !== 'analyze' && bottomPanel === item.id);
          return (
            <Pressable
              key={item.id}
              onPress={() => handleNavClick(item.id)}
              style={[styles.navSquare, isActive && styles.navSquareActive]}
            >
              <Text style={styles.navSquareIcon}>{item.icon}</Text>
              <Text style={[styles.navSquareLabel, isActive && styles.navSquareLabelActive]}>
                {item.label}
              </Text>
            </Pressable>
          );
        })}
      </View>

      {/* Bottom Quick-Reset / All-Models Toggle Square */}
      <Pressable
        style={[
          styles.navSquare,
          styles.bottomRailSquare,
          selectedChoice === 'all' && styles.navSquareAllActive,
        ]}
        onPress={() => setSelectedChoice('all')}
      >
        <Text style={styles.navSquareIcon}>⚡</Text>
        <Text style={styles.navSquareLabel}>
          {selectedChoice === 'all' ? '4 Models' : 'Reset 4'}
        </Text>
      </Pressable>
    </View>
  );

  // ============================================================================
  // 2. CENTER COLUMN — TOP VIEWPORT STAGE + FLOATING 4-BUTTON DOCK
  // ============================================================================
  const renderTopStage = () => (
    <View style={styles.topStageWrapper}>
      <View style={styles.stageScreen}>
        {/* Top HUD Overlay Bar */}
        <View style={styles.stageHudTop}>
          <View style={styles.hudBadgeRow}>
            <View
              style={[
                styles.livePulseDot,
                {
                  backgroundColor:
                    stageMode === 'live'
                      ? liveMonitoring
                        ? '#EF4444'
                        : '#10B981'
                      : analyzing
                      ? '#38BDF8'
                      : '#10B981',
                },
              ]}
            />
            <Text style={styles.hudTitleText}>
              {stageMode === 'live'
                ? liveMonitoring
                  ? `LIVE AI STREAM • ${liveFps} FPS • BUFFER ${bufferedCount}/24`
                  : 'LIVE CAMERA VIEWPORT'
                : pickedFile
                ? `MP4 STREAM • ${pickedFile.name}`
                : selectedSample
                ? `MIVIA TEST SAMPLE • ${selectedSample.video_id}`
                : 'SURVEILLANCE VIEWPORT'}
            </Text>
          </View>

          <View style={styles.hudRightPills}>
            {activeResult?.ground_truth_verdict && stageMode === 'analyze' && (
              <View style={styles.gtHudPill}>
                <Text style={styles.gtHudPillText}>
                  GT: {activeResult.ground_truth_verdict}
                </Text>
              </View>
            )}
            <Pressable
              style={styles.modeSwitchPill}
              onPress={() => setStageMode((m) => (m === 'live' ? 'analyze' : 'live'))}
            >
              <Text style={styles.modeSwitchPillText}>
                {stageMode === 'live' ? 'Switch to MP4 Mode' : 'Switch to Live Camera'}
              </Text>
            </Pressable>
          </View>
        </View>

        {/* Main Visual Content inside the Stage */}
        {stageMode === 'live' ? (
          !permission?.granted ? (
            <View style={styles.stageEmptyState}>
              <Text style={styles.stageEmptyIcon}>📷</Text>
              <Text style={styles.stageEmptyTitle}>Camera Permission Required</Text>
              <Text style={styles.stageEmptySub}>
                Enable camera access to run real-time 5-Guard Falling-Object & MLP detection.
              </Text>
              <Pressable style={styles.stageGrantBtn} onPress={requestPermission}>
                <Text style={styles.stageGrantBtnText}>Allow Camera Access</Text>
              </Pressable>
            </View>
          ) : (
            <View testID="live-camera-frame" style={styles.cameraStageFill}>
              {Platform.OS === 'web' &&
                React.createElement(
                  'style',
                  null,
                  '[data-testid="live-camera-frame"] video { object-fit: contain !important; width: 100% !important; height: 100% !important; }'
                )}
              <CameraView
                ref={cameraRef}
                style={styles.cameraView}
                facing={cameraFacing}
                onCameraReady={() => setCameraReady(true)}
              />
              {/* Real-Time Bounding Box Overlay */}
              {liveResult?.yolo_live?.detections?.map((det, idx) => {
                const [nx1, ny1, nx2, ny2] = det.norm_box || [0.25, 0.25, 0.75, 0.75];
                const left = Platform.OS === 'web' && cameraMirrored ? 1 - nx2 : nx1;
                return (
                  <View
                    key={idx}
                    pointerEvents="none"
                    style={{
                      position: 'absolute',
                      left: `${left * 100}%`,
                      top: `${ny1 * 100}%`,
                      width: `${Math.max(8, (nx2 - nx1) * 100)}%`,
                      height: `${Math.max(8, (ny2 - ny1) * 100)}%`,
                      borderWidth: 3,
                      borderColor: '#EF4444',
                      backgroundColor: 'rgba(239, 68, 68, 0.20)',
                      borderRadius: 6,
                    }}
                  >
                    <View style={styles.bboxTag}>
                      <Text style={styles.bboxTagText}>
                        LITTERING {(Number(det.confidence || 0.85) * 100).toFixed(0)}%
                      </Text>
                    </View>
                  </View>
                );
              })}
            </View>
          )
        ) : (
          /* MP4 / Sample / YOLO Frame Stage */
          <View style={styles.mp4StageFill}>
            {selectedFramePreview ? (
              <View style={styles.previewSplitStage}>
                <Image
                  source={{ uri: selectedFramePreview.image_url }}
                  style={styles.stagePreviewImage}
                  resizeMode="contain"
                />
                <View style={styles.previewFrameBadge}>
                  <Text style={styles.previewFrameBadgeText}>
                    YOLOv8n Frame #{selectedFramePreview.frame_number} (
                    {selectedFramePreview.timestamp_seconds}s) •{' '}
                    {selectedFramePreview.detections?.length || 0} box(es)
                  </Text>
                  <Pressable onPress={() => setSelectedFramePreview(null)}>
                    <Text style={styles.backToVideoLink}>Back to Video ✕</Text>
                  </Pressable>
                </View>
              </View>
            ) : Platform.OS === 'web' && stageVideoUrl ? (
              <View style={styles.webVideoWrapper}>
                {React.createElement('video', {
                  key: stageVideoUrl,
                  src: stageVideoUrl,
                  controls: true,
                  playsInline: true,
                  style: {
                    width: '100%',
                    height: '100%',
                    maxHeight: 330,
                    objectFit: 'contain',
                    backgroundColor: '#090D16',
                  },
                })}
              </View>
            ) : (
              <View style={styles.stageEmptyState}>
                <Text style={styles.stageEmptyIcon}>🎬</Text>
                <Text style={styles.stageEmptyTitle}>
                  {pickedFile
                    ? pickedFile.name
                    : selectedSample
                    ? `Sample ${selectedSample.video_id}`
                    : 'No Video Loaded'}
                </Text>
                <Text style={styles.stageEmptySub}>{statusText}</Text>
              </View>
            )}
          </View>
        )}

        {/* FLOATING 4-BUTTON ACTION DOCK AT BOTTOM OF STAGE (Matches the 4 Red Bars in Wireframe) */}
        <View style={styles.stageActionDock}>
          {stageMode === 'live' ? (
            <>
              <Pressable
                style={[
                  styles.dockActionBtn,
                  liveMonitoring ? styles.dockBtnDanger : styles.dockBtnCrimson,
                ]}
                onPress={liveMonitoring ? stopLiveMonitoring : startLiveMonitoring}
              >
                <Text style={styles.dockActionBtnText}>
                  {liveMonitoring ? '⏹ Stop Live AI' : '▶ Start Live AI'}
                </Text>
              </Pressable>

              <Pressable
                style={[
                  styles.dockActionBtn,
                  styles.dockBtnCrimson,
                  (!cameraReady || liveAnalyzing) && styles.buttonDisabled,
                ]}
                disabled={!cameraReady || liveAnalyzing}
                onPress={handleCaptureBurstAndAnalyze}
              >
                <Text style={styles.dockActionBtnText}>
                  {liveAnalyzing ? '⏳ Capturing 24f...' : '⚡ Capture 24f Burst'}
                </Text>
              </Pressable>

              <Pressable
                style={[styles.dockActionBtn, styles.dockBtnCrimsonSecondary]}
                onPress={handleResetBaseline}
              >
                <Text style={styles.dockActionBtnText}>🎯 Lock Background</Text>
              </Pressable>

              <Pressable
                style={[styles.dockActionBtn, styles.dockBtnCrimsonSecondary]}
                onPress={() => {
                  baselineFrameRef.current = null;
                  setCameraFacing((prev) => (prev === 'back' ? 'front' : 'back'));
                }}
              >
                <Text style={styles.dockActionBtnText}>🔄 Flip Camera</Text>
              </Pressable>
            </>
          ) : (
            <>
              <Pressable
                style={[
                  styles.dockActionBtn,
                  styles.dockBtnCrimson,
                  ((!pickedFile && !selectedSample) || analyzing) && styles.buttonDisabled,
                ]}
                disabled={(!pickedFile && !selectedSample) || analyzing}
                onPress={handleRunAnalysis}
              >
                <Text style={styles.dockActionBtnText}>
                  {analyzing ? '⏳ Analyzing...' : '▶ Analyze Video'}
                </Text>
              </Pressable>

              <Pressable
                style={[styles.dockActionBtn, styles.dockBtnCrimsonSecondary]}
                onPress={handlePickVideo}
              >
                <Text style={styles.dockActionBtnText}>📁 Upload MP4</Text>
              </Pressable>

              <Pressable
                style={[styles.dockActionBtn, styles.dockBtnCrimsonSecondary]}
                onPress={() => {
                  setBottomPanel('samples');
                  if (samples.length > 0) {
                    const idx = samples.findIndex((s) => s.video_id === selectedSample?.video_id);
                    const next = samples[(idx + 1) % samples.length];
                    handleSelectSample(next);
                  }
                }}
              >
                <Text style={styles.dockActionBtnText}>🎞️ Next MIVIA Sample</Text>
              </Pressable>

              <Pressable
                style={[styles.dockActionBtn, styles.dockBtnCrimsonSecondary]}
                onPress={() => setStageMode('live')}
              >
                <Text style={styles.dockActionBtnText}>📷 Live Camera</Text>
              </Pressable>
            </>
          )}
        </View>
      </View>
    </View>
  );

  // ============================================================================
  // 3. CENTER COLUMN — BOTTOM WORKSPACE PANEL (Wireframe Lower-Center Box)
  // ============================================================================
  const renderBottomWorkspace = () => {
    const selectedEvalObj = modelsEval[selectedEvalModel] || modelsEval.neural_network;
    const cm = selectedEvalObj?.metrics?.confusion_matrix || [
      [37, 13],
      [13, 37],
    ];

    return (
      <View style={styles.bottomWorkspaceCard}>
        {/* Sub-navigation bar inside Bottom Workspace */}
        <View style={styles.workspaceHeader}>
          <View style={styles.workspaceTabsRow}>
            {[
              { id: 'samples', title: `📂 MIVIA Test Set (${samples.length || 12})` },
              {
                id: 'frames',
                title: `🖼️ YOLOv8n Detections (${
                  stageMode === 'live'
                    ? liveResult?.yolo_live?.total_detections || 0
                    : analysisResult?.yolo?.frame_results?.length || 0
                })`,
              },
              { id: 'matrix', title: `📊 Confusion Matrix (${MODEL_META[selectedEvalModel]?.shortTitle})` },
              { id: 'settings', title: '⚙️ API & Stream Config' },
            ].map((tab) => {
              const active = bottomPanel === tab.id;
              return (
                <Pressable
                  key={tab.id}
                  onPress={() => setBottomPanel(tab.id)}
                  style={[styles.wsTabPill, active && styles.wsTabPillActive]}
                >
                  <Text style={[styles.wsTabText, active && styles.wsTabTextActive]}>
                    {tab.title}
                  </Text>
                </Pressable>
              );
            })}
          </View>

          <Text style={styles.wsStatusMini} numberOfLines={1}>
            {stageMode === 'live' ? liveStatus : statusText}
          </Text>
        </View>

        {/* TAB 1: MIVIA Official Test Samples Grid */}
        {bottomPanel === 'samples' && (
          <View style={styles.wsBody}>
            <Text style={styles.wsHelper}>
              Click any official MIVIA test video below to load it into the viewport above, then press "Analyze Video":
            </Text>
            <ScrollView
              horizontal
              showsHorizontalScrollIndicator={false}
              contentContainerStyle={styles.samplesCardStrip}
            >
              {samples.length > 0 ? (
                samples.map((s) => {
                  const isSel = selectedSample?.video_id === s.video_id;
                  const isDump = s.label === 1;
                  return (
                    <Pressable
                      key={s.video_id}
                      onPress={() => handleSelectSample(s)}
                      style={[styles.sampleTile, isSel && styles.sampleTileSelected]}
                    >
                      <View style={styles.sampleTileTop}>
                        <Text style={styles.sampleTileIcon}>{isDump ? '🚨' : '✅'}</Text>
                        <View
                          style={[
                            styles.sampleTileTag,
                            {
                              backgroundColor: isDump
                                ? 'rgba(239, 68, 68, 0.18)'
                                : 'rgba(16, 185, 129, 0.18)',
                            },
                          ]}
                        >
                          <Text
                            style={[
                              styles.sampleTileTagText,
                              { color: isDump ? '#FCA5A5' : '#6EE7B7' },
                            ]}
                          >
                            {isDump ? 'DUMPING' : 'NO DUMPING'}
                          </Text>
                        </View>
                      </View>
                      <Text style={styles.sampleTileTitle}>{s.video_id}</Text>
                      <Text style={styles.sampleTileSub}>Official Test Split • 24f</Text>
                    </Pressable>
                  );
                })
              ) : (
                <Text style={styles.wsHelper}>
                  Connect to Python API ({cleanBaseUrl}) to browse the 12 official MIVIA test videos.
                </Text>
              )}
            </ScrollView>
          </View>
        )}

        {/* TAB 2: YOLOv8n Annotated Frames / Live Snapshot */}
        {bottomPanel === 'frames' && (
          <View style={styles.wsBody}>
            {stageMode === 'live' && liveResult?.yolo_live?.annotated_data_uri ? (
              <View style={styles.liveSnapRow}>
                <Image
                  source={{ uri: liveResult.yolo_live.annotated_data_uri }}
                  style={styles.liveSnapThumb}
                  resizeMode="contain"
                />
                <View style={{ flex: 1, justifyContent: 'center' }}>
                  <Text style={styles.wsSectionTitle}>
                    Live Projectile Trajectory Snapshot
                  </Text>
                  <Text style={styles.wsHelper}>
                    5-Guard Falling Object Tracker verified {liveResult.yolo_live.total_detections}{' '}
                    dropped object(s) while rejecting human body/head motion.
                  </Text>
                </View>
              </View>
            ) : analysisResult?.yolo?.frame_results?.length > 0 ? (
              <>
                <Text style={styles.wsHelper}>
                  YOLOv8n scanned {analysisResult.yolo.frames_sampled} evenly-spaced frames (
                  {analysisResult.yolo.total_detections} total boxes). Click any frame to inspect in main viewport:
                </Text>
                <ScrollView
                  horizontal
                  showsHorizontalScrollIndicator={false}
                  contentContainerStyle={styles.samplesCardStrip}
                >
                  {analysisResult.yolo.frame_results.map((frame) => {
                    const hasBox = frame.detections?.length > 0;
                    const isFocused =
                      selectedFramePreview?.frame_number === frame.frame_number;
                    return (
                      <Pressable
                        key={frame.frame_number}
                        onPress={() => setSelectedFramePreview(frame)}
                        style={[
                          styles.yoloFrameTile,
                          hasBox && styles.yoloFrameTileDetected,
                          isFocused && styles.yoloFrameTileFocused,
                        ]}
                      >
                        <Image
                          source={{ uri: frame.image_url }}
                          style={styles.yoloFrameImg}
                          resizeMode="cover"
                        />
                        <View style={styles.yoloFrameFooter}>
                          <Text style={styles.yoloFrameTitle}>
                            #{frame.frame_number} ({frame.timestamp_seconds}s)
                          </Text>
                          <Text
                            style={[
                              styles.yoloFrameCount,
                              { color: hasBox ? '#FCA5A5' : '#94A3B8' },
                            ]}
                          >
                            {hasBox ? `${frame.detections.length} box` : 'Clear'}
                          </Text>
                        </View>
                      </Pressable>
                    );
                  })}
                </ScrollView>
              </>
            ) : (
              <View style={styles.wsEmptyBox}>
                <Text style={styles.wsHelper}>
                  Run "Analyze Video" on an MP4/sample or "Capture 24f Burst" on Live Camera to inspect annotated YOLOv8n frames here.
                </Text>
              </View>
            )}
          </View>
        )}

        {/* TAB 3: Interactive 2x2 Confusion Matrix for Selected Model */}
        {bottomPanel === 'matrix' && (
          <View style={styles.wsBody}>
            <View style={styles.matrixWorkspaceRow}>
              {[
                {
                  code: 'TN',
                  title: 'True Negative',
                  val: cm[0][0],
                  sub: 'Actual: Clean → Pred: Clean',
                  color: '#10B981',
                },
                {
                  code: 'FP',
                  title: 'False Positive',
                  val: cm[0][1],
                  sub: 'Actual: Clean → Pred: Dumping',
                  color: '#F59E0B',
                },
                {
                  code: 'FN',
                  title: 'False Negative',
                  val: cm[1][0],
                  sub: 'Actual: Dumping → Pred: Clean',
                  color: '#EF4444',
                },
                {
                  code: 'TP',
                  title: 'True Positive',
                  val: cm[1][1],
                  sub: 'Actual: Dumping → Pred: Dumping',
                  color: '#3B82F6',
                },
              ].map((cell) => (
                <View
                  key={cell.code}
                  style={[styles.wsMatrixCell, { borderColor: `${cell.color}66` }]}
                >
                  <View style={styles.wsMatrixTop}>
                    <Text style={[styles.wsMatrixCode, { color: cell.color }]}>{cell.code}</Text>
                    <Text style={styles.wsMatrixNumber}>{cell.val}</Text>
                  </View>
                  <Text style={styles.wsMatrixTitle}>{cell.title}</Text>
                  <Text style={styles.wsMatrixSub}>{cell.sub}</Text>
                </View>
              ))}
            </View>
          </View>
        )}

        {/* TAB 4: API Endpoint & Filter Settings */}
        {bottomPanel === 'settings' && (
          <View style={styles.wsBody}>
            <Text style={styles.wsHelper}>
              Python Backend Server URL (serves MIVIA samples, MoViNet A0, Optical Flow SVM, MLP & YOLOv8n):
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
              <Pressable style={styles.refreshBtn} onPress={fetchBackendData}>
                <Text style={styles.refreshBtnText}>Reconnect API</Text>
              </Pressable>
            </View>
          </View>
        )}
      </View>
    );
  };

  // ============================================================================
  // 4. RIGHT ANALYTICS SIDEBAR (Top 2x2 Grid + 7 Stacked Telemetry/Metric Bars)
  // ============================================================================
  const renderRightSidebar = () => {
    const modelSlugs = ['neural_network', 'movinet_a0', 'optical_flow_svm', 'yolov8n'];
    const selectedEvalObj = modelsEval[selectedEvalModel] || modelsEval.neural_network;
    const cm = selectedEvalObj?.metrics?.confusion_matrix || [
      [37, 13],
      [13, 37],
    ];

    const anyDumping =
      activeResult?.actions && activeResult.actions.some((a) => a.prediction === 1);

    return (
      <View style={[styles.rightSidebar, !isDesktop && styles.rightSidebarMobile]}>
        {/* TOP 2x2 GRID OF 4 ALGORITHM CARDS (Matches the 4 Top Rectangles in Wireframe) */}
        <View style={styles.grid2x2}>
          {modelSlugs.map((slug) => {
            const meta = MODEL_META[slug];
            const act = actionMap[slug];
            const isSelectedFilter = selectedChoice === slug;
            const isYolo = slug === 'yolov8n';

            let liveBadgeText = meta.badge;
            let liveVerdictColor = '#94A3B8';
            let cardBorderColor = isSelectedFilter ? meta.color : '#2A3444';

            if (isYolo && yoloDetCount !== null) {
              liveBadgeText = yoloDetCount > 0 ? `${yoloDetCount} BOXES` : '0 BOXES';
              liveVerdictColor = yoloDetCount > 0 ? '#F87171' : '#34D399';
              if (yoloDetCount > 0) cardBorderColor = '#EF4444';
            } else if (act) {
              const isDump = act.prediction === 1;
              liveBadgeText = isDump ? 'DUMPING' : 'CLEAR';
              liveVerdictColor = isDump ? '#F87171' : '#34D399';
              if (isDump) cardBorderColor = '#EF4444';
            }

            return (
              <Pressable
                key={slug}
                onPress={() => {
                  setSelectedChoice((prev) => (prev === slug ? 'all' : slug));
                  if (slug !== 'yolov8n') setSelectedEvalModel(slug);
                }}
                style={[
                  styles.gridCard,
                  { borderColor: cardBorderColor },
                  isSelectedFilter && { backgroundColor: '#1E293B' },
                ]}
              >
                <View style={styles.gridCardTopRow}>
                  <Text style={styles.gridCardIcon}>{meta.icon}</Text>
                  <View
                    style={[
                      styles.gridCardPill,
                      { backgroundColor: `${liveVerdictColor}20` },
                    ]}
                  >
                    <Text style={[styles.gridCardPillText, { color: liveVerdictColor }]}>
                      {liveBadgeText}
                    </Text>
                  </View>
                </View>

                <Text style={styles.gridCardTitle} numberOfLines={1}>
                  {meta.shortTitle}
                </Text>
                <Text style={styles.gridCardSub} numberOfLines={1}>
                  {act
                    ? `Score: ${Number(act.score).toFixed(3)} • ${(Number(act.seconds) * 1000).toFixed(0)}ms`
                    : meta.subtitle}
                </Text>
              </Pressable>
            );
          })}
        </View>

        {/* STACK OF 7 HORIZONTAL TELEMETRY & BENCHMARK BARS (Matches the 7 Bars in Wireframe) */}
        <View style={styles.barsStack}>
          {/* Bar 1: Overall Verdict & Active Filter Status */}
          <Pressable
            style={[
              styles.stackBar,
              activeResult && {
                borderColor: anyDumping ? '#EF4444' : '#10B981',
                backgroundColor: anyDumping
                  ? 'rgba(239, 68, 68, 0.12)'
                  : 'rgba(16, 185, 129, 0.10)',
              },
            ]}
            onPress={() => setSelectedChoice('all')}
          >
            <View style={{ flex: 1 }}>
              <Text style={styles.stackBarLabel}>1. ENSEMBLE VERDICT & ACTIVE FILTER</Text>
              <Text style={styles.stackBarValue} numberOfLines={1}>
                {activeResult
                  ? anyDumping
                    ? '🚨 POSSIBLE ILLEGAL DUMPING DETECTED'
                    : '✅ NO DUMPING DETECTED (SCENE CLEAR)'
                  : `Ready • Active: ${
                      selectedChoice === 'all' ? 'All 4 Algorithms' : MODEL_META[selectedChoice]?.title
                    }`}
              </Text>
            </View>
            {(analyzing || liveAnalyzing) && <ActivityIndicator size="small" color="#38BDF8" />}
          </Pressable>

          {/* Bar 2: Neural Network (MLP) Benchmark + Live Score */}
          <Pressable
            style={[
              styles.stackBar,
              selectedEvalModel === 'neural_network' && styles.stackBarSelected,
            ]}
            onPress={() => {
              setSelectedEvalModel('neural_network');
              setBottomPanel('matrix');
            }}
          >
            <View style={[styles.barAccent, { backgroundColor: '#10B981' }]} />
            <View style={{ flex: 1 }}>
              <View style={styles.stackBarRowBetween}>
                <Text style={styles.stackBarTitle}>2. Neural Network (MLP)</Text>
                <Text style={[styles.stackBarMetric, { color: '#10B981' }]}>
                  Acc 74.0% • F1 74.0%
                </Text>
              </View>
              <Text style={styles.stackBarSub}>
                {actionMap.neural_network
                  ? `Live Verdict: ${actionMap.neural_network.verdict} (Prob: ${Number(
                      actionMap.neural_network.score
                    ).toFixed(4)})`
                  : '528 features • 21.5s train • 0.15 MiB checkpoint'}
              </Text>
            </View>
          </Pressable>

          {/* Bar 3: MoViNet A0 Benchmark + Live Score */}
          <Pressable
            style={[
              styles.stackBar,
              selectedEvalModel === 'movinet_a0' && styles.stackBarSelected,
            ]}
            onPress={() => {
              setSelectedEvalModel('movinet_a0');
              setBottomPanel('matrix');
            }}
          >
            <View style={[styles.barAccent, { backgroundColor: '#3B82F6' }]} />
            <View style={{ flex: 1 }}>
              <View style={styles.stackBarRowBetween}>
                <Text style={styles.stackBarTitle}>3. MoViNet A0 (3D CNN)</Text>
                <Text style={[styles.stackBarMetric, { color: '#3B82F6' }]}>
                  Acc 72.0% • F1 72.5%
                </Text>
              </View>
              <Text style={styles.stackBarSub}>
                {actionMap.movinet_a0
                  ? `Live Verdict: ${actionMap.movinet_a0.verdict} (Sigmoid: ${Number(
                      actionMap.movinet_a0.score
                    ).toFixed(4)})`
                  : '24×112×112 RGB • Recall 74.0% • 7.38 MiB checkpoint'}
              </Text>
            </View>
          </Pressable>

          {/* Bar 4: Optical Flow + SVM Benchmark + Live Score */}
          <Pressable
            style={[
              styles.stackBar,
              selectedEvalModel === 'optical_flow_svm' && styles.stackBarSelected,
            ]}
            onPress={() => {
              setSelectedEvalModel('optical_flow_svm');
              setBottomPanel('matrix');
            }}
          >
            <View style={[styles.barAccent, { backgroundColor: '#F59E0B' }]} />
            <View style={{ flex: 1 }}>
              <View style={styles.stackBarRowBetween}>
                <Text style={styles.stackBarTitle}>4. Optical Flow + SVM</Text>
                <Text style={[styles.stackBarMetric, { color: '#F59E0B' }]}>
                  Acc 64.0% • F1 65.4%
                </Text>
              </View>
              <Text style={styles.stackBarSub}>
                {actionMap.optical_flow_svm
                  ? `Live Verdict: ${actionMap.optical_flow_svm.verdict} (Margin: ${Number(
                      actionMap.optical_flow_svm.score
                    ).toFixed(4)})`
                  : 'Farnebäck 4×4×4 grid • Recall 68.0% • 0.38 MiB'}
              </Text>
            </View>
          </Pressable>

          {/* Bar 5: YOLOv8n Object Detector Benchmark + Live Detections */}
          <Pressable
            style={styles.stackBar}
            onPress={() => setBottomPanel('frames')}
          >
            <View style={[styles.barAccent, { backgroundColor: '#8B5CF6' }]} />
            <View style={{ flex: 1 }}>
              <View style={styles.stackBarRowBetween}>
                <Text style={styles.stackBarTitle}>5. YOLOv8n Region Detector</Text>
                <Text style={[styles.stackBarMetric, { color: '#A78BFA' }]}>
                  mAP@50 54.3% • Rec 72.2%
                </Text>
              </View>
              <Text style={styles.stackBarSub}>
                {yoloDetCount !== null
                  ? `Detected ${yoloDetCount} littering region(s) in current video`
                  : 'Roboflow Littering val (1,168 imgs) • 5.93 MiB'}
              </Text>
            </View>
          </Pressable>

          {/* Bar 6: Confusion Matrix Summary Bar */}
          <Pressable
            style={styles.stackBar}
            onPress={() => setBottomPanel('matrix')}
          >
            <View style={{ flex: 1 }}>
              <View style={styles.stackBarRowBetween}>
                <Text style={styles.stackBarLabel}>
                  6. CONFUSION MATRIX ({MODEL_META[selectedEvalModel]?.shortTitle})
                </Text>
                <Text style={styles.stackBarMiniLink}>Inspect ↗</Text>
              </View>
              <View style={styles.cmInlineRow}>
                <Text style={[styles.cmInlineItem, { color: '#34D399' }]}>TN: {cm[0][0]}</Text>
                <Text style={[styles.cmInlineItem, { color: '#FBBF24' }]}>FP: {cm[0][1]}</Text>
                <Text style={[styles.cmInlineItem, { color: '#F87171' }]}>FN: {cm[1][0]}</Text>
                <Text style={[styles.cmInlineItem, { color: '#60A5FA' }]}>TP: {cm[1][1]}</Text>
              </View>
            </View>
          </Pressable>

          {/* Bar 7: System & Dataset Telemetry Bar */}
          <Pressable
            style={styles.stackBar}
            onPress={() => setBottomPanel('settings')}
          >
            <View style={{ flex: 1 }}>
              <View style={styles.stackBarRowBetween}>
                <Text style={styles.stackBarLabel}>7. SYSTEM & DATASET TELEMETRY</Text>
                <Text
                  style={[
                    styles.stackBarMetric,
                    { color: apiConnected ? '#10B981' : '#38BDF8' },
                  ]}
                >
                  {apiConnected ? '● API ONLINE' : '● EDGE MODE'}
                </Text>
              </View>
              <Text style={styles.stackBarSub} numberOfLines={1}>
                MIVIA Split: 320 Train / 80 Val / 100 Test • {cleanBaseUrl}
              </Text>
            </View>
          </Pressable>
        </View>
      </View>
    );
  };

  return (
    <SafeAreaView style={styles.safeArea}>
      <StatusBar style="light" />
      <ScrollView
        contentContainerStyle={[
          styles.outerScroll,
          isDesktop && { minHeight: windowHeight },
        ]}
      >
        <View style={[styles.dashboardShell, !isDesktop && styles.dashboardShellMobile]}>
          {/* Column 1: Left Icon Rail */}
          {renderLeftSidebar()}

          {/* Column 2: Center Split (Top Stage + Bottom Workspace) */}
          <View style={styles.centerColumn}>
            {renderTopStage()}
            {renderBottomWorkspace()}
          </View>

          {/* Column 3: Right Analytics Panel (2x2 Grid + 7 Telemetry Rows) */}
          {renderRightSidebar()}
        </View>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safeArea: {
    flex: 1,
    backgroundColor: '#12151C',
  },
  outerScroll: {
    flexGrow: 1,
    padding: 12,
    justifyContent: 'center',
  },
  dashboardShell: {
    flex: 1,
    flexDirection: 'row',
    backgroundColor: '#181C24',
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#262D3B',
    overflow: 'hidden',
    minHeight: 740,
  },
  dashboardShellMobile: {
    flexDirection: 'column',
  },

  // ---------------------------------------------------------------------------
  // 1. LEFT VERTICAL ICON SIDEBAR
  // ---------------------------------------------------------------------------
  leftRail: {
    width: 74,
    backgroundColor: '#1E232E',
    borderRightWidth: 1,
    borderRightColor: '#2A3140',
    paddingVertical: 14,
    paddingHorizontal: 8,
    alignItems: 'center',
    justifyContent: 'space-between',
  },
  leftRailMobile: {
    width: '100%',
    flexDirection: 'row',
    borderRightWidth: 0,
    borderBottomWidth: 1,
    borderBottomColor: '#2A3140',
    paddingVertical: 10,
    paddingHorizontal: 12,
  },
  brandSquare: {
    width: 52,
    height: 52,
    borderRadius: 12,
    backgroundColor: '#84CC16',
    alignItems: 'center',
    justifyContent: 'center',
    position: 'relative',
  },
  brandSquareIcon: {
    fontSize: 22,
  },
  brandOnlineDot: {
    position: 'absolute',
    bottom: 4,
    right: 4,
    width: 9,
    height: 9,
    borderRadius: 5,
    borderWidth: 1.5,
    borderColor: '#1E232E',
  },
  navStack: {
    gap: 10,
    alignItems: 'center',
    width: '100%',
  },
  navStackMobile: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    flex: 1,
    marginHorizontal: 8,
  },
  navSquare: {
    width: 56,
    height: 54,
    borderRadius: 11,
    backgroundColor: '#262C3A',
    borderWidth: 1,
    borderColor: '#323A4B',
    alignItems: 'center',
    justifyContent: 'center',
    paddingVertical: 4,
  },
  navSquareActive: {
    backgroundColor: '#84CC16',
    borderColor: '#A3E635',
  },
  navSquareAllActive: {
    borderColor: '#84CC16',
    backgroundColor: 'rgba(132, 204, 22, 0.18)',
  },
  navSquareIcon: {
    fontSize: 17,
  },
  navSquareLabel: {
    color: '#94A3B8',
    fontSize: 9.5,
    fontWeight: '700',
    marginTop: 3,
  },
  navSquareLabelActive: {
    color: '#0F172A',
    fontWeight: '800',
  },
  bottomRailSquare: {
    marginTop: 6,
  },

  // ---------------------------------------------------------------------------
  // 2. CENTER COLUMN (TOP VIEWPORT + BOTTOM PANEL)
  // ---------------------------------------------------------------------------
  centerColumn: {
    flex: 1.45,
    backgroundColor: '#161A22',
    padding: 14,
    justifyContent: 'space-between',
    gap: 14,
  },
  topStageWrapper: {
    flex: 1.15,
    minHeight: 370,
    backgroundColor: '#222733',
    borderRadius: 14,
    padding: 10,
    borderWidth: 1,
    borderColor: '#2E3646',
  },
  stageScreen: {
    flex: 1,
    backgroundColor: '#0D1017',
    borderRadius: 10,
    overflow: 'hidden',
    position: 'relative',
    justifyContent: 'space-between',
  },
  stageHudTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    flexWrap: 'wrap',
    paddingHorizontal: 14,
    paddingVertical: 10,
    backgroundColor: 'rgba(13, 16, 23, 0.86)',
    borderBottomWidth: 1,
    borderBottomColor: '#1E2532',
    zIndex: 10,
    gap: 8,
  },
  hudBadgeRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  livePulseDot: {
    width: 9,
    height: 9,
    borderRadius: 5,
  },
  hudTitleText: {
    color: '#E2E8F0',
    fontSize: 12,
    fontWeight: '800',
    letterSpacing: 0.4,
  },
  hudRightPills: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  gtHudPill: {
    backgroundColor: 'rgba(56, 189, 248, 0.16)',
    borderWidth: 1,
    borderColor: '#38BDF8',
    paddingHorizontal: 9,
    paddingVertical: 4,
    borderRadius: 6,
  },
  gtHudPillText: {
    color: '#38BDF8',
    fontSize: 11,
    fontWeight: '800',
  },
  modeSwitchPill: {
    backgroundColor: '#1E293B',
    borderWidth: 1,
    borderColor: '#334155',
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 6,
  },
  modeSwitchPillText: {
    color: '#CBD5E1',
    fontSize: 11,
    fontWeight: '700',
  },
  cameraStageFill: {
    flex: 1,
    minHeight: 260,
    position: 'relative',
  },
  cameraView: {
    width: '100%',
    height: '100%',
  },
  bboxTag: {
    backgroundColor: '#EF4444',
    alignSelf: 'flex-start',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderBottomRightRadius: 6,
  },
  bboxTagText: {
    color: '#FFFFFF',
    fontSize: 10,
    fontWeight: '800',
  },
  mp4StageFill: {
    flex: 1,
    minHeight: 260,
    justifyContent: 'center',
    alignItems: 'center',
  },
  webVideoWrapper: {
    width: '100%',
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  previewSplitStage: {
    width: '100%',
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    padding: 8,
  },
  stagePreviewImage: {
    width: '100%',
    height: 250,
    borderRadius: 8,
  },
  previewFrameBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 14,
    backgroundColor: 'rgba(15, 23, 42, 0.9)',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    marginTop: 6,
  },
  previewFrameBadgeText: {
    color: '#F8FAFC',
    fontSize: 12,
    fontWeight: '700',
  },
  backToVideoLink: {
    color: '#38BDF8',
    fontSize: 12,
    fontWeight: '800',
  },
  stageEmptyState: {
    padding: 24,
    alignItems: 'center',
    justifyContent: 'center',
  },
  stageEmptyIcon: {
    fontSize: 36,
    marginBottom: 8,
  },
  stageEmptyTitle: {
    color: '#F8FAFC',
    fontSize: 16,
    fontWeight: '800',
  },
  stageEmptySub: {
    color: '#94A3B8',
    fontSize: 12.5,
    marginTop: 4,
    textAlign: 'center',
    maxWidth: 420,
  },
  stageGrantBtn: {
    marginTop: 12,
    backgroundColor: '#DC2626',
    paddingHorizontal: 16,
    paddingVertical: 9,
    borderRadius: 8,
  },
  stageGrantBtnText: {
    color: '#FFFFFF',
    fontWeight: '800',
    fontSize: 12.5,
  },

  // 4-BUTTON FLOATING DOCK AT BOTTOM OF VIEWPORT (Matches Wireframe Red Bar)
  stageActionDock: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    backgroundColor: '#2A303D',
    padding: 7,
    marginHorizontal: 16,
    marginBottom: 10,
    borderRadius: 10,
    gap: 8,
    borderWidth: 1,
    borderColor: '#3A4254',
  },
  dockActionBtn: {
    flex: 1,
    minWidth: 115,
    paddingVertical: 9,
    paddingHorizontal: 10,
    borderRadius: 6,
    alignItems: 'center',
    justifyContent: 'center',
  },
  dockBtnCrimson: {
    backgroundColor: '#C83232',
  },
  dockBtnCrimsonSecondary: {
    backgroundColor: '#992626',
  },
  dockBtnDanger: {
    backgroundColor: '#EF4444',
  },
  dockActionBtnText: {
    color: '#FFFFFF',
    fontSize: 12,
    fontWeight: '800',
  },
  buttonDisabled: {
    opacity: 0.45,
  },

  // ---------------------------------------------------------------------------
  // 3. BOTTOM WORKSPACE PANEL
  // ---------------------------------------------------------------------------
  bottomWorkspaceCard: {
    flex: 0.85,
    minHeight: 240,
    backgroundColor: '#232834',
    borderRadius: 14,
    padding: 14,
    borderWidth: 1,
    borderColor: '#31394A',
    justifyContent: 'space-between',
  },
  workspaceHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    flexWrap: 'wrap',
    gap: 8,
    paddingBottom: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#2E3646',
  },
  workspaceTabsRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 6,
  },
  wsTabPill: {
    backgroundColor: '#181C24',
    paddingHorizontal: 11,
    paddingVertical: 6,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#2E3646',
  },
  wsTabPillActive: {
    backgroundColor: '#323B4E',
    borderColor: '#84CC16',
  },
  wsTabText: {
    color: '#94A3B8',
    fontSize: 11.5,
    fontWeight: '700',
  },
  wsTabTextActive: {
    color: '#F8FAFC',
  },
  wsStatusMini: {
    color: '#94A3B8',
    fontSize: 11.5,
    maxWidth: 260,
  },
  wsBody: {
    flex: 1,
    justifyContent: 'center',
    paddingTop: 10,
  },
  wsHelper: {
    color: '#94A3B8',
    fontSize: 12,
    marginBottom: 10,
  },
  wsSectionTitle: {
    color: '#F8FAFC',
    fontSize: 14,
    fontWeight: '800',
    marginBottom: 4,
  },
  samplesCardStrip: {
    gap: 10,
    alignItems: 'center',
    paddingVertical: 4,
  },
  sampleTile: {
    width: 145,
    backgroundColor: '#161A22',
    borderRadius: 10,
    padding: 11,
    borderWidth: 1,
    borderColor: '#2E3646',
  },
  sampleTileSelected: {
    borderColor: '#84CC16',
    backgroundColor: 'rgba(132, 204, 22, 0.10)',
  },
  sampleTileTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 8,
  },
  sampleTileIcon: {
    fontSize: 16,
  },
  sampleTileTag: {
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: 5,
  },
  sampleTileTagText: {
    fontSize: 9.5,
    fontWeight: '800',
  },
  sampleTileTitle: {
    color: '#F8FAFC',
    fontSize: 13.5,
    fontWeight: '800',
  },
  sampleTileSub: {
    color: '#64748B',
    fontSize: 11,
    marginTop: 2,
  },
  yoloFrameTile: {
    width: 156,
    backgroundColor: '#161A22',
    borderRadius: 10,
    padding: 6,
    borderWidth: 1,
    borderColor: '#2E3646',
  },
  yoloFrameTileDetected: {
    borderColor: '#EF4444',
  },
  yoloFrameTileFocused: {
    borderColor: '#84CC16',
    borderWidth: 2,
  },
  yoloFrameImg: {
    width: '100%',
    height: 92,
    borderRadius: 6,
    backgroundColor: '#0D1017',
  },
  yoloFrameFooter: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginTop: 6,
    paddingHorizontal: 2,
  },
  yoloFrameTitle: {
    color: '#CBD5E1',
    fontSize: 11,
    fontWeight: '700',
  },
  yoloFrameCount: {
    fontSize: 10.5,
    fontWeight: '800',
  },
  liveSnapRow: {
    flexDirection: 'row',
    gap: 14,
    alignItems: 'center',
  },
  liveSnapThumb: {
    width: 190,
    height: 125,
    borderRadius: 10,
    backgroundColor: '#0D1017',
    borderWidth: 1,
    borderColor: '#EF4444',
  },
  wsEmptyBox: {
    paddingVertical: 24,
    alignItems: 'center',
  },
  matrixWorkspaceRow: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  wsMatrixCell: {
    flex: 1,
    minWidth: 130,
    backgroundColor: '#161A22',
    borderRadius: 10,
    padding: 12,
    borderWidth: 1.5,
  },
  wsMatrixTop: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  wsMatrixCode: {
    fontSize: 12,
    fontWeight: '900',
  },
  wsMatrixNumber: {
    color: '#F8FAFC',
    fontSize: 22,
    fontWeight: '900',
  },
  wsMatrixTitle: {
    color: '#E2E8F0',
    fontSize: 12,
    fontWeight: '700',
    marginTop: 4,
  },
  wsMatrixSub: {
    color: '#64748B',
    fontSize: 10.5,
    marginTop: 2,
  },
  serverRow: {
    flexDirection: 'row',
    gap: 10,
    alignItems: 'center',
  },
  serverInput: {
    flex: 1,
    backgroundColor: '#161A22',
    color: '#F8FAFC',
    borderRadius: 8,
    paddingHorizontal: 12,
    paddingVertical: 9,
    borderWidth: 1,
    borderColor: '#31394A',
    fontSize: 13,
  },
  refreshBtn: {
    backgroundColor: '#84CC16',
    paddingHorizontal: 14,
    paddingVertical: 10,
    borderRadius: 8,
  },
  refreshBtnText: {
    color: '#0F172A',
    fontWeight: '800',
    fontSize: 12.5,
  },

  // ---------------------------------------------------------------------------
  // 4. RIGHT ANALYTICS SIDEBAR (2x2 GRID + 7 HORIZONTAL BARS)
  // ---------------------------------------------------------------------------
  rightSidebar: {
    flex: 1,
    minWidth: 330,
    maxWidth: 430,
    backgroundColor: '#202531',
    borderLeftWidth: 1,
    borderLeftColor: '#2C3344',
    padding: 14,
    justifyContent: 'space-between',
    gap: 14,
  },
  rightSidebarMobile: {
    maxWidth: '100%',
    borderLeftWidth: 0,
    borderTopWidth: 1,
    borderTopColor: '#2C3344',
  },
  grid2x2: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  gridCard: {
    width: '48.2%',
    flexGrow: 1,
    backgroundColor: '#2A3140',
    borderRadius: 10,
    padding: 12,
    borderWidth: 1.5,
    borderColor: '#363F52',
    minHeight: 88,
    justifyContent: 'space-between',
  },
  gridCardTopRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  gridCardIcon: {
    fontSize: 18,
  },
  gridCardPill: {
    paddingHorizontal: 7,
    paddingVertical: 2,
    borderRadius: 5,
  },
  gridCardPillText: {
    fontSize: 10,
    fontWeight: '900',
  },
  gridCardTitle: {
    color: '#F8FAFC',
    fontSize: 13,
    fontWeight: '800',
    marginTop: 6,
  },
  gridCardSub: {
    color: '#94A3B8',
    fontSize: 11,
    marginTop: 2,
  },
  barsStack: {
    flex: 1,
    justifyContent: 'space-between',
    gap: 8,
  },
  stackBar: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#2A3140',
    borderRadius: 9,
    paddingVertical: 10,
    paddingHorizontal: 12,
    borderWidth: 1,
    borderColor: '#363F52',
    gap: 10,
  },
  stackBarSelected: {
    borderColor: '#84CC16',
    backgroundColor: '#2F384A',
  },
  barAccent: {
    width: 4,
    height: '100%',
    minHeight: 28,
    borderRadius: 2,
  },
  stackBarRowBetween: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    gap: 6,
  },
  stackBarLabel: {
    color: '#94A3B8',
    fontSize: 10,
    fontWeight: '800',
    letterSpacing: 0.5,
  },
  stackBarValue: {
    color: '#F8FAFC',
    fontSize: 12.5,
    fontWeight: '800',
    marginTop: 2,
  },
  stackBarTitle: {
    color: '#F8FAFC',
    fontSize: 12.5,
    fontWeight: '800',
  },
  stackBarMetric: {
    fontSize: 11,
    fontWeight: '800',
  },
  stackBarSub: {
    color: '#94A3B8',
    fontSize: 11,
    marginTop: 2,
  },
  stackBarMiniLink: {
    color: '#84CC16',
    fontSize: 10.5,
    fontWeight: '800',
  },
  cmInlineRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: 4,
  },
  cmInlineItem: {
    fontSize: 11.5,
    fontWeight: '800',
  },
});
