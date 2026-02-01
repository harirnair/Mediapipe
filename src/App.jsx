import { useEffect, useRef, useState } from 'react';
import { PoseLandmarker, FaceLandmarker, ImageClassifier, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';
import './App.css';

// Note: tf and tflite are loaded via CDN in index.html for better compatibility with Vite

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [poseLandmarker, setPoseLandmarker] = useState(null);
  const [faceLandmarker, setFaceLandmarker] = useState(null);
  const [loading, setLoading] = useState(true);
  const [cameraActive, setCameraActive] = useState(false);
  const [specsClassifier, setSpecsClassifier] = useState(null);
  const [specsLoading, setSpecsLoading] = useState(false);
  const [visionResolver, setVisionResolver] = useState(null);
  const [resultsDisplay, setResultsDisplay] = useState([]);
  const [debugStatus, setDebugStatus] = useState("Initializing...");
  const cropCanvasRef = useRef(null);
  const lastClassifyTimeRef = useRef({});
  const specsResultsRef = useRef({});
  const [emotionLoading, setEmotionLoading] = useState(false);
  const emotionClassifierRef = useRef(null); // use Ref for detection loop to avoid closure staleness
  const emotionResultsRef = useRef({});

  const drawingUtilsRef = useRef(null);
  const requestRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);
  const lastEmotionClassifyTimeRef = useRef({});

  // Load MediaPipe Models
  useEffect(() => {
    const createModels = async () => {
      try {
        setDebugStatus("Loading Vision Tasks...");
        console.log("Step 1: Initializing FilesetResolver");
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
        );

        setDebugStatus("Loading Pose Model...");
        console.log("Step 2: Loading PoseLandmarker");
        const pose = await PoseLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numPoses: 5,
        });
        setPoseLandmarker(pose);

        setDebugStatus("Loading Face Model...");
        console.log("Step 3: Loading FaceLandmarker");
        const face = await FaceLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`,
            delegate: "GPU"
          },
          outputFaceBlendshapes: true,
          runningMode: "VIDEO",
          numFaces: 5,
        });
        setFaceLandmarker(face);

        setVisionResolver(vision); // Keep for on-demand tools
        console.log("Step 4: Core Vision Ready.");

        // Auto-load Specs Classifier
        const specs = await ImageClassifier.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/image_classifier/efficientnet_lite0/float32/1/efficientnet_lite0.tflite`,
            delegate: "GPU"
          },
          maxResults: 5,
          runningMode: "IMAGE"
        });
        setSpecsClassifier(specs);
        console.log("Step 5: Specs Classifier Active.");

        // Auto-load Emotion CPU Model (High Compatibility)
        setDebugStatus("Loading Emotion Model...");
        window.tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.7/dist/');
        const response = await fetch('/models/emotion_model.tflite');
        const blob = await response.blob();
        const modelUrl = URL.createObjectURL(blob);
        const eModel = await window.tflite.loadTFLiteModel(modelUrl, { numThreads: 1 });
        emotionClassifierRef.current = eModel;
        URL.revokeObjectURL(modelUrl);
        console.log("Step 6: FER+ Emotion Active.");

        setLoading(false);
        setDebugStatus("System Ready.");
      } catch (error) {
        console.error("Initialization Error:", error);
        setDebugStatus("System Error: " + error.message);
      }
    };

    createModels();
  }, []);

  const enableCam = () => {
    if (!poseLandmarker || !faceLandmarker) return;

    setCameraActive(true);
    setDebugStatus("Starting Camera...");

    navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadeddata = () => {
            setDebugStatus("Camera Live. Starting Detection...");
            predictWebcam();
          };
        }
      })
      .catch(err => {
        console.error(err);
        setDebugStatus("Camera Error: " + err.message);
      });
  };

  const startCamera = () => {
    setCameraActive(true);
    enableCam();
  };

  const calculateAngle = (a, b, c) => {
    if (!a || !b || !c) return 0;
    const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
    let angle = Math.abs(radians * 180.0 / Math.PI);
    if (angle > 180.0) angle = 360 - angle;
    return angle;
  };

  const colors = ["#38bdf8", "#f472b6", "#34d399", "#fbbf24", "#a78bfa"];

  /* 
   * EMOTION DETECTION LOGIC
   * Model: MediaPipe Face Landmarker (Blendshapes)
   * We map 52 facial muscle parameters to 7 basic emotions + Confusion.
   */
  const analyzeFace = (blendshapes) => {
    const categories = blendshapes.categories;
    const score = (name) => categories.find(b => b.categoryName === name)?.score || 0;

    // --- MediaPipe Logic (Fallback) ---
    const browDown = (score('browDownLeft') + score('browDownRight')) / 2;
    const eyeWide = (score('eyeWideLeft') + score('eyeWideRight')) / 2;
    const squint = (score('eyeSquintLeft') + score('eyeSquintRight')) / 2;

    return {
      isSquinting: squint > 0.3,
    };
  };

  const analyzePose = (landmarks) => {

    // Knee Angles
    const leftKneeAngle = calculateAngle(landmarks[23], landmarks[25], landmarks[27]);
    const rightKneeAngle = calculateAngle(landmarks[24], landmarks[26], landmarks[28]);
    const isSitting = (leftKneeAngle < 140) || (rightKneeAngle < 140);

    // Visibility Check
    const leftStrict = (landmarks[15].visibility > 0.65);
    const rightStrict = (landmarks[16].visibility > 0.65);

    let armStatus = "None";
    if (leftStrict && rightStrict) armStatus = "Both";
    else if (leftStrict) armStatus = "Left Only";
    else if (rightStrict) armStatus = "Right Only";

    // Simplified Return
    return {
      posture: isSitting ? "Sitting" : "Standing",
      arms: armStatus,
      nose: landmarks[0] // Return nose for face matching
    };
  };

  const predictWebcam = () => {
    if (!videoRef.current || !canvasRef.current || !poseLandmarker || !faceLandmarker) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;

    // Initialize drawing utils once
    if (!drawingUtilsRef.current) {
      const ctx = canvas.getContext("2d");
      drawingUtilsRef.current = new DrawingUtils(ctx);
    }

    const ctx = canvas.getContext("2d");

    // Resize canvas if needed
    if (video.videoWidth > 0 && (canvas.width !== video.videoWidth || canvas.height !== video.videoHeight)) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    let startTimeMs = performance.now();

    if (video.currentTime !== lastVideoTimeRef.current) {
      lastVideoTimeRef.current = video.currentTime;

      try {
        // Run both detections
        const poseResults = poseLandmarker.detectForVideo(video, startTimeMs);
        const faceResults = faceLandmarker.detectForVideo(video, startTimeMs);

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const currentStats = [];

        // Map Faces to Poses
        // We'll simplisticly match Face Center to Pose Nose
        const faces = faceResults.faceBlendshapes.map((shapes, i) => {
          const landmarks = faceResults.faceLandmarks[i];
          const nose = landmarks ? landmarks[1] : { x: 0, y: 0 };
          const faceAnalysis = analyzeFace(shapes);

          // Use FER+ if available, otherwise hide emotion
          const mappedEmotion = emotionResultsRef.current[i] || "---";

          return { id: i, nose, emotion: mappedEmotion, isSquinting: faceAnalysis.isSquinting };
        });

        if (poseResults.landmarks && poseResults.landmarks.length > 0) {
          poseResults.landmarks.forEach((landmarks, index) => {
            const personColor = colors[index % colors.length];

            // Analyze Pose
            const poseData = analyzePose(landmarks);

            // Find matching face
            let assignedEmotion = "Scanning...";
            let isSquinting = false;
            let matchedFaceLandmarks = null;
            let matchedFaceId = null;
            let minDist = 0.2;

            if (poseData.nose && faces.length > 0) {
              faces.forEach(face => {
                const dist = Math.sqrt(
                  Math.pow(face.nose.x - poseData.nose.x, 2) +
                  Math.pow(face.nose.y - poseData.nose.y, 2)
                );
                if (dist < minDist) {
                  minDist = dist;
                  assignedEmotion = face.emotion;
                  isSquinting = face.isSquinting;
                  matchedFaceLandmarks = faceResults.faceLandmarks[face.id];
                  matchedFaceId = face.id;
                }
              });
            }

            // 1. Prepare Face Crop for AI (Specs + Emotion)
            const now = performance.now();
            let isWearingSpecs = false; // Reset by default for each person

            // Check if we have a cached result for THIS specific face
            if (matchedFaceId !== null && specsResultsRef.current[matchedFaceId] !== undefined) {
              isWearingSpecs = specsResultsRef.current[matchedFaceId];
            }

            if (matchedFaceLandmarks) {
              const cropCtx = cropCanvasRef.current.getContext('2d');
              let minX = 1, minY = 1, maxX = 0, maxY = 0;
              matchedFaceLandmarks.forEach(l => {
                if (l.x < minX) minX = l.x; if (l.y < minY) minY = l.y;
                if (l.x > maxX) maxX = l.x; if (l.y > maxY) maxY = l.y;
              });

              const w = maxX - minX;
              const h = maxY - minY;
              const pad = 0.10; // Tightened crop for better Specs/FER focus
              const sx = Math.max(0, (minX - w * pad) * video.videoWidth);
              const sy = Math.max(0, (minY - h * pad) * video.videoHeight);
              const sw = Math.min(video.videoWidth - sx, (w * (1 + 2 * pad)) * video.videoWidth);
              const sh = Math.min(video.videoHeight - sy, (h * (1 + 2 * pad)) * video.videoHeight);

              // Update the crop for ALL detectors
              cropCtx.clearRect(0, 0, 224, 224);
              cropCtx.drawImage(video, sx, sy, sw, sh, 0, 0, 224, 224);

              // 2. Specs Detection (Slow Throttle: 800ms)
              if (specsClassifier && matchedFaceId !== null && (!lastClassifyTimeRef.current[matchedFaceId] || now - lastClassifyTimeRef.current[matchedFaceId] > 800)) {
                lastClassifyTimeRef.current[matchedFaceId] = now;
                const classificationResult = specsClassifier.classify(cropCanvasRef.current);
                const topCategory = classificationResult.classifications[0].categories[0];
                console.log(`[Specs Insight] Face ${matchedFaceId}: Top Guess = "${topCategory.categoryName}" (Conf: ${topCategory.score.toFixed(2)})`);

                if (classificationResult && classificationResult.classifications && classificationResult.classifications.length > 0) {
                  isWearingSpecs = classificationResult.classifications[0].categories.some(cat =>
                    ['spectacle', 'specs', 'eyeglasses', 'sunglass', 'sunglasses'].some(s =>
                      cat.categoryName.toLowerCase().includes(s)
                    )
                  );
                  specsResultsRef.current[matchedFaceId] = isWearingSpecs;
                }
              }

              // Emotion Detection (FER+ 64x64 Grayscale) - Runs independently of the specs throttle
              if (emotionClassifierRef.current && matchedFaceLandmarks && matchedFaceId !== null && (!lastEmotionClassifyTimeRef.current[matchedFaceId] || now - lastEmotionClassifyTimeRef.current[matchedFaceId] > 100)) {
                lastEmotionClassifyTimeRef.current[matchedFaceId] = now;
                try {
                  window.tf.tidy(() => {
                    const faceTensor = window.tf.browser.fromPixels(cropCanvasRef.current)
                      .resizeNearestNeighbor([48, 48]) // FER+ size
                      .mean(2) // Grayscale
                      .expandDims(-1)
                      .toFloat()
                      .div(window.tf.scalar(255.0))
                      .expandDims(0);

                    const prediction = emotionClassifierRef.current.predict(faceTensor);
                    const data = prediction.dataSync();

                    const labels = ['Neutral', 'Happy', 'Surprise', 'Sad', 'Angry', 'Disgust', 'Fear', 'Contempt'];
                    const emojiMap = {
                      'Neutral': '😐', 'Happy': '😊', 'Surprise': '😲', 'Sad': '😢',
                      'Angry': '😠', 'Disgust': '🤢', 'Fear': '😨', 'Contempt': '😒'
                    };

                    let maxIdx = 0;
                    for (let j = 1; j < data.length; j++) {
                      if (data[j] > data[maxIdx]) maxIdx = j;
                    }

                    const label = labels[maxIdx] || 'Neutral';

                    // Save cleaned label without prefix or emoji
                    emotionResultsRef.current[matchedFaceId] = label;
                  });
                } catch (e) {
                  console.error("FER+ Real-time Error:", e);
                }
              }
            }

            // 1. Draw
            drawingUtilsRef.current.drawLandmarks(landmarks, {
            });
            drawingUtilsRef.current.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
              color: personColor,
              lineWidth: 4
            });

            currentStats.push({
              id: index + 1,
              color: personColor,
              ...poseData,
              emotion: assignedEmotion,
              isSquinting: isSquinting,
              specs: isWearingSpecs
            });
          });
          setDebugStatus(`Detecting: ${poseResults.landmarks.length} Person(s)`);
        } else {
          setDebugStatus("Looking for people...");
        }

        setResultsDisplay(currentStats);

      } catch (err) {
        console.error("Prediction Error:", err);
      }
    }

    requestRef.current = requestAnimationFrame(predictWebcam);
  };

  return (
    <div className="container">
      {/* Top Status Bar for direct feedback */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0,
        background: 'rgba(0,0,0,0.5)', color: 'lime',
        padding: '5px', zIndex: 100, fontSize: '12px', textAlign: 'center'
      }}>
        System Status: {debugStatus} | Resolutions: {videoRef.current?.videoWidth}x{videoRef.current?.videoHeight}
      </div>

      {loading && (
        <div className="loading">
          <div className="spinner"></div>
          <h2>Initializing AI...</h2>
        </div>
      )}

      {!cameraActive && !loading && (
        <div className="loading" style={{ background: 'transparent', backdropFilter: 'blur(10px)' }}>
          <button
            className="start-button"
            onClick={startCamera}
          >
            Start System
          </button>
        </div>
      )}

      {cameraActive && !emotionClassifierRef.current && emotionLoading && (
        <div style={{
          position: 'absolute', bottom: '100px', left: '50%', transform: 'translateX(-50%)',
          zIndex: 1000, color: 'gold', fontWeight: 'bold', textShadow: '0 0 5px black'
        }}>
          <div className="spinner" style={{ width: '20px', height: '20px', display: 'inline-block', marginRight: '10px' }}></div>
          Initializing FER+ Model...
        </div>
      )}

      <video
        ref={videoRef}
        id="webcam"
        autoPlay
        playsInline
        muted
        style={{ position: 'absolute', left: 0, top: 0, width: '100%', height: '100%' }}
      ></video>
      <canvas
        ref={canvasRef}
        id="output_canvas"
        style={{ position: 'absolute', left: 0, top: 0, width: '100%', height: '100%', pointerEvents: 'none' }}
      ></canvas>

      <div className="overlay">
        {resultsDisplay.map((person) => (
          <div key={person.id} className="person-card" style={{ borderTop: `4px solid ${person.color}` }}>
            <div className="person-title" style={{ color: person.color }}>Detected Person {person.id}</div>
            <div className="status-row">
              <span className="label">Posture</span>
              <span className={`value ${person.posture.toLowerCase()}`}>{person.posture}</span>
            </div>
            <div className="status-row">
              <span className="label">Arms</span>
              <span className="value" style={{ color: person.arms === 'Both' ? '#a7f3d0' : (person.arms === 'None' ? '#94a3b8' : '#38bdf8') }}>
                {person.arms}
              </span>
            </div>
            <div className="status-row">
              <span className="label">Emotion</span>
              <span className="value" style={{ color: '#fbbf24' }}>
                {person.emotion || "Scanning"}
              </span>
            </div>
            <div className="status-row">
              <span className="label">Specs</span>
              <span className="value" style={{ color: person.specs ? '#a7f3d0' : '#94a3b8' }}>
                {person.specs ? 'Detected' : 'None'}
              </span>
            </div>
            <div className="status-row">
              <span className="label">Eye Strain</span>
              <span className="value" style={{ color: person.isSquinting ? '#f87171' : '#94a3b8' }}>
                {person.isSquinting ? 'Strain Detected' : 'Comfortable'}
              </span>
            </div>
          </div>
        ))}
      </div>
      {/* Hidden canvas for face classification */}
      <canvas ref={cropCanvasRef} width={224} height={224} style={{ display: 'none' }} />
    </div>
  );
}

export default App;
