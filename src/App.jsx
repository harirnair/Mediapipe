import { useEffect, useRef, useState } from 'react';
import { PoseLandmarker, FaceLandmarker, HandLandmarker, ImageClassifier, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';
import './App.css';

// Note: tf and tflite are loaded via CDN in index.html for better compatibility with Vite

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [poseLandmarker, setPoseLandmarker] = useState(null);
  const [faceLandmarker, setFaceLandmarker] = useState(null);
  const [handLandmarker, setHandLandmarker] = useState(null);
  const [loading, setLoading] = useState(true);
  const [cameraActive, setCameraActive] = useState(false);
  const [specsClassifier, setSpecsClassifier] = useState(null);
  const [visionResolver, setVisionResolver] = useState(null);
  const [resultsDisplay, setResultsDisplay] = useState([]);
  const [debugStatus, setDebugStatus] = useState("Initializing...");
  const cropCanvasRef = useRef(null);
  const lastClassifyTimeRef = useRef({});
  const specsResultsRef = useRef({});
  const emotionClassifierRef = useRef(null);
  const emotionResultsRef = useRef({});

  // New state for ATM flow
  const [currentScreen, setCurrentScreen] = useState('landing'); // 'landing', 'main', 'pin', 'menu', 'withdraw', 'balance', 'account-balance', 'confirm', 'success'
  const [pinValue, setPinValue] = useState('');
  const [selectedAmount, setSelectedAmount] = useState(0);
  const [selectedAccount, setSelectedAccount] = useState('');
  const [transactionType, setTransactionType] = useState('');
  const [accountBalances, setAccountBalances] = useState({
    savings: 25000,
    current: 15000
  });
  const [customAmount, setCustomAmount] = useState('');
  const [handTracking, setHandTracking] = useState(false);
  const [cursorPosition, setCursorPosition] = useState({ x: 0, y: 0 });
  const [clickGestureDetected, setClickGestureDetected] = useState(false);
  const [handVisible, setHandVisible] = useState(false);
  const virtualCursorRef = useRef(null);
  const smoothedCursorRef = useRef({ x: 0, y: 0 });
  const SMOOTHING_FACTOR = 0.15;
  const frameCountRef = useRef(0);
  const [detectionsReady, setDetectionsReady] = useState(false);

  const drawingUtilsRef = useRef(null);
  const requestRef = useRef(null);
  const wasPinchingRef = useRef(false);
  const lastClickTimeRef = useRef(0);
  const lastVideoTimeRef = useRef(-1);
  const lastEmotionClassifyTimeRef = useRef({});

  // Load MediaPipe Models
  useEffect(() => {
    const createModels = async () => {
      try {
        setDebugStatus("Loading Vision Tasks...");
        const vision = await FilesetResolver.forVisionTasks(
          "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.0/wasm"
        );

        setDebugStatus("Loading Pose Model...");
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

        setDebugStatus("Loading Hand Model...");
        const hand = await HandLandmarker.createFromOptions(vision, {
          baseOptions: {
            modelAssetPath: `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`,
            delegate: "GPU"
          },
          runningMode: "VIDEO",
          numHands: 2,
          minHandDetectionConfidence: 0.8,
          minHandPresenceConfidence: 0.8,
          minTrackingConfidence: 0.8
        });
        setHandLandmarker(hand);
        console.log("Hand landmarker initialized successfully");

        setVisionResolver(vision); // Keep for on-demand tools

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

        // Auto-load Emotion CPU Model (High Compatibility)
        setDebugStatus("Loading Emotion Model...");
        window.tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.7/dist/');
        const response = await fetch('/models/emotion_model.tflite');
        const blob = await response.blob();
        const modelUrl = URL.createObjectURL(blob);
        const eModel = await window.tflite.loadTFLiteModel(modelUrl, { numThreads: 1 });
        emotionClassifierRef.current = eModel;
        URL.revokeObjectURL(modelUrl);

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
    if (!poseLandmarker || !faceLandmarker || !handLandmarker) {
      console.log("Cannot start camera - missing landmarkers:", {
        pose: !!poseLandmarker,
        face: !!faceLandmarker,
        hand: !!handLandmarker
      });
      return;
    }

    console.log("Starting camera with all landmarkers ready");
    setCameraActive(true);
    setDebugStatus("Starting Camera...");

    navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } })
      .then((stream) => {
        if (videoRef.current) {
          console.log("Got camera stream, setting up video element");
          videoRef.current.srcObject = stream;

          // Add multiple event listeners to ensure we catch when video is ready
          const onVideoReady = () => {
            console.log("Camera stream loaded, starting detection loop");
            setDebugStatus("Camera Live. Starting Detection...");
            predictWebcam();
          };

          videoRef.current.onloadeddata = onVideoReady;
          videoRef.current.oncanplay = onVideoReady;
          videoRef.current.onloadedmetadata = () => {
            console.log("Video metadata loaded");
          };

          // Fallback timeout in case events don't fire
          setTimeout(() => {
            if (videoRef.current && videoRef.current.readyState >= 2) {
              console.log("Video ready via timeout fallback");
              onVideoReady();
            } else {
              console.log("Video not ready after timeout, readyState:", videoRef.current?.readyState);
            }
          }, 2000);
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

    // --- MediaPipe Logic (Direct Analysis) ---
    const browDown = (score('browDownLeft') + score('browDownRight')) / 2;
    const squint = (score('eyeSquintLeft') + score('eyeSquintRight')) / 2;
    const browInnerUp = score('browInnerUp');
    const browAsymmetry = Math.abs(score('browOuterUpLeft') - score('browOuterUpRight'));

    return {
      isSquinting: squint > 0.3,
      isConfused: (browInnerUp > 0.25 && browDown > 0.15) || (browAsymmetry > 0.15)
    };
  };

  // Hand gesture analysis for cursor control and clicking
  const analyzeHand = (landmarks) => {
    if (!landmarks || landmarks.length === 0) {
      console.log("No landmarks provided to analyzeHand");
      return null;
    }

    const hand = landmarks[0]; // Use first detected hand

    // Get key landmarks
    const wrist = hand[0];      // Wrist - stable point for cursor
    const indexTip = hand[8];   // Index finger tip
    const indexPip = hand[6];   // Index finger PIP joint
    const thumbTip = hand[4];   // Thumb tip
    const thumbIp = hand[3];    // Thumb IP joint
    const middleTip = hand[12]; // Middle finger tip
    const middlePip = hand[10]; // Middle finger PIP joint

    // Calculate if fingers are extended
    const indexExtended = indexTip.y < indexPip.y;
    const thumbExtended = thumbTip.x > thumbIp.x; // Assuming right hand
    const middleExtended = middleTip.y < middlePip.y;

    // Determine gesture using 3D distance
    // Using simple Euclidean distance for pinch (Thumb tip to Index tip)
    const dx = thumbTip.x - indexTip.x;
    const dy = thumbTip.y - indexTip.y;
    const dz = thumbTip.z - indexTip.z;
    const pinchDistance3D = Math.sqrt(dx * dx + dy * dy + dz * dz);

    // Tightened threshold: 0.045 requires fingers to be much closer,
    // reducing accidental triggers during movement.
    const isPinching = pinchDistance3D < 0.045;
    const isPointing = indexExtended && !middleExtended;
    const isOpen = indexExtended && middleExtended && thumbExtended;

    return {
      cursorPosition: {
        x: 1 - wrist.x, // Mirror horizontal axis (1 - x flips it)
        y: wrist.y      // Keep vertical as is
      },
      isPointing,
      isOpen,
      isPinching,
      pinchDistance: pinchDistance3D
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
    if (!videoRef.current || !canvasRef.current || !poseLandmarker || !faceLandmarker || !handLandmarker) {
      console.log("Missing components:", {
        video: !!videoRef.current,
        canvas: !!canvasRef.current,
        pose: !!poseLandmarker,
        face: !!faceLandmarker,
        hand: !!handLandmarker
      });
      return;
    }

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
        frameCountRef.current++;

        // 1. Hand Tracking - Every Frame
        const handResults = handLandmarker.detectForVideo(video, startTimeMs);

        // 2. Pose & Face - Every Other Frame (Cached for performance)
        if (frameCountRef.current % 2 === 0 || !window._lastPoseResults) {
          window._lastPoseResults = poseLandmarker.detectForVideo(video, startTimeMs);
          window._lastFaceResults = faceLandmarker.detectForVideo(video, startTimeMs);
        }
        const poseResults = window._lastPoseResults;
        const faceResults = window._lastFaceResults;

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const currentStats = [];

        // Process hand tracking for cursor control
        if (handResults.landmarks && handResults.landmarks.length > 0) {
          if (!detectionsReady) setDetectionsReady(true);
          const handAnalysis = analyzeHand(handResults.landmarks);
          if (handAnalysis) {
            const targetX = handAnalysis.cursorPosition.x * window.innerWidth;
            const targetY = handAnalysis.cursorPosition.y * window.innerHeight;

            if (!handTracking) {
              setHandTracking(true);
              setHandVisible(true);
              smoothedCursorRef.current = { x: targetX, y: targetY };
            } else {
              const smoothedX = (targetX * SMOOTHING_FACTOR) + (smoothedCursorRef.current.x * (1 - SMOOTHING_FACTOR));
              const smoothedY = (targetY * SMOOTHING_FACTOR) + (smoothedCursorRef.current.y * (1 - SMOOTHING_FACTOR));
              smoothedCursorRef.current = { x: smoothedX, y: smoothedY };
            }

            // Update DOM directly for zero-latency cursor movement
            if (virtualCursorRef.current) {
              virtualCursorRef.current.style.left = `${smoothedCursorRef.current.x}px`;
              virtualCursorRef.current.style.top = `${smoothedCursorRef.current.y}px`;
              virtualCursorRef.current.style.backgroundColor = handAnalysis.isPinching ? '#ff0000' : '#00ff00';
              virtualCursorRef.current.style.boxShadow = handAnalysis.isPinching ? '0 0 15px rgba(255,0,0,0.8)' : '0 0 10px rgba(0,255,0,0.5)';
            }

            // CLICK LOGIC: Detect the exact moment the hand transitions to a pinch (Rising Edge)
            const isPinchingNow = handAnalysis.isPinching;
            const now = performance.now();

            // Only trigger if:
            // 1. We just started pinching (wasPinchingRef was false)
            // 2. Cooldown of 0.5 seconds (500ms) has passed since last click
            if (isPinchingNow && !wasPinchingRef.current && (now - lastClickTimeRef.current > 500)) {
              lastClickTimeRef.current = now;
              setClickGestureDetected(true); // Update state for UI
              setCursorPosition(smoothedCursorRef.current);

              const el = document.elementFromPoint(smoothedCursorRef.current.x, smoothedCursorRef.current.y);
              if (el) {
                const button = el.closest('button');
                if (button) {
                  button.click();
                } else {
                  el.click();
                }
              }
            } else if (!isPinchingNow && wasPinchingRef.current) {
              // Reset the detected state as soon as hand is opened
              setClickGestureDetected(false);
            }

            wasPinchingRef.current = isPinchingNow;

            // Draw hand landmarks
            handResults.landmarks.forEach((landmarks) => {
              drawingUtilsRef.current.drawLandmarks(landmarks, {
                color: '#00ff00',
                lineWidth: 2
              });
              drawingUtilsRef.current.drawConnectors(landmarks, HandLandmarker.HAND_CONNECTIONS, {
                color: '#00ff00',
                lineWidth: 2
              });
            });
          }
        } else {
          setHandVisible(false);
          setHandTracking(false);
          setClickGestureDetected(false);
          wasPinchingRef.current = false; // Reset pinch state when hand is lost
        }

        // Map Faces to Poses
        // We'll simplisticly match Face Center to Pose Nose
        const faces = faceResults.faceBlendshapes.map((shapes, i) => {
          const landmarks = faceResults.faceLandmarks[i];
          const nose = landmarks ? landmarks[1] : { x: 0, y: 0 };
          const faceAnalysis = analyzeFace(shapes);

          // Use FER+ if available, otherwise hide emotion
          const mappedEmotion = emotionResultsRef.current[i] || "---";

          return {
            id: i,
            nose,
            emotion: mappedEmotion,
            isSquinting: faceAnalysis.isSquinting,
            isConfused: faceAnalysis.isConfused
          };
        });

        if (poseResults.landmarks && poseResults.landmarks.length > 0) {
          if (!detectionsReady) setDetectionsReady(true);
          poseResults.landmarks.forEach((landmarks, index) => {
            const personColor = colors[0];

            // Analyze Pose
            const poseData = analyzePose(landmarks);

            // Find matching face
            let assignedEmotion = "Scanning...";
            let isSquinting = false;
            let isConfused = false;
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
                  isConfused = face.isConfused;
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

            let minX = 1, minY = 1, maxX = 0, maxY = 0;

            if (matchedFaceLandmarks) {
              const cropCtx = cropCanvasRef.current.getContext('2d');
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

              // 2. Specs Detection (Slower Throttle: 2000ms)
              if (specsClassifier && matchedFaceId !== null && (!lastClassifyTimeRef.current[matchedFaceId] || now - lastClassifyTimeRef.current[matchedFaceId] > 2000)) {
                lastClassifyTimeRef.current[matchedFaceId] = now;
                const classificationResult = specsClassifier.classify(cropCanvasRef.current);

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

            // Calculate prominence score (Face Area)
            const prominence = matchedFaceLandmarks ? (maxX - minX) * (maxY - minY) : 0;

            currentStats.push({
              id: index + 1,
              color: personColor,
              ...poseData,
              emotion: assignedEmotion,
              isSquinting: isSquinting,
              isConfused: isConfused,
              specs: isWearingSpecs,
              prominence: prominence,
              landmarks: landmarks, // Store for drawing later
            });
          });

          // Sort by prominence and pick the main person
          currentStats.sort((a, b) => b.prominence - a.prominence);
          const totalPeople = currentStats.length;
          const mainPerson = currentStats.length > 0 ? currentStats[0] : null;

          // Draw ONLY the main person's skeleton
          if (mainPerson && drawingUtilsRef.current) {
            drawingUtilsRef.current.drawLandmarks(mainPerson.landmarks);
            drawingUtilsRef.current.drawConnectors(mainPerson.landmarks, PoseLandmarker.POSE_CONNECTIONS, {
              color: mainPerson.color,
              lineWidth: 4
            });
          }

          // Throttle UI updates (cards/text) to every 3rd frame to reduce React re-render overhead
          if (frameCountRef.current % 3 === 0) {
            setResultsDisplay(mainPerson ? [{ ...mainPerson, totalPeople }] : []);
            setDebugStatus(totalPeople > 0 ? `Detecting: ${totalPeople} Person(s)` : "Looking for people...");
          }
        } else {
          if (frameCountRef.current % 10 === 0) {
            setResultsDisplay([]);
            setDebugStatus("Looking for people...");
          }
        }

      } catch (err) {
        console.error("Prediction Error:", err);
      }
    }

    requestRef.current = requestAnimationFrame(predictWebcam);
  };

  // PIN Screen Component
  const PinScreen = () => {
    const [lastClickTime, setLastClickTime] = useState(0);
    const [visibleDigits, setVisibleDigits] = useState(''); // Track which digits are visible
    const [digitTimers, setDigitTimers] = useState({}); // Track timers for each digit

    const handlePinClick = (digit) => {
      if (pinValue.length < 4) {
        const newPinValue = pinValue + digit;
        const digitIndex = pinValue.length;

        setPinValue(newPinValue);
        setVisibleDigits(prev => prev + digit);

        // Set timer to hide this digit after 1 second
        const timerId = setTimeout(() => {
          setVisibleDigits(prev => {
            const newVisible = prev.split('');
            newVisible[digitIndex] = '*';
            return newVisible.join('');
          });
        }, 1000);

        // Store timer reference
        setDigitTimers(prev => ({
          ...prev,
          [digitIndex]: timerId
        }));
      }
    };

    const handleClear = () => {
      // Clear all timers
      Object.values(digitTimers).forEach(timerId => clearTimeout(timerId));
      setDigitTimers({});

      setPinValue('');
      setVisibleDigits('');
    };

    const handleSubmit = () => {
      if (pinValue.length === 4) {
        // Simulate PIN validation
        console.log(`PIN Entered: ${pinValue}`);
        handleClear();
        setCurrentScreen('menu'); // Go to main menu instead of main screen
      }
    };

    const handleBackToMain = () => {
      // Clear all timers
      Object.values(digitTimers).forEach(timerId => clearTimeout(timerId));
      setDigitTimers({});

      setDetectionsReady(false); // Reset tracking status for next entry
      setCurrentScreen('landing');
      setPinValue('');
      setVisibleDigits('');
    };

    // Standard onClick handlers are now triggered directly by the global predictWebcam loop
    // No extra 'GestureClick' logic needed here as it was causing double-triggers

    return (
      <div className="pin-screen">
        <div className="pin-container">
          <h2>Enter PIN</h2>

          {/* PIN Display */}
          <div className="pin-display">
            {[0, 1, 2, 3].map(index => (
              <div
                key={index}
                className={`pin-digit ${index === pinValue.length - 1 && pinValue.length > 0 ? 'entering' : ''}`}
              >
                {index < pinValue.length ? (
                  visibleDigits[index] || '●'
                ) : '○'}
              </div>
            ))}
          </div>

          {/* Number Pad */}
          <div className="number-pad">
            {[1, 2, 3, 4, 5, 6, 7, 8, 9].map(digit => (
              <button
                key={digit}
                className="pin-button"
                onClick={() => handlePinClick(digit.toString())}
              >
                {digit}
              </button>
            ))}
            <button
              className="pin-button clear-btn"
              onClick={handleClear}
            >
              Clear
            </button>
            <button
              className="pin-button"
              onClick={() => handlePinClick('0')}
            >
              0
            </button>
            <button
              className="pin-button submit-btn"
              onClick={handleSubmit}
            >
              Enter
            </button>
          </div>

          <button
            className="back-button"
            onClick={handleBackToMain}
          >
            Back to Main
          </button>
        </div>

        {/* Hand Tracking Status */}
        <div className="hand-tracking-status">
          <div className="status-item">
            <span className="status-label">Hand Tracking:</span>
            <span className={`status-value ${handTracking ? 'active' : 'inactive'}`}>
              {handTracking ? 'ACTIVE' : 'INACTIVE'}
            </span>
          </div>
          <div className="status-item">
            <span className="status-label">Hand Visible:</span>
            <span className={`status-value ${handVisible ? 'active' : 'inactive'}`}>
              {handVisible ? 'YES' : 'NO'}
            </span>
          </div>
          <div className="status-item">
            <span className="status-label">Click Gesture:</span>
            <span className={`status-value ${clickGestureDetected ? 'detected' : 'not-detected'}`}>
              {clickGestureDetected ? 'DETECTED' : 'NOT DETECTED'}
            </span>
          </div>
          <div className="status-item">
            <span className="status-label">Cursor Position:</span>
            <span className="status-value">
              X: {Math.round(cursorPosition.x)}, Y: {Math.round(cursorPosition.y)}
            </span>
          </div>
        </div>
      </div>
    );
  };

  // Main Menu Screen
  const MainMenuScreen = () => {
    const menuOptions = [
      { id: 'withdraw', label: 'Withdraw Cash', icon: '💰' },
      { id: 'balance', label: 'Check Balance', icon: '📊' },
      { id: 'services', label: 'Other Services', icon: '⚙️' }
    ];

    const handleMenuClick = (optionId) => {
      setTransactionType(optionId);
      if (optionId === 'withdraw') {
        setTransactionType('withdraw');
        setCurrentScreen('withdraw-account');
      } else if (optionId === 'balance') {
        setCurrentScreen('balance');
      } else {
        setCurrentScreen('coming-soon');
      }
    };

    return (
      <div className="atm-screen">
        <div className="atm-container">
          <h2>Select Transaction</h2>
          <div className="menu-options">
            {menuOptions.map(option => (
              <button
                key={option.id}
                className="menu-button"
                onClick={() => handleMenuClick(option.id)}
              >
                <span className="menu-icon">{option.icon}</span>
                <span className="menu-label">{option.label}</span>
              </button>
            ))}
          </div>
          <button className="back-button" onClick={() => {
            setDetectionsReady(false);
            setCurrentScreen('landing');
          }}>
            Exit
          </button>
        </div>
        <div className="hand-tracking-status">
          <div className="status-item">
            <span className="status-label">Hand Tracking:</span>
            <span className={`status-value ${handTracking ? 'active' : 'inactive'}`}>
              {handTracking ? 'ACTIVE' : 'INACTIVE'}
            </span>
          </div>
          <div className="status-item">
            <span className="status-label">Click Gesture:</span>
            <span className={`status-value ${clickGestureDetected ? 'detected' : 'not-detected'}`}>
              {clickGestureDetected ? 'DETECTED' : 'NOT DETECTED'}
            </span>
          </div>
        </div>
      </div>
    );
  };

  // Withdraw Account Selection Screen
  const WithdrawAccountScreen = () => {
    const handleAccountSelect = (id) => {
      setSelectedAccount(id);
      setCurrentScreen('withdraw-amount');
    };

    return (
      <div className="atm-screen">
        <div className="atm-container">
          <h2>Select Account</h2>
          <div className="account-options">
            <button className="account-button" onClick={() => handleAccountSelect('savings')}>
              <span className="account-icon">💰</span>
              <span className="account-label">Savings Account</span>
            </button>
            <button className="account-button" onClick={() => handleAccountSelect('current')}>
              <span className="account-icon">💳</span>
              <span className="account-label">Current Account</span>
            </button>
          </div>
          <button className="back-button" onClick={() => setCurrentScreen('menu')}>
            Back
          </button>
        </div>
      </div>
    );
  };

  // Withdraw Amount Screen
  const WithdrawAmountScreen = () => {
    const amounts = [500, 1000, 2000, 5000, 10000, 'Other'];

    const handleAmountSelect = (amount) => {
      if (amount === 'Other') {
        setCustomAmount('');
        setCurrentScreen('other-amount');
      } else {
        // Check balance immediately
        if (amount > accountBalances[selectedAccount]) {
          setCurrentScreen('insufficient-balance');
        } else {
          setSelectedAmount(amount);
          setCurrentScreen('confirm');
        }
      }
    };

    return (
      <div className="atm-screen">
        <div className="atm-container">
          <h2>Select Amount</h2>
          <p style={{ color: '#94a3b8', marginBottom: '20px' }}>
            Account: <span style={{ color: '#38bdf8', fontWeight: 'bold' }}>
              {selectedAccount === 'savings' ? 'Savings' : 'Current'}
            </span>
          </p>
          <div className="amount-grid">
            {amounts.map(amount => (
              <button
                key={amount}
                className="amount-button"
                onClick={() => handleAmountSelect(amount)}
              >
                {amount === 'Other' ? amount : `₹${amount}`}
              </button>
            ))}
          </div>
          <button className="back-button" onClick={() => setCurrentScreen('withdraw-account')}>
            Back
          </button>
        </div>
      </div>
    );
  };

  // Balance Screen
  const BalanceScreen = () => {
    const accounts = [
      { id: 'savings', label: 'Savings Account', icon: '🏦' },
      { id: 'current', label: 'Current Account', icon: '💼' }
    ];

    const handleAccountSelect = (accountId) => {
      setSelectedAccount(accountId);
      setCurrentScreen('account-balance');
    };

    return (
      <div className="atm-screen">
        <div className="atm-container">
          <h2>Select Account</h2>
          <div className="account-options">
            {accounts.map(account => (
              <button
                key={account.id}
                className="account-button"
                onClick={() => handleAccountSelect(account.id)}
              >
                <span className="account-icon">{account.icon}</span>
                <span className="account-label">{account.label}</span>
              </button>
            ))}
          </div>
          <button className="back-button" onClick={() => setCurrentScreen('menu')}>
            Back
          </button>
        </div>
        <div className="hand-tracking-status">
          <div className="status-item">
            <span className="status-label">Hand Tracking:</span>
            <span className={`status-value ${handTracking ? 'active' : 'inactive'}`}>
              {handTracking ? 'ACTIVE' : 'INACTIVE'}
            </span>
          </div>
        </div>
      </div>
    );
  };

  // Other Amount Screen
  const OtherAmountScreen = () => {
    const handleNumberClick = (num) => {
      if (customAmount.length < 6) {
        setCustomAmount(prev => prev + num);
      }
    };

    const handleClear = () => setCustomAmount('');

    const handleConfirm = () => {
      const amount = parseInt(customAmount);
      if (amount && amount > 0) {
        if (amount > accountBalances[selectedAccount]) {
          setCurrentScreen('insufficient-balance');
        } else {
          setSelectedAmount(amount);
          setCurrentScreen('confirm');
        }
      }
    };

    return (
      <div className="atm-screen">
        <div className="atm-container">
          <h2>Enter Amount</h2>
          <div className="custom-amount-display">
            ₹{customAmount || '0'}
          </div>
          <div className="number-pad">
            {[1, 2, 3, 4, 5, 6, 7, 8, 9].map(num => (
              <button key={num} className="pin-button" onClick={() => handleNumberClick(num.toString())}>
                {num}
              </button>
            ))}
            <button className="pin-button clear-btn" onClick={handleClear}>Clear</button>
            <button className="pin-button" onClick={() => handleNumberClick('0')}>0</button>
            <button className="pin-button submit-btn" onClick={handleConfirm}>Confirm</button>
          </div>
          <button className="back-button" onClick={() => setCurrentScreen('withdraw-amount')}>Back</button>
        </div>
      </div>
    );
  };

  // Insufficient Balance Screen
  const InsufficientBalanceScreen = () => (
    <div className="atm-screen">
      <div className="atm-container">
        <div className="modal-icon" style={{ color: '#ef4444' }}>⚠️</div>
        <h2 style={{ color: '#ef4444' }}>Insufficient Funds</h2>
        <p style={{ color: '#94a3b8', marginBottom: '30px' }}>
          Your selected account does not have enough balance for this transaction.<br />
          Current Balance: <span style={{ color: '#22c55e', fontWeight: 'bold' }}>₹{accountBalances[selectedAccount].toLocaleString()}</span>
        </p>
        <button className="back-button" onClick={() => setCurrentScreen('withdraw-amount')}>
          Try Different Amount
        </button>
      </div>
    </div>
  );

  // Coming Soon Screen
  const ComingSoonScreen = () => (
    <div className="atm-screen">
      <div className="atm-container">
        <div className="modal-icon">⚙️</div>
        <h2>Service Coming Soon</h2>
        <p style={{ color: '#94a3b8', marginBottom: '30px' }}>
          This feature is currently being developed to provide you with the best experience.
        </p>
        <button className="back-button" onClick={() => setCurrentScreen('menu')}>
          Go Back
        </button>
      </div>
    </div>
  );

  // Account Balance Screen
  const AccountBalanceScreen = () => {
    const balance = accountBalances[selectedAccount];
    const accountName = selectedAccount === 'savings' ? 'Savings Account' : 'Current Account';

    return (
      <div className="atm-screen">
        <div className="atm-container">
          <h2>Account Balance</h2>
          <div className="balance-display">
            <div className="account-info">
              <h3>{accountName}</h3>
              <div className="balance-amount">₹{balance.toLocaleString()}</div>
            </div>
          </div>
          <button className="back-button" onClick={() => setCurrentScreen('balance')}>
            Back
          </button>
        </div>
        <div className="hand-tracking-status">
          <div className="status-item">
            <span className="status-label">Hand Tracking:</span>
            <span className={`status-value ${handTracking ? 'active' : 'inactive'}`}>
              {handTracking ? 'ACTIVE' : 'INACTIVE'}
            </span>
          </div>
        </div>
      </div>
    );
  };

  // Confirmation Screen
  const ConfirmationScreen = () => {
    const handleConfirm = () => {
      // Deduct from balance
      setAccountBalances(prev => ({
        ...prev,
        [selectedAccount]: prev[selectedAccount] - selectedAmount
      }));
      setCurrentScreen('success');
    };

    return (
      <div className="atm-screen">
        <div className="atm-container">
          <h2>Confirm Transaction</h2>
          <div className="confirmation-details">
            <div className="detail-row">
              <span>Transaction:</span>
              <span>Cash Withdrawal</span>
            </div>
            <div className="detail-row">
              <span>Amount:</span>
              <span>₹{selectedAmount.toLocaleString()}</span>
            </div>
            <div className="detail-row">
              <span>Account:</span>
              <span>{selectedAccount === 'savings' ? 'Savings Account' : 'Current Account'}</span>
            </div>
          </div>
          <div className="confirmation-buttons">
            <button className="confirm-button" onClick={handleConfirm}>
              Confirm
            </button>
            <button className="cancel-button" onClick={() => setCurrentScreen('withdraw-amount')}>
              Cancel
            </button>
          </div>
        </div>
        <div className="hand-tracking-status">
          <div className="status-item">
            <span className="status-label">Hand Tracking:</span>
            <span className={`status-value ${handTracking ? 'active' : 'inactive'}`}>
              {handTracking ? 'ACTIVE' : 'INACTIVE'}
            </span>
          </div>
        </div>
      </div>
    );
  };

  // Success Screen
  const SuccessScreen = () => {
    const handleComplete = () => {
      setCurrentScreen('landing');
      setSelectedAmount(0);
      setSelectedAccount('');
      setTransactionType('');
    };

    const handleBackToServices = () => {
      setCurrentScreen('menu');
      setSelectedAmount(0);
      setSelectedAccount('');
      setTransactionType('');
    };

    return (
      <div className="atm-screen">
        <div className="atm-container success-container">
          <div className="success-icon">✅</div>
          <h2>Transaction Successful</h2>
          <div className="success-details">
            <p>Your withdrawal of <strong>₹{selectedAmount.toLocaleString()}</strong> has been processed successfully.</p>
            <p>Please collect your cash from the dispenser.</p>
            <p>Thank you for using our ATM service!</p>
          </div>
          <div className="confirmation-buttons" style={{ marginTop: '30px' }}>
            <button className="confirm-button" onClick={handleBackToServices}>
              Back to Services
            </button>
            <button className="cancel-button" onClick={handleComplete}>
              Complete & Exit
            </button>
          </div>
        </div>
        <div className="hand-tracking-status">
          <div className="status-item">
            <span className="status-label">Hand Tracking:</span>
            <span className={`status-value ${handTracking ? 'active' : 'inactive'}`}>
              {handTracking ? 'ACTIVE' : 'INACTIVE'}
            </span>
          </div>
        </div>
      </div>
    );
  };

  return (
    <div className="container">
      {/* Virtual Cursor */}
      {handTracking && (
        <div
          ref={virtualCursorRef}
          className={`virtual-cursor ${clickGestureDetected ? 'clicking' : ''}`}
          style={{
            left: cursorPosition.x,
            top: cursorPosition.y,
            position: 'fixed',
            width: '20px',
            height: '20px',
            borderRadius: '50%',
            backgroundColor: clickGestureDetected ? '#ff0000' : '#00ff00',
            border: '2px solid #ffffff',
            pointerEvents: 'none',
            zIndex: 10000,
            transform: 'translate(-50%, -50%)',
            boxShadow: '0 0 10px rgba(0,255,0,0.5)'
          }}
        />
      )}

      {/* Top Status Bar for direct feedback */}
      <div style={{
        position: 'absolute', top: 0, left: 0, right: 0,
        background: 'rgba(0,0,0,0.5)', color: 'lime',
        padding: '5px', zIndex: 100, fontSize: '12px', textAlign: 'center'
      }}>
        System Status: {debugStatus} | Resolutions: {videoRef.current?.videoWidth}x{videoRef.current?.videoHeight} | Screen: {currentScreen}
      </div>

      {loading && (
        <div className="loading">
          <div className="spinner"></div>
          <h2>Initializing AI...</h2>
        </div>
      )}

      {/* New Sensor Warming Loader */}
      {!loading && currentScreen !== 'landing' && !detectionsReady && (
        <div className="loading" style={{ background: 'rgba(15, 23, 42, 0.94)' }}>
          <div className="spinner" style={{ borderTopColor: '#38bdf8' }}></div>
          <h2 style={{ color: '#38bdf8' }}>Sensor Warming...</h2>
          <p style={{ color: '#94a3b8' }}>{frameCountRef.current > 5 ? 'Please bring your hand or face into view' : 'Initializing tracking engine...'}</p>
        </div>
      )}

      {!loading && currentScreen === 'landing' && (
        <div className="landing-screen">
          <div className="landing-content">
            <h1 className="landing-title">Anticipatory Accessibility AI for ATMs & Kiosks</h1>
            <div className="landing-buttons">
              <button
                className="start-button intelligence-btn"
                onClick={() => {
                  setCurrentScreen('main');
                  if (!cameraActive) startCamera();
                }}
              >
                Anticipation Intelligence
              </button>
              <button
                className="start-button gesture-btn"
                onClick={() => {
                  console.log("Entering Gesture interaction mode - starting camera first");
                  setCurrentScreen('pin');
                  if (!cameraActive) startCamera();
                }}
              >
                Gesture Interaction
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Video and Canvas - Always present for camera stream */}
      <video
        ref={videoRef}
        id="webcam"
        autoPlay
        playsInline
        muted
        style={{
          position: 'absolute',
          left: 0,
          top: 0,
          width: '100%',
          height: '100%',
          visibility: currentScreen === 'pin' ? 'hidden' : 'visible'
        }}
      ></video>
      <canvas
        ref={canvasRef}
        id="output_canvas"
        style={{
          position: 'absolute',
          left: 0,
          top: 0,
          width: '100%',
          height: '100%',
          pointerEvents: 'none',
          visibility: currentScreen === 'pin' ? 'hidden' : 'visible'
        }}
      ></canvas>

      {/* Show appropriate screen based on current state */}
      {currentScreen === 'pin' ? (
        <PinScreen />
      ) : currentScreen === 'menu' ? (
        <MainMenuScreen />
      ) : currentScreen === 'withdraw-account' ? (
        <WithdrawAccountScreen />
      ) : currentScreen === 'withdraw-amount' ? (
        <WithdrawAmountScreen />
      ) : currentScreen === 'insufficient-balance' ? (
        <InsufficientBalanceScreen />
      ) : currentScreen === 'balance' ? (
        <BalanceScreen />
      ) : currentScreen === 'account-balance' ? (
        <AccountBalanceScreen />
      ) : currentScreen === 'other-amount' ? (
        <OtherAmountScreen />
      ) : currentScreen === 'coming-soon' ? (
        <ComingSoonScreen />
      ) : currentScreen === 'confirm' ? (
        <ConfirmationScreen />
      ) : currentScreen === 'success' ? (
        <SuccessScreen />
      ) : cameraActive ? (
        <div className="overlay">
          {currentScreen === 'main' && (
            <button
              className="back-button"
              onClick={() => setCurrentScreen('landing')}
              style={{
                marginBottom: '10px',
                background: 'rgba(15, 23, 42, 0.8)',
                borderColor: '#38bdf8',
                color: '#38bdf8',
                width: 'fit-content'
              }}
            >
              ← Back to Home
            </button>
          )}
          {resultsDisplay.map((person) => (
            <div key={person.id} className="person-card" style={{ borderTop: `4px solid ${person.color}` }}>
              <div className="person-title" style={{ color: person.color }}>
                Main Subject
              </div>
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
              <div className="status-row">
                <span className="label">Complexity</span>
                <span className="value" style={{ color: person.isConfused ? '#fca5a5' : '#94a3b8' }}>
                  {person.isConfused ? 'Confusion Detected' : 'Clear'}
                </span>
              </div>
              <div className="status-row">
                <span className="label">Multi-Person</span>
                <span className="value" style={{ color: person.totalPeople > 1 ? '#fca5a5' : '#94a3b8' }}>
                  {person.totalPeople > 1 ? 'Detected' : 'Not Detected'}
                </span>
              </div>
            </div>
          ))}
        </div>
      ) : null}

      {/* Hidden canvas for face classification */}
      <canvas ref={cropCanvasRef} width={224} height={224} style={{ display: 'none' }} />
    </div>
  );
}

export default App;
