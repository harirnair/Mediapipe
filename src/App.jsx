import { useEffect, useRef, useState } from 'react';
import { PoseLandmarker, FaceLandmarker, FilesetResolver, DrawingUtils } from '@mediapipe/tasks-vision';
import './App.css';

function App() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [poseLandmarker, setPoseLandmarker] = useState(null);
  const [faceLandmarker, setFaceLandmarker] = useState(null);
  const [loading, setLoading] = useState(true);
  const [cameraActive, setCameraActive] = useState(false);
  const [resultsDisplay, setResultsDisplay] = useState([]);
  const [debugStatus, setDebugStatus] = useState("Initializing...");

  const drawingUtilsRef = useRef(null);
  const requestRef = useRef(null);
  const lastVideoTimeRef = useRef(-1);

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
          minPoseDetectionConfidence: 0.85,
          minPosePresenceConfidence: 0.85,
          minTrackingConfidence: 0.85,
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
          minFaceDetectionConfidence: 0.7,
          minFacePresenceConfidence: 0.7,
          minTrackingConfidence: 0.7
        });
        setFaceLandmarker(face);

        setLoading(false);
        setDebugStatus("Models Ready. Start Camera.");
      } catch (error) {
        console.error("Error loading MediaPipe:", error);
        setDebugStatus("Error: " + error.message);
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

  const calculateAngle = (a, b, c) => {
    if (!a || !b || !c) return 0;
    const radians = Math.atan2(c.y - b.y, c.x - b.x) - Math.atan2(a.y - b.y, a.x - b.x);
    let angle = Math.abs(radians * 180.0 / Math.PI);
    if (angle > 180.0) angle = 360 - angle;
    return angle;
  };

  const colors = ["#38bdf8", "#f472b6", "#34d399", "#fbbf24", "#a78bfa"];

  const analyzeFace = (blendshapes) => {
    // Logic for "Confused"
    // Features: Asymmetric brows, brow furrow (inner up/down), squinting

    const brands = blendshapes.categories;
    const getScore = (name) => brands.find(b => b.categoryName === name)?.score || 0;

    const browInnerUp = getScore('browInnerUp');
    const browDownLeft = getScore('browDownLeft');
    const browDownRight = getScore('browDownRight');
    const browOuterUpLeft = getScore('browOuterUpLeft');
    const browOuterUpRight = getScore('browOuterUpRight');
    const eyeSquintLeft = getScore('eyeSquintLeft');
    const eyeSquintRight = getScore('eyeSquintRight');
    const jawOpen = getScore('jawOpen');
    const mouthSmile = (getScore('mouthSmileLeft') + getScore('mouthSmileRight')) / 2;

    // 1. Asymmetry (One brow raised)
    const browAsymmetry = Math.abs(browOuterUpLeft - browOuterUpRight);

    // 2. Furrow (Brows down or inner up + squint)
    const furrowScore = (browDownLeft + browDownRight + browInnerUp) / 3;
    const squintScore = (eyeSquintLeft + eyeSquintRight) / 2;

    let emotion = "Neutral";
    let confidence = 0;

    // Confusion Heuristic
    if (browAsymmetry > 0.4) {
      emotion = "Confused? (Brow)";
      confidence = browAsymmetry;
    } else if (furrowScore > 0.5 && squintScore > 0.3 && mouthSmile < 0.2) {
      emotion = "Confused (Furrow)";
      confidence = furrowScore;
    } else if (mouthSmile > 0.5) {
      emotion = "Happy";
    } else if (jawOpen > 0.5) {
      emotion = "Surprised";
    }

    return emotion;
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
          // Calculate simplistic center from landmarks if needed, 
          // but here we just have blendshapes and landmarks.
          // Let's use landmark 1 (nose tip) from faceResults.faceLandmarks
          const landmarks = faceResults.faceLandmarks[i];
          const nose = landmarks ? landmarks[1] : { x: 0, y: 0 };
          return { id: i, nose, emotion: analyzeFace(shapes) };
        });

        if (poseResults.landmarks && poseResults.landmarks.length > 0) {
          poseResults.landmarks.forEach((landmarks, index) => {
            const personColor = colors[index % colors.length];

            // Analyze Pose
            const poseData = analyzePose(landmarks);

            // Find matching face
            // Find face closest to pose nose (landmarks[0])
            let assignedEmotion = "Scanning...";
            let minDist = 0.2; // Threshold for matching (normalized coords)

            if (poseData.nose && faces.length > 0) {
              faces.forEach(face => {
                const dist = Math.sqrt(
                  Math.pow(face.nose.x - poseData.nose.x, 2) +
                  Math.pow(face.nose.y - poseData.nose.y, 2)
                );
                if (dist < minDist) {
                  minDist = dist;
                  assignedEmotion = face.emotion;
                }
              });
            }

            // 1. Draw
            drawingUtilsRef.current.drawLandmarks(landmarks, {
              radius: (data) => DrawingUtils.lerp(data.from.z, -0.15, 0.1, 5, 1),
              color: "white",
              lineWidth: 2
            });
            drawingUtilsRef.current.drawConnectors(landmarks, PoseLandmarker.POSE_CONNECTIONS, {
              color: personColor, // Match ID color
              lineWidth: 4
            });

            currentStats.push({
              id: index + 1,
              color: personColor,
              ...poseData,
              emotion: assignedEmotion
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
          <h1>Pose + Emotion AI</h1>
          <button className="btn" onClick={enableCam}>
            Start Camera
          </button>
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
                {person.emotion}
              </span>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default App;
