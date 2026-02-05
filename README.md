# MediaPipe Demo with Hand Tracking & Gesture Control

This application combines pose detection, face analysis, emotion recognition, and hand tracking with gesture-based interaction.

## Features

### Main Screen
- **Pose Detection**: Real-time body pose tracking
- **Face Analysis**: Emotion recognition, eye strain detection, and specs detection
- **Hand Tracking**: Real-time hand landmark detection
- **Multi-person Detection**: Identifies and tracks multiple people

### PIN Entry Screen
- **ATM-style PIN Entry**: Secure 4-digit PIN input interface
- **Hand Gesture Control**: Use your hand as a cursor to navigate and click
- **Real-time Status Display**: Shows hand tracking status and gesture detection

## Hand Gesture Controls

### Cursor Movement
- **Point with Index Finger**: Move your index finger to control the virtual cursor
- The cursor follows your index finger tip position

### Click Gesture
- **Pinch Gesture**: Bring your thumb and index finger close together to simulate a click
- The virtual cursor turns red when a click gesture is detected
- Clicks are debounced to prevent accidental multiple clicks

### Status Indicators
- **Hand Tracking**: Shows if hand detection is active
- **Hand Visible**: Indicates if a hand is currently visible
- **Click Gesture**: Shows when pinch gesture is detected
- **Cursor Position**: Displays real-time X,Y coordinates

## How to Use

1. **Start the System**: Click "Start System" to enable camera and AI detection
2. **Enter PIN Mode**: Click the "Enter PIN" button in the top-right corner
3. **Use Hand Gestures**: 
   - Point with your index finger to move the cursor
   - Pinch thumb and index finger together over a button to click it
   - Enter your 4-digit PIN using gestures or mouse clicks
4. **Return to Main**: Use the "Back to Main" button to return to the main screen

## Technical Details

- **MediaPipe Integration**: Uses MediaPipe for hand, pose, and face detection
- **Real-time Processing**: All detection runs at video frame rate
- **Gesture Recognition**: Custom gesture analysis for cursor control and clicking
- **Responsive UI**: Adapts to different screen sizes and orientations

## Browser Requirements

- Modern browser with WebRTC support
- Camera access permissions
- Hardware acceleration recommended for optimal performance

## Getting Started

```bash
npm install
npm run dev
```

Open your browser to `http://localhost:5173` and allow camera access when prompted.