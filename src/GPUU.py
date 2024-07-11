import cv2
import torch
import numpy as np
from ultralytics import YOLO
import time
from flask import Flask, render_template, Response, jsonify

app = Flask(__name__)

# Check CUDA availability
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Initialize the YOLO model
try:
    model = YOLO("yolov8n.pt").to(device)
    print("YOLO model loaded successfully")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    model = None

# Load class names
class_names = model.names if model else {}

detection_logs = []

def preprocess_frame(frame):
    try:
        # Resize the frame
        frame_resized = cv2.resize(frame, (640, 640))
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        # Normalize the frame
        frame_normalized = frame_rgb.astype(np.float32) / 255.0
        # Convert to tensor and move to GPU
        frame_tensor = torch.from_numpy(frame_normalized).permute(2, 0, 1).unsqueeze(0).to(device)
        return frame_tensor
    except Exception as e:
        print(f"Error in preprocess_frame: {e}")
        return None

@torch.no_grad()
def detect_objects(frame_tensor):
    try:
        if model is None:
            return None
        return model(frame_tensor)
    except Exception as e:
        print(f"Error in detect_objects: {e}")
        return None

def generate_frames():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        print("Error: Could not open video source.")
        return

    frame_count = 0
    while True:
        success, frame = camera.read()
        if not success:
            print("Error: Failed to capture frame.")
            break
        
        frame_count += 1
        print(f"Processing frame {frame_count}")
        
        try:
            # Preprocess frame
            frame_tensor = preprocess_frame(frame)
            if frame_tensor is None:
                continue
            
            # Perform object detection
            results = detect_objects(frame_tensor)
            if results is None:
                continue
            
            # Process results
            boxes = results[0].boxes
            detections = []
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                class_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                detections.append({
                    "class": class_names.get(class_id, "Unknown"),
                    "confidence": round(conf, 2),
                    "box": [int(x1.item()), int(y1.item()), int(x2.item()), int(y2.item())]
                })
            
            print(f"Detected {len(detections)} objects")
            
            # Create log entry
            log_entry = {
                "timestamp": time.time(),
                "detections": detections
            }
            detection_logs.append(log_entry)
            
            # Limit the number of logs stored in memory
            if len(detection_logs) > 100000:
                detection_logs.pop(0)
            
            # Draw bounding boxes on frame
            for det in detections:
                x1, y1, x2, y2 = det['box']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{det['class']} {det['confidence']:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Encode frame
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            
            print("Yielding frame")
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        except Exception as e:
            print(f"Error during frame processing: {e}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear GPU memory
            continue

    camera.release()
    print("Camera released")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/logs')
def logs():
    return jsonify(detection_logs)

if __name__ == "__main__":
    print("Starting application...")
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        torch.backends.cudnn.benchmark = True
    
    print("Running Flask app...")
    app.run(debug=True, host='0.0.0.0', port=5000)