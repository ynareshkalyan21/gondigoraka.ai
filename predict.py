# Created by yarramsettinaresh GORAKA DIGITAL PRIVATE LIMITED at 04/06/24

import traceback
import cv2
from ultralytics import YOLO
import time


def ultralytics_predict(model, frame):
        confidence_threshold =0.5
        start_time = time.time()  # Correct time function
        results = model(frame)  # Perform inference on each frame
        end_time = time.time()

        duration = end_time - start_time
        print(f"Prediction duration: {duration:.4f} seconds")
        duration = f"{duration:.4f} S"


        for detection in results:
                try:
                        conf = float(detection.boxes.conf[0])  # Confidence score (if available)
                        if conf > confidence_threshold:
                                return ultralytics(detection, duration)
                        # Process detections
                except Exception as e:
                        print(f"error: {e}")
                        return None, None, None, None
        return None, None, None, None
def ultralytics(detection, duration):
        COLOUR_MAP = {0: (0, 0, 255), 1: (0, 0, 255)}
        COLOUR_MAP = {
                0: (0, 0, 255),  # Red in BGR format
                1: (0, 255, 0)  # Green in BGR format
        }
        conf = float(detection.boxes.conf[0])
        class_id = int(detection.boxes.cls[0])  # Class ID
        name = detection.names[class_id]
        xmin, ymin, xmax, ymax = detection.boxes.xyxy[0]
        color = COLOUR_MAP[class_id]  # Default color for detections
        # Draw bounding box and label on the frame
        pos = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        text = f"{name} {round(conf,2) if conf is not None else ''} :{duration}"
        return conf, pos, text, color
video_path = "/Users/yarramsettinaresh/PycharmProjects/ai.goraka/aigoraka/WhatsApp Video 2024-05-31 at 6.39.42 PM.mp4"
video_path = "/Users/yarramsettinaresh/PycharmProjects/ai.goraka/aigoraka/myPondHoriMultiObjects-4.MOV"
model_path = "/Users/yarramsettinaresh/PycharmProjects/ai.goraka/aigoraka/rotationfanwater.pt"

try :
  import os
  if os.path.exists(video_path):
    print(f"video_path file size: {os.path.getsize(video_path)} bytes")
  else:
    print("video_path file does not exist.")

  # Replace with the correct path to your model file

  if os.path.exists(model_path):
    print(f"Model file size: {os.path.getsize(model_path)} bytes")
  else:
    print("Model file does not exist.")
  model = YOLO(model_path)
except Exception as e:
  print(f"Error loading model: {e}")

# Video path (replace with your video path)
# video_path = "WhatsApp Video 2024-05-31 at 6.39.42 PM.mp4"

# Open the video capture object
cap = cv2.VideoCapture(video_path)

# Process each frame (limited functionality - consider performance)
while True:
  # Capture frame-by-frame
  ret, frame = cap.read()

  # Check if frame is read correctly
  if not ret:
    print("Error: Unable to capture frame")
    break

  # Convert frame to RGB for Roboflow model (might need adjustments)
  frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  try:
    conf, pos, text, box_color = ultralytics_predict(model, frame_rgb)
    # box_color = (0, 0, 255)
    if conf:
      (x1, y1), (x2, y2) = pos
      cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
      cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color,
                2)
    cv2.imshow('Frame', frame)
  except Exception as e:
    traceback.print_exc()
    print(f"** {e}")
    conf, pos, text = None, None, None
  ultra_pos = pos
  if cv2.waitKey(1) & 0xFF == ord('q'):
    break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
