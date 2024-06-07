import cv2
from ultralytics import YOLO
import time
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

class UltraYTicsGUI:
    def __init__(self, master):
        self.master = master
        master.title("UltraYTics Object Detection with GUI")

        # Create video display frames
        self.frame_left = tk.Frame(master, width=640, height=480)
        self.frame_left.grid(row=0, column=0, rowspan=2)
        self.left_canvas = tk.Canvas(self.frame_left, width=640, height=480)
        self.left_canvas.grid(row=0, column=0)

        self.frame_right = tk.Frame(master, width=640, height=480)
        self.frame_right.grid(row=0, column=1, rowspan=2)
        self.right_canvas = tk.Canvas(self.frame_right, width=640, height=480)
        self.right_canvas.grid(row=0, column=0)

        # Video and model paths (replace with yours)
        self.video_path = None
        self.model_path = "/Users/yarramsettinaresh/PycharmProjects/ai.goraka/aigoraka/rotationfanwater.pt"  # Replace with actual path

        # YOLO model, confidence threshold, and colors
        self.model = None
        self.confidence_threshold = 0.5
        self.colors = [(0, 0, 255), (0, 255, 0)]  # Red for class 0, Green for class 1

        # Buttons for video selection and prediction
        self.button_select_video = tk.Button(master, text="Select Video", command=self.select_video)
        self.button_select_video.grid(row=2, column=0)

        self.button_predict = tk.Button(master, text="Predict", command=self.toggle_prediction, state=tk.DISABLED)
        self.button_predict.grid(row=2, column=1)

        # Main loop flag
        self.running = False

    def select_video(self):
        self.video_path = filedialog.askopenfilename(title="Select Video", filetypes=[("MP4 files", "*.mp4")])
        if self.video_path:
            self.button_predict.config(state=tk.NORMAL)  # Enable prediction button if video selected

    def toggle_prediction(self):
        if self.running:
            self.running = False
            self.button_predict.config(text="Predict")
        else:
            self.running = True
            self.button_predict.config(text="Stop")
            self.run_prediction()

    def run_prediction(self):
        if not self.video_path or not self.model_path:
            print("Error: Please select a video and ensure the model path is correct.")
            return

        # Load YOLO model
        try:
            self.model = YOLO(self.model_path)
        except Exception as e:
            print(f"Error loading model: {e}")
            return

        # Open video capture
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error: Unable to open video capture")
            return

        self.process_frames(cap)

    def process_frames(self, cap):
        while self.running:
            ret, frame = cap.read()

            if not ret:
                print("Error: Unable to capture frame")
                break

            # Convert frame to RGB for UltraYTics model
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Perform object detection with prediction
            start_time = time.time()
            results = self.model(frame_rgb)
            end_time = time.time()
            prediction_duration = end_time - start_time
            print(f"Prediction duration: {prediction_duration:.4f} seconds")

            # Display frame with bounding boxes and labels
            frame_with_boxes = frame.copy()
            for detection in results:
                try:
                    conf = float(detection.boxes.conf[0])
                    if conf > self.confidence_threshold:
                        pos, text, box_color = self.process_detection(detection, prediction_duration)
                        (x1, y1), (x2, y2) = pos
                        cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), box_color, 2)
                        cv2.putText(frame_with_boxes, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)
                except Exception as e:
                    print(f"Error processing detection: {e}")

            self.display_frames(frame, frame_with_boxes)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def process_detection(self, detection, duration):
        COLOUR_MAP = {0: (0, 0, 255), 1: (0, 255, 0)}
        conf = float(detection.boxes.conf[0])
        class_id = int(detection.boxes.cls[0])  # Class ID
        name = detection.names[class_id]
        xmin, ymin, xmax, ymax = detection.boxes.xyxy[0]
        color = COLOUR_MAP[class_id]  # Default color for detections
        pos = (int(xmin), int(ymin)), (int(xmax), int(ymax))
        text = f"{name} {round(conf, 2) if conf is not None else ''} :{duration:.4f}s"
        return pos, text, color

    def display_frames(self, original_frame, processed_frame):
        original_image = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
        processed_image = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

        original_image = Image.fromarray(original_image)
        processed_image = Image.fromarray(processed_image)

        original_image_tk = ImageTk.PhotoImage(image=original_image)
        processed_image_tk = ImageTk.PhotoImage(image=processed_image)

        self.left_canvas.create_image(0, 0, anchor=tk.NW, image=original_image_tk)
        self.right_canvas.create_image(0, 0, anchor=tk.NW, image=processed_image_tk)

        self.master.update_idletasks()
        self.master.update()

def main():
    root = tk.Tk()
    gui = UltraYTicsGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()
