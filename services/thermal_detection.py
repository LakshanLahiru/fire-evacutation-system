import cv2
import time
import threading
from ultralytics import YOLO

class ThermalHumanDetector:
    def __init__(self, model_path="AI_models/yolov8n.pt"):
        self.model = YOLO(model_path)
        self.stop_flag = {"video": False, "webcam": False}
        self.latest_count = {"video": 0, "webcam": 0}
        self.latest_fps = {"video": 0.0, "webcam": 0.0}
        self.lock = threading.Lock()
        self.threads = {"video": None, "webcam": None}

    def apply_thermal_effect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        thermal = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
        return thermal

    def get_status(self):
        with self.lock:
            return {
                "video": {
                    "persons": self.latest_count["video"],
                    "fps": self.latest_fps["video"]
                },
                "webcam": {
                    "persons": self.latest_count["webcam"],
                    "fps": self.latest_fps["webcam"]
                }
            }

    def draw_info(self, frame, results, fps, source):
        person_count = 0
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if self.model.names[cls_id] == "person":
                    person_count += 1
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame, f"Person {person_count} ({conf:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        with self.lock:
            self.latest_count[source] = person_count
            self.latest_fps[source] = fps

        cv2.putText(frame, f"Total Persons: {person_count}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

    def detect_in_video(self, video_path):
        cap = cv2.VideoCapture(video_path)
        prev_time = time.time()
        self.stop_flag["video"] = False

        while cap.isOpened() and not self.stop_flag["video"]:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame)
            thermal_frame = self.apply_thermal_effect(frame)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            self.draw_info(thermal_frame, results, fps, source="video")
            cv2.imshow("Thermal Human Detection - Video", thermal_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def detect_in_webcam(self):
        cap = cv2.VideoCapture(0)
        prev_time = time.time()
        self.stop_flag["webcam"] = False

        while cap.isOpened() and not self.stop_flag["webcam"]:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame)
            thermal_frame = self.apply_thermal_effect(frame)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time)
            prev_time = curr_time
            self.draw_info(thermal_frame, results, fps, source="webcam")
            cv2.imshow("Thermal Human Detection - Webcam", thermal_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def stop_all(self):
        for key in self.stop_flag:
            self.stop_flag[key] = True

        for key in self.threads:
            thread = self.threads[key]
            if thread and thread.is_alive():
                thread.join(timeout=2)

        with self.lock:
            for key in self.latest_count:
                self.latest_count[key] = 0
                self.latest_fps[key] = 0.0
