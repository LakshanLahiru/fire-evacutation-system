import cv2
import time
import threading
from ultralytics import YOLO
import numpy as np 
from typing import List, Tuple, Dict


class ThermalHumanDetector:
    def __init__(self, model_path="AI_models/yolov8n.pt"):
        self.model = YOLO(model_path)
        self.model.to("cpu") 
        self.stop_flag = {"video": False, "webcam": False}
        self.latest_count = {"video": 0, "webcam": 0}
        self.latest_fps = {"video": 0.0, "webcam": 0.0}
        self.lock = threading.Lock()
        self.threads = {"video": None, "webcam": None}
        self.status_lock = threading.Lock()  # Separate lock for status updates

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
        
        return person_count
        
    def detect_in_video_multi(self, video_id, video_path, status_store):
        """
        Process a video and update status_store with detection results.
        
        Args:
            video_id: Unique identifier for this video
            video_path: Path to the video file
            status_store: Shared dictionary to store status updates
        """
        print(f"Starting detection for video {video_id}")
        
        # Initialize stop flag if not exists
        if video_id not in self.stop_flag:
            self.stop_flag[video_id] = False

        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            status_store[video_id]["running"] = False
            status_store[video_id]["error"] = "Failed to open video"
            return

        prev_time = time.time()
        frame_count = 0

        try:
            while cap.isOpened() and not self.stop_flag[video_id]:
                ret, frame = cap.read()
                if not ret:
                    print(f"Video {video_id} finished or failed to read frame")
                    break

                frame_count += 1

                try:
                    # Run YOLO detection
                    results = self.model(frame, verbose=False)  # verbose=False to reduce console output
                except Exception as e:
                    print(f"YOLO ERROR on video {video_id}:", e)
                    break

                # Count persons (class 0 is 'person' in COCO dataset)
                person_count = sum(
                    1 for r in results for box in r.boxes if int(box.cls[0]) == 0
                )

                # Calculate FPS
                curr_time = time.time()
                fps = 1 / (curr_time - prev_time + 1e-8)
                prev_time = curr_time

                # Thread-safe update of status
                status_store[video_id]["count"] = person_count
                status_store[video_id]["fps"] = round(fps, 2)
                status_store[video_id]["running"] = True
                status_store[video_id]["frames_processed"] = frame_count

                # Apply thermal effect and draw info
                thermal_frame = self.apply_thermal_effect(frame)
                self.draw_info(thermal_frame, results, fps, source=video_id)

                # Display window (optional - can be disabled for headless servers)
                cv2.imshow(f"Thermal Detection {video_id}", thermal_frame)
                if cv2.waitKey(1) == ord("q"):
                    print(f"User pressed 'q' - stopping video {video_id}")
                    break

        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            status_store[video_id]["error"] = str(e)
        
        finally:
            # Clean up
            print(f"Stopping video {video_id}. Processed {frame_count} frames")
            status_store[video_id]["running"] = False
            cap.release()
            cv2.destroyWindow(f"Thermal Detection {video_id}")

    def detect_in_webcam(self):
        cap = cv2.VideoCapture(0)
        prev_time = time.time()
        self.stop_flag["webcam"] = False

        while cap.isOpened() and not self.stop_flag["webcam"]:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.model(frame, verbose=False)
            thermal_frame = self.apply_thermal_effect(frame)
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time + 1e-8)
            prev_time = curr_time
            self.draw_info(thermal_frame, results, fps, source="webcam")
            cv2.imshow("Thermal Human Detection - Webcam", thermal_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def detect(self, frame: np.ndarray, conf_thresh: float = 0.3) -> List[Dict]:
 
        results = self.model(frame, verbose=False)
        out = []
        if len(results) == 0:
            return out
        r = results[0]
        
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id != 0:  
                continue
            if conf < conf_thresh:
                continue
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            out.append({"bbox": tuple(xyxy), "conf": conf})
        return out

    def stop_all(self):
        """Stop all running detection threads"""
        print("Stopping all detections...")
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
        
        cv2.destroyAllWindows()