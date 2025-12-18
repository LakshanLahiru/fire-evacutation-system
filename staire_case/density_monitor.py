import cv2
import time
import threading
from typing import Dict, Any, Optional
from services.detector import YoloDetector
from services.tracker import DeepSortWrapper

class StaircaseDensityMonitor:
    def __init__(self, staircase_area_m2: float = 10.0, density_threshold: float = 0.5):
        self.staircase_area_m2 = float(staircase_area_m2)
        self.density_threshold = float(density_threshold)
        self.detector = YoloDetector()
        self.tracker = DeepSortWrapper()
        self._stop = False
        self._thread = None
        self.status = {
            "running": False,
            "current_count": 0,
            "density": 0.0,
            "threshold": self.density_threshold,
            "reroute_signal": False,
            "fps": 0.0,
            "last_update": None
        }
        self.status_lock = threading.Lock()

    def start(self, source: str, shared_status: Dict[str, Any] = None):
        self._stop = False
        self.status["running"] = True
        if shared_status is None:
            shared_status = {}
        self.shared_status = shared_status
        self._thread = threading.Thread(target=self._run, args=(source,), daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=2)
        with self.status_lock:
            self.status["running"] = False

    def _run(self, source: str):
        cap = cv2.VideoCapture(source)
        prev_time = time.time()
        frames = 0
        
        if source.isdigit():
            window_name = f"Staircase Density Monitor - Camera {source}"
        else:
            window_name = f"Staircase Density Monitor - {source.split('/')[-1]}"

        while cap.isOpened() and not self._stop:
            ret, frame = cap.read()
            if not ret:
                break
            frames += 1

            display_frame = frame.copy()

            dets = self.detector.detect(frame, conf_thresh=0.4)
            
            h_frame, w_frame = frame.shape[:2]
            filtered_dets = []
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                width = x2 - x1
                height = y2 - y1
                
                if width < 40 or height < 60:
                    continue
                if width > w_frame * 0.6 or height > h_frame * 0.7:
                    continue
                
                aspect_ratio = height / (width + 1e-6)
                if aspect_ratio < 1.2 or aspect_ratio > 4.0:
                    continue
                
                bbox_area = width * height
                frame_area = h_frame * w_frame
                if bbox_area > frame_area * 0.4:
                    continue
                
                d["feature"] = None
                filtered_dets.append(d)
            
            tracks = self.tracker.update(filtered_dets, frame)
            current_count = len(tracks)
            
            density = current_count / (self.staircase_area_m2 + 1e-9)
            reroute_signal = density > self.density_threshold
            
            for t in tracks:
                x1, y1, x2, y2 = t["bbox"]
                track_id = t["track_id"]
                conf = t.get("conf", 0.0)
                
                try:
                    id_num = int(track_id) if isinstance(track_id, (int, float)) else int(str(track_id))
                except (ValueError, TypeError):
                    id_num = hash(str(track_id)) % 10000
                
                color_r = (id_num * 67) % 156 + 100
                color_g = (id_num * 131) % 156 + 100
                color_b = (id_num * 199) % 156 + 100
                color = (color_b, color_g, color_r)
                
                label = f"ID: {track_id}"
                
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(display_frame, 
                            (x1, y1 - label_size[1] - 8), 
                            (x1 + label_size[0] + 8, y1),
                            color, -1)
                cv2.putText(display_frame, label, (x1 + 4, y1 - 4),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-8)
            prev_time = curr_time

            info_y = 30
            cv2.putText(display_frame, "Staircase Density Monitor", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            info_y += 30
            
            cv2.putText(display_frame, f"People Count: {current_count}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            info_y += 30
            
            density_color = (0, 0, 255) if reroute_signal else (0, 255, 0)
            cv2.putText(display_frame, f"Density: {density:.3f} / {self.density_threshold:.3f}", 
                       (10, info_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, density_color, 2)
            info_y += 30
            
            reroute_text = "REROUTE SIGNAL: ACTIVE" if reroute_signal else "REROUTE SIGNAL: INACTIVE"
            reroute_color = (0, 0, 255) if reroute_signal else (0, 255, 0)
            cv2.putText(display_frame, reroute_text, (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, reroute_color, 2)
            info_y += 30
            
            cv2.putText(display_frame, f"Area: {self.staircase_area_m2} m2", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            info_y += 30
            
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, info_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

            display_h, display_w = display_frame.shape[:2]
            if display_w > 1280 or display_h > 720:
                scale = min(1280 / display_w, 720 / display_h)
                new_w = int(display_w * scale)
                new_h = int(display_h * scale)
                display_frame = cv2.resize(display_frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            cv2.imshow(window_name, display_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            with self.status_lock:
                self.status.update({
                    "current_count": current_count,
                    "density": round(density, 4),
                    "reroute_signal": reroute_signal,
                    "fps": round(fps, 2),
                    "last_update": time.time()
                })
                
                if hasattr(self, 'shared_status'):
                    self.shared_status.update(self.status)

        cap.release()
        cv2.destroyWindow(window_name)
        with self.status_lock:
            self.status["running"] = False
            if hasattr(self, 'shared_status'):
                self.shared_status.update(self.status)

    def get_status(self) -> Dict[str, Any]:
        with self.status_lock:
            return self.status.copy()

