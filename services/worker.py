# services/worker.py
import cv2
import time
import numpy as np
from typing import Dict, Any, Optional
from services.detector import YoloDetector
from services.reid import ReIDExtractor
from services.tracker import DeepSortWrapper
from services.identity_manager import IdentityManager
import threading

class CameraWorker:
    """
    Per-camera/video worker: reads video source, runs detection, extracts embeddings,
    matches to global identities, updates tracker, and writes status to shared structure.
    
    Now supports global Re-ID across multiple videos.
    """

    def __init__(self, cam_name: str, source: str, shared_status: Dict[str, Any],
                 lock: threading.Lock, area_m2: float = 1.0,
                 detector: YoloDetector = None, reid: ReIDExtractor = None, 
                 tracker: DeepSortWrapper = None, identity_manager: IdentityManager = None):
        self.cam_name = cam_name
        self.source = source  # file path or device index (string)
        self.shared_status = shared_status
        self.lock = lock
        self.area_m2 = float(area_m2)
        self.detector = detector or YoloDetector()
        self.reid = reid or ReIDExtractor()
        self.tracker = tracker or DeepSortWrapper()
        self.identity_manager = identity_manager  # Global identity manager
        self._stop = False
        self._thread = None
        
        # Local tracking: map local track_id -> global_id
        self.track_to_global: Dict[int, int] = {}
        
        # Optimization: Process Re-ID features only every N frames
        self.reid_frame_skip = 5  # Extract features every 5 frames (faster processing)

    def start(self):
        self._stop = False
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=2)

    def run(self):
        cap = cv2.VideoCapture(self.source)
        prev_time = time.time()
        frames = 0

        while cap.isOpened() and not self._stop:
            ret, frame = cap.read()
            if not ret:
                break
            frames += 1

            # Create a copy for display
            display_frame = frame.copy()

            # Detection (always run)
            dets = self.detector.detect(frame, conf_thresh=0.4)  # Slightly higher threshold for speed
            
            # Extract Re-ID features only every N frames (optimization)
            extract_reid = (frames % self.reid_frame_skip == 0)
            
            for d in dets:
                x1, y1, x2, y2 = d["bbox"]
                
                # Only extract features periodically for speed
                if extract_reid:
                    # safety crop
                    h, w = frame.shape[:2]
                    x1c, y1c = max(0, x1), max(0, y1)
                    x2c, y2c = min(w - 1, x2), min(h - 1, y2)
                    if x2c <= x1c or y2c <= y1c:
                        d["feature"] = None
                        continue
                    crop = frame[y1c:y2c, x1c:x2c]
                    feat = None
                    try:
                        feat = self.reid.extract(crop)
                    except Exception as e:
                        feat = None
                    d["feature"] = feat
                else:
                    d["feature"] = None  # Skip feature extraction this frame

            # Update DeepSORT tracker with detections + features
            tracks = self.tracker.update(dets, frame)

            # Match each track to global identity
            tracks_with_global_ids = []
            for t in tracks:
                track_id = t["track_id"]
                feature = t.get("feature")
                
                # Check if we already have a global ID for this track
                if track_id in self.track_to_global:
                    global_id = self.track_to_global[track_id]
                else:
                    # Try to match to global identity using Re-ID feature
                    global_id = None
                    if feature is not None and self.identity_manager is not None:
                        global_id, is_new, similarity = self.identity_manager.register_or_match(
                            feature, self.cam_name
                        )
                        self.track_to_global[track_id] = global_id
                    else:
                        # No feature - use local track ID temporarily
                        # Will be updated when features are extracted
                        global_id = track_id
                        if extract_reid and self.identity_manager is not None:
                            # Store for next feature extraction
                            self.track_to_global[track_id] = global_id
                
                tracks_with_global_ids.append({
                    "local_track_id": track_id,
                    "global_id": global_id,
                    "bbox": t["bbox"],
                    "conf": t.get("conf", 0.0)
                })

            # DRAW PERSON IDs ON VIDEO
            for t in tracks_with_global_ids:
                x1, y1, x2, y2 = t["bbox"]
                global_id = t["global_id"]
                conf = t["conf"]
                
                # Generate consistent color for this ID
                # Ensure global_id is integer
                id_num = int(global_id) if isinstance(global_id, (int, float, str)) else hash(str(global_id))
                
                # Use hash-based color generation (consistent per ID)
                color_r = (id_num * 67) % 156 + 100
                color_g = (id_num * 131) % 156 + 100
                color_b = (id_num * 199) % 156 + 100
                color = (color_b, color_g, color_r)  # BGR format for OpenCV
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw ID label with background
                label = f"ID: {global_id}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                
                # Draw background rectangle for text
                cv2.rectangle(display_frame, 
                            (x1, y1 - label_size[1] - 10), 
                            (x1 + label_size[0] + 10, y1),
                            color, -1)
                
                # Draw text
                cv2.putText(display_frame, label, (x1 + 5, y1 - 5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            # Compute metrics
            person_count = len(tracks_with_global_ids)
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time + 1e-8)
            prev_time = curr_time
            density = person_count / (self.area_m2 + 1e-9)

            # Get unique global IDs in this frame
            unique_global_ids = list(set(t["global_id"] for t in tracks_with_global_ids))

            # Draw overall statistics on frame
            cv2.putText(display_frame, f"Video: {self.cam_name}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(display_frame, f"Persons: {person_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"Unique IDs: {len(unique_global_ids)}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

            # Resize display frame to 1/4 size for smaller window
            display_h, display_w = display_frame.shape[:2]
            small_frame = cv2.resize(display_frame, (display_w // 2, display_h // 2), interpolation=cv2.INTER_LINEAR)
            
            # Display the resized video with IDs
            cv2.imshow(f"Re-ID: {self.cam_name}", small_frame)
            
            # Press 'q' to stop this video
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print(f"User pressed 'q' - stopping {self.cam_name}")
                break

            # build status for this camera
            status = {
                "camera": self.cam_name,
                "persons": person_count,
                "unique_identities": len(unique_global_ids),
                "global_ids": unique_global_ids,
                "tracks": tracks_with_global_ids,
                "fps": round(fps, 2),
                "density": round(density, 4),
                "frames_processed": frames,
                "running": True,
                "last_update": time.time()
            }

            # write to shared_status thread-safely
            with self.lock:
                self.shared_status[self.cam_name] = status

        cap.release()
        cv2.destroyWindow(f"Re-ID: {self.cam_name}")
        
        # mark stopped
        with self.lock:
            s = self.shared_status.get(self.cam_name, {})
            s.update({"running": False})
            self.shared_status[self.cam_name] = s
