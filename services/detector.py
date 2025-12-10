# services/detector.py
from ultralytics import YOLO
import numpy as np
from typing import List, Tuple, Dict

class YoloDetector:
    def __init__(self, model_path="AI_models/yolov8n.pt"):
        self.model = YOLO(model_path)
        self.model.to("cpu") 

    def detect(self, frame: np.ndarray, conf_thresh: float = 0.3) -> List[Dict]:
        """
        Detect persons in a frame.
        Returns list of dicts: {bbox: (x1,y1,x2,y2), conf: float}
        """
        results = self.model(frame, verbose=False)
        out = []
        if len(results) == 0:
            return out
        r = results[0]
        # each box: xyxy, conf, cls
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if cls_id != 0:  # class 0 is person in COCO
                continue
            if conf < conf_thresh:
                continue
            xyxy = box.xyxy[0].cpu().numpy().astype(int).tolist()
            out.append({"bbox": tuple(xyxy), "conf": conf})
        return out
