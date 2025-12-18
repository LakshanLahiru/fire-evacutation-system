import cv2
import numpy as np
from typing import List, Dict

class FaceDetector:
    def __init__(self):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            raise FileNotFoundError("Face detection model not found. Please ensure OpenCV is properly installed.")
    
    def detect(self, frame: np.ndarray, conf_thresh: float = 0.5) -> List[Dict]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rect = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        faces = []
        for (x, y, w, h) in faces_rect:
            faces.append({
                "bbox": (x, y, x + w, y + h),
                "conf": 0.8
            })
        
        return faces

