# services/tracker.py
from deep_sort_realtime.deepsort_tracker import DeepSort
from typing import List, Tuple, Dict
import numpy as np

class DeepSortWrapper:
    def __init__(self, max_age=30, n_init=3):
        # metric 'cosine' default; max_iou_distance, max_cosine_distance could be tuned
        self.tracker = DeepSort(max_age=max_age, n_init=n_init)

    def update(self, detections: List[Dict], frame) -> List[Dict]:
        """
        detections: list of {"bbox":(x1,y1,x2,y2), "conf":float, "feature": np.array or None}
        returns list of tracks: {"track_id":int, "bbox":(x1,y1,x2,y2), "conf":float, "feature":np.array}
        """
        dets_for_tracker = []
        
        # Build a map of detection confidence for each track
        det_conf_map = {}
        for i, d in enumerate(detections):
            x1, y1, x2, y2 = d["bbox"]
            conf = d.get("conf", 1.0)
            det_conf_map[i] = conf
            
            # deep_sort_realtime expects bbox in [x, y, width, height]
            bw = x2 - x1
            bh = y2 - y1
            dets_for_tracker.append(([x1, y1, bw, bh], conf, d.get("feature", None)))

        tracks = self.tracker.update_tracks(dets_for_tracker, frame=frame)
        out = []
        
        for t in tracks:
            if not t.is_confirmed():
                continue
            tid = t.track_id
            ltrb = t.to_ltrb()  # left, top, right, bottom
            x1, y1, x2, y2 = map(int, ltrb)
            
            # Get confidence - use detection conf if available, else default
            conf = 0.0
            try:
                # Try to get det_conf from the track object
                if hasattr(t, 'det_conf') and t.det_conf is not None:
                    conf = t.det_conf
                elif hasattr(t, 'detection_confidence'):
                    conf = t.detection_confidence
                else:
                    conf = 0.8  # Default confidence for confirmed tracks
            except Exception:
                conf = 0.8
            
            # Get feature
            feat = None
            try:
                feat = t.get_det_feature()
            except Exception:
                feat = None
            
            out.append({
                "track_id": tid, 
                "bbox": (x1, y1, x2, y2), 
                "conf": conf,
                "feature": feat
            })
        return out

