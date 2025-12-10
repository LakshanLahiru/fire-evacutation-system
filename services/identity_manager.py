# services/identity_manager.py
"""
Global Identity Manager for Cross-Video Person Re-Identification
Based on EAAI2025 Re-ID evacuation paper methodology
"""
import numpy as np
import threading
from typing import Dict, Optional, Tuple, List
from collections import defaultdict
import time


class IdentityManager:
    """
    Manages global person identities across multiple videos/cameras.
    Uses cosine similarity on Re-ID feature vectors to match people.
    """
    
    def __init__(self, similarity_threshold: float = 0.6):
        """
        Args:
            similarity_threshold: Threshold for considering two features as same person (0-1)
                                 Higher = more strict matching
                                 Paper suggests 0.5-0.7 for OSNet features
        """
        self.similarity_threshold = similarity_threshold
        
        # Global identity database: {global_id: feature_vector}
        self.identity_db: Dict[int, np.ndarray] = {}
        
        # Track which videos have seen which global IDs
        self.video_identities: Dict[str, set] = defaultdict(set)  # {video_name: {global_ids}}
        
        # Metadata about each identity
        self.identity_metadata: Dict[int, dict] = {}  # {global_id: {first_seen, last_seen, count, videos}}
        
        # Next available ID
        self.next_id = 1
        
        # Thread lock for concurrent access
        self.lock = threading.Lock()
        
    def cosine_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """
        Compute cosine similarity between two feature vectors.
        Returns value in [0, 1] where 1 = identical
        """
        # Features should already be L2-normalized from ReIDExtractor
        dot_product = np.dot(feat1, feat2)
        # Clamp to [-1, 1] to handle numerical errors
        dot_product = np.clip(dot_product, -1.0, 1.0)
        # Convert from [-1, 1] to [0, 1]
        similarity = (dot_product + 1.0) / 2.0
        return float(similarity)
    
    def find_best_match(self, feature: np.ndarray) -> Tuple[Optional[int], float]:
        """
        Find the best matching identity in the database.
        
        Returns:
            (global_id, similarity) if match found above threshold
            (None, 0.0) if no match found
        """
        if len(self.identity_db) == 0:
            return None, 0.0
        
        best_id = None
        best_similarity = 0.0
        
        for gid, stored_feat in self.identity_db.items():
            sim = self.cosine_similarity(feature, stored_feat)
            if sim > best_similarity:
                best_similarity = sim
                best_id = gid
        
        if best_similarity >= self.similarity_threshold:
            return best_id, best_similarity
        else:
            return None, best_similarity
    
    def register_or_match(self, feature: np.ndarray, video_name: str) -> Tuple[int, bool, float]:
        """
        Register a new person or match to existing identity.
        
        Args:
            feature: L2-normalized Re-ID feature vector
            video_name: Name/ID of the video source
        
        Returns:
            (global_id, is_new, similarity)
            - global_id: The assigned or matched global ID
            - is_new: True if this is a new identity, False if matched to existing
            - similarity: Similarity score (0 if new, >threshold if matched)
        """
        with self.lock:
            # Try to find existing match
            matched_id, similarity = self.find_best_match(feature)
            
            if matched_id is not None:
                # Update the stored feature with running average (optional, for robustness)
                # This helps improve feature quality over multiple observations
                alpha = 0.3  # Weight for new feature
                old_feat = self.identity_db[matched_id]
                updated_feat = (1 - alpha) * old_feat + alpha * feature
                # Re-normalize
                updated_feat = updated_feat / (np.linalg.norm(updated_feat) + 1e-8)
                self.identity_db[matched_id] = updated_feat
                
                # Update metadata
                self.identity_metadata[matched_id]["last_seen"] = time.time()
                self.identity_metadata[matched_id]["count"] += 1
                self.identity_metadata[matched_id]["videos"].add(video_name)
                
                # Track in video
                self.video_identities[video_name].add(matched_id)
                
                return matched_id, False, similarity
            
            else:
                # Create new identity
                new_id = self.next_id
                self.next_id += 1
                
                self.identity_db[new_id] = feature.copy()
                self.identity_metadata[new_id] = {
                    "first_seen": time.time(),
                    "last_seen": time.time(),
                    "count": 1,
                    "videos": {video_name}
                }
                
                # Track in video
                self.video_identities[video_name].add(new_id)
                
                return new_id, True, 0.0
    
    def get_identity_info(self, global_id: int) -> Optional[dict]:
        """Get metadata about a specific identity."""
        with self.lock:
            if global_id in self.identity_metadata:
                return self.identity_metadata[global_id].copy()
            return None
    
    def get_video_identities(self, video_name: str) -> List[int]:
        """Get all identities seen in a specific video."""
        with self.lock:
            return list(self.video_identities.get(video_name, set()))
    
    def get_all_identities(self) -> Dict[int, dict]:
        """Get all identities and their metadata."""
        with self.lock:
            return {
                gid: {
                    **meta,
                    "videos": list(meta["videos"])  # Convert set to list for JSON
                }
                for gid, meta in self.identity_metadata.items()
            }
    
    def reset(self):
        """Clear all identities (useful for testing or reset)."""
        with self.lock:
            self.identity_db.clear()
            self.video_identities.clear()
            self.identity_metadata.clear()
            self.next_id = 1
    
    def get_statistics(self) -> dict:
        """Get overall system statistics."""
        with self.lock:
            total_identities = len(self.identity_db)
            total_videos = len(self.video_identities)
            
            # Count cross-video identities (people appearing in multiple videos)
            cross_video_count = sum(
                1 for meta in self.identity_metadata.values()
                if len(meta["videos"]) > 1
            )
            
            return {
                "total_identities": total_identities,
                "total_videos": total_videos,
                "cross_video_identities": cross_video_count,
                "similarity_threshold": self.similarity_threshold
            }

