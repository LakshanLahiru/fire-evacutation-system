import numpy as np
from typing import Tuple
from services.grid import Grid

class FireModel:
    def __init__(self, grid: Grid):
        self.grid = grid
        self.intensity = np.zeros((grid.h, grid.w), dtype=float)
    
        self.intensity[grid.mat == 1] = -1.0
      
        self.current_stage = "initial"

    def ignite(self, positions):
     
        for (r, c) in positions:
            if self.intensity[r, c] >= 0:
                self.intensity[r, c] = 0.5

    def stage_update(self, stage: str):
    
      
        self.current_stage = stage
        if stage == "initial":
          
            self._diffuse(rate=0.05, steps=2)
        elif stage == "growth":
      
            self._diffuse(rate=0.12, steps=3)
            self._amplify(factor=1.3)
        elif stage == "spread":
    
            self._diffuse(rate=0.20, steps=4)
            self._amplify(factor=1.6)
        
        self.intensity = np.clip(self.intensity, 0.0, 1.0)

        self.intensity[self.grid.mat == 1] = -1.0

    def is_unsafe(self, r: int, c: int, threshold: float | None = None, buffer: int = 0) -> bool:

        if self.intensity[r, c] < 0:
            return True

        default_thresholds = {"initial": 0.35, "growth": 0.25, "spread": 0.20}
        t = threshold if threshold is not None else default_thresholds.get(self.current_stage, 0.3)
        rr0 = max(0, r - buffer)
        rr1 = min(self.grid.h - 1, r + buffer)
        cc0 = max(0, c - buffer)
        cc1 = min(self.grid.w - 1, c + buffer)
        region = self.intensity[rr0:rr1 + 1, cc0:cc1 + 1]
        return bool(np.any(region >= t))

    def _diffuse(self, rate=0.1, steps=1):
        """Diffuse fire intensity to neighbors"""
        for _ in range(steps):
            new = self.intensity.copy()
            h, w = self.intensity.shape
            
            for r in range(h):
                for c in range(w):
                    if self.intensity[r, c] < 0: 
                        continue
                    
                    current_intensity = self.intensity[r, c]
                    if current_intensity <= 0.01:
                        continue
                    
                   
                    nbrs = self.grid.neighbors(r, c)
                    valid_nbrs = [(nr, nc) for (nr, nc) in nbrs 
                                 if self.intensity[nr, nc] >= 0]
                    
                    if not valid_nbrs:
                        continue
                    
                
                    spread_amount = rate * current_intensity / len(valid_nbrs)
                    
                    for (nr, nc) in valid_nbrs:

                        if self.intensity[nr, nc] > 0:
                            new[nr, nc] += spread_amount * 1.5
                        else:
                            new[nr, nc] += spread_amount * 0.8
            
            self.intensity = new

    def _amplify(self, factor=1.2):

        mask = self.intensity > 0
        self.intensity[mask] *= factor

    def get_fire_penalty(self, r: int, c: int) -> float:

        if self.intensity[r, c] < 0:
            return float('inf')

        block_thresholds = {"initial": 0.60, "growth": 0.50, "spread": 0.40}
        if self.intensity[r, c] >= block_thresholds.get(self.current_stage, 0.55):
            return float('inf')

        max_penalty = 20.0
        return self.intensity[r, c] * max_penalty