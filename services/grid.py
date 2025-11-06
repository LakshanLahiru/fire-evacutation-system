# grid.py
from typing import List, Tuple, Optional
import numpy as np
import math

class Grid:
    """
    Grid representation.
    cell codes:
      0 -> free
      1 -> static obstacle (wall)
      2 -> dynamic obstacle / fire-affected (fire products)
      3 -> exit (goal)
      4 -> start (agent)
    """
    def __init__(self, matrix: List[List[int]]):
        self.mat = np.array(matrix, dtype=int)
        self.h, self.w = self.mat.shape

    @classmethod
    def from_txt(cls, path: str) -> "Grid":
        """
        Load grid from a text file where each line contains space-separated integers.
        Example row: "0 0 1 0 3"
        """
        with open(path, "r") as f:
            matrix = [[int(x) for x in line.strip().split()] for line in f if line.strip()]
        return cls(matrix)
    def neighbors(self, r: int, c: int) -> List[Tuple[int,int]]:
        """8-connected neighbors (including diagonals), excluding walls (1)."""
        nbrs = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1),
                       (-1,-1),(-1,1),(1,-1),(1,1)]:
            rr, cc = r+dr, c+dc
            if 0 <= rr < self.h and 0 <= cc < self.w and self.mat[rr,cc] != 1:
                nbrs.append((rr,cc))
        return nbrs
    def find_value(self, val: int) -> List[Tuple[int,int]]:
        locs = list(zip(*np.where(self.mat == val)))
        return locs

    def is_free(self, r: int, c: int) -> bool:
        return self.mat[r,c] in (0,3,4)

    def copy(self) -> "Grid":
        return Grid(self.mat.copy().tolist())

    @staticmethod
    def distance(a: Tuple[int,int], b: Tuple[int,int]) -> float:
        """Euclidean distance (1 for straight, √2 for diagonal)."""
        return math.hypot(a[0]-b[0], a[1]-b[1])
