import random
import math
import heapq
import numpy as np
from typing import List, Tuple, Dict, Optional
from services.grid import Grid
from services.fire_model import FireModel


class TurningPoint:
    def __init__(self,position:Tuple[int,int],step_index:int,direction:str,distance:float):
        self.position=position
        self.step_index=step_index
        self.direction=direction
        self.distance=distance

class NavigationInstruction:
    def __init__(self,instruction: str, turning_points: List[TurningPoint], segment_distance: float):
        self.instruction = instruction
        self.turning_points = turning_points
        self.segment_distance = segment_distance
    
    def __repr__(self):
        return f"{self.instruction} ({self.segment_distance:.2f}m)"

class AntColony:
    def __init__(self,
                 grid: Grid,
                 fire_model: FireModel,
                 start: Tuple[int,int],
                 exits: List[Tuple[int,int]],
                 m_ants: int = 30,
                 alpha: float = 1.0,
                 beta: float = 5.0,
                 rho: float = 0.5,
                 Q: float = 15.0,
                 max_iter: int = 50):
        self.grid = grid
        self.fire = fire_model
        self.start = start
        self.exits = exits
        self.m = m_ants
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.Q = Q
        self.max_iter = max_iter

        self.tau = np.ones((grid.h, grid.w), dtype=float) * 0.1
        self.best_path = None
        self.best_len = float('inf')

    def _is_valid_step(self, curr: tuple[int,int], nxt: tuple[int,int], buf: int) -> bool:
 
        if self.grid.mat[nxt] == 1:
            return False
        if self.fire.is_unsafe(nxt[0], nxt[1], buffer=buf):
            return False
        dr = nxt[0] - curr[0]
        dc = nxt[1] - curr[1]
        if dr != 0 and dc != 0: 
            ortho1 = (curr[0], curr[1] + dc)
            ortho2 = (curr[0] + dr, curr[1])

            if self.grid.mat[ortho1] == 1 or self.grid.mat[ortho2] == 1:
                return False
            if self.fire.is_unsafe(ortho1[0], ortho1[1], buffer=buf):
                return False
            if self.fire.is_unsafe(ortho2[0], ortho2[1], buffer=buf):
                return False
        return True

    def run(self):
        for it in range(self.max_iter):
            all_paths = []
            all_lens = []
            for _ in range(self.m):
                path, length = self._construct_solution()
                if path is not None:
                    all_paths.append(path)
                    all_lens.append(length)
                    if length < self.best_len:
                        self.best_len = length
                        self.best_path = path
            
            if all_paths:
                self._global_pheromone_update(all_paths, all_lens)
            

            if (it + 1) % 10 == 0:
                print(f"  Iteration {it+1}/{self.max_iter}, best length={self.best_len:.4f}")
        

        a_path, a_len = self._a_star()
        if self.best_path is None or math.isinf(self.best_len):
            return a_path, a_len
        if a_path is not None and a_len < self.best_len:
            return a_path, a_len
        return self.best_path, self.best_len

    def _construct_solution(self) -> Tuple[Optional[List[Tuple[int,int]]], float]:
        current = self.start
        visited = {current: 1}  
        path = [current]
        length = 0.0
        steps = 0
        max_steps = self.grid.h * self.grid.w  

        while steps < max_steps:
            if current in self.exits:
                return path, length
            

            raw_nbrs = self.grid.neighbors(*current)
            nbrs = []
            for n in raw_nbrs:

                buffer_by_stage = {"initial": 0, "growth": 1, "spread": 1}
                buf = buffer_by_stage.get(self.fire.current_stage, 0)
                if not self._is_valid_step(current, n, buf):
                    continue

                if n in visited:
                    continue
                nbrs.append(n)
            
            if not nbrs:
                return None, float('inf')
            

            probs = []
            for n in nbrs:
                if self.fire.intensity[n] < 0:  
                    probs.append(0.0)
                    continue
                
                tau = self.tau[n]
                eta = self._heuristic(n)
                
                
                visit_penalty = 1.0
                if n in visited:
                    visit_penalty = 0.05 ** visited[n]  
                
                val = (tau ** self.alpha) * (eta ** self.beta) * visit_penalty
                probs.append(max(val, 1e-12))
            
            s = sum(probs)
            if s < 1e-12:
                return None, float('inf')
            
           
            probs = [p/s for p in probs]
            
            
            epsilon = 0.15
            if random.random() > epsilon:

                idx = max(range(len(nbrs)), key=lambda i: probs[i])
                chosen = nbrs[idx]
            else:

                r = random.random()
                cum = 0.0
                chosen = nbrs[-1]
                for i, p in enumerate(probs):
                    cum += p
                    if r <= cum:
                        chosen = nbrs[i]
                        break
            

            step_cost = self._distance(current, chosen)
            fire_penalty = self.fire.get_fire_penalty(*chosen)
            
            if fire_penalty == float('inf'):
                return None, float('inf')
            

            turn_penalty = 0.0
            if len(path) >= 2:
                prev = path[-2]
                v1 = (current[0] - prev[0], current[1] - prev[1])
                v2 = (chosen[0] - current[0], chosen[1] - current[1])
                if v1 != v2:
                    turn_penalty = 0.15
            length += step_cost * (1.0 + fire_penalty) + turn_penalty
            path.append(chosen)
            visited[chosen] = visited.get(chosen, 0) + 1
            current = chosen
            steps += 1
            

            if length > self.best_len * 1.5 and self.best_len < float('inf'):  # Changed from *2 to *1.5
                return None, float('inf')
        
        return None, float('inf')

    def _heuristic(self, pos: Tuple[int,int]) -> float:


        min_dist = min(self._distance(pos, ex) for ex in self.exits)

        fire_penalty = self.fire.get_fire_penalty(*pos)
        
        if fire_penalty == float('inf'):
            return 1e-12
        

        k_by_stage = {"initial": 0.6, "growth": 1.0, "spread": 1.2}
        k = k_by_stage.get(self.fire.current_stage, 0.8)
        fire_factor = math.exp(-fire_penalty * k)
        

        h = (fire_factor + 1e-6) / (min_dist + 1.0)
        return h

    def _a_star(self) -> Tuple[Optional[List[Tuple[int,int]]], float]:

        start = self.start
        goals = set(self.exits)
        buffer_by_stage = {"initial": 0, "growth": 1, "spread": 1}
        buf = buffer_by_stage.get(self.fire.current_stage, 0)

        def h(n: Tuple[int,int]) -> float:
            return min(self._distance(n, g) for g in goals)

        open_heap: list[tuple[float, Tuple[int,int]]] = []
        heapq.heappush(open_heap, (0.0, start))
        g_cost = {start: 0.0}
        parent: dict[Tuple[int,int], Tuple[int,int]] = {}

        while open_heap:
            _, node = heapq.heappop(open_heap)
            if node in goals:

                path: list[Tuple[int,int]] = [node]
                while node in parent:
                    node = parent[node]
                    path.append(node)
                path.reverse()

                length = 0.0
                for i in range(1, len(path)):
                    step_cost = self._distance(path[i-1], path[i])
                    fire_pen = self.fire.get_fire_penalty(*path[i])
                    if fire_pen == float('inf'):
                        return None, float('inf')
                    length += step_cost * (1.0 + fire_pen)
                return path, length

            for n in self.grid.neighbors(*node):
                if not self._is_valid_step(node, n, buf):
                    continue
                tentative = g_cost[node] + self._distance(node, n) * (1.0 + self.fire.get_fire_penalty(*n))
                if math.isinf(tentative):
                    continue
                if tentative < g_cost.get(n, float('inf')):
                    g_cost[n] = tentative
                    parent[n] = node
                    f = tentative + h(n)
                    heapq.heappush(open_heap, (f, n))

        return None, float('inf')

    def _distance(self, a: Tuple[int,int], b: Tuple[int,int]) -> float:
  
        return math.hypot(a[0] - b[0], a[1] - b[1])

    def _global_pheromone_update(self, all_paths: List[List[Tuple[int,int]]], 
                                 all_lens: List[float]):
      

        self.tau *= (1 - self.rho)
        

        for path, length in zip(all_paths, all_lens):
            if length <= 0 or math.isinf(length):
                continue
            

            delta = self.Q / length

            if path == self.best_path:
                delta *= 3.0  
            
            for (r, c) in path:
                self.tau[r, c] += delta

        self.tau = np.clip(self.tau, 0.01, 10.0)

    def identify_turning_points(self, path: List[Tuple[int, int]]) -> List[TurningPoint]:
     
        if len(path) < 3:
            return []

        turning_points = []
        cumulative_distance = 0.0

        for i in range(1, len(path) - 1):
            prev_pos = path[i - 1]
            curr_pos = path[i]
            next_pos = path[i + 1]

            
            v1 = (prev_pos[0] - curr_pos[0], prev_pos[1] - curr_pos[1])
            v2 = (next_pos[0] - curr_pos[0], next_pos[1] - curr_pos[1])

            
            if v1 != v2:
                
                dist_to_point = sum(self._distance(path[j], path[j + 1]) 
                                   for j in range(i))

                
                direction = self._get_turn_direction(v1, v2)

                tp = TurningPoint(
                    position=curr_pos,
                    step_index=i,
                    direction=direction,
                    distance=dist_to_point
                )
                turning_points.append(tp)

        return turning_points

    def _get_turn_direction(self, v1: Tuple[int, int], v2: Tuple[int, int]) -> str:

        
        cross = v1[0] * v2[1] - v1[1] * v2[0]

        if cross > 0:
            return "left"
        elif cross < 0:
            return "right"
        else:
            return "straight"

    def generate_navigation_instructions(self, path: List[Tuple[int, int]]) -> List[NavigationInstruction]:

        turning_points = self.identify_turning_points(path)

        if not turning_points:
            
            total_dist = sum(self._distance(path[i], path[i + 1]) 
                           for i in range(len(path) - 1))
            return [NavigationInstruction(
                f"Go straight {total_dist:.2f}m to exit",
                [],
                total_dist
            )]

        instructions = []
        prev_tp_idx = 0

        for i, tp in enumerate(turning_points):
            
            if i == 0:
                start_idx = 0
            else:
                start_idx = turning_points[i - 1].step_index

            segment_dist = sum(self._distance(path[j], path[j + 1])
                             for j in range(start_idx, tp.step_index))

         
            if i == 0:
                instr = f"Go straight {segment_dist:.2f}m"
            else:
                prev_direction = turning_points[i - 1].direction
                instr = f"Go straight {segment_dist:.2f}m"

            instructions.append(NavigationInstruction(instr, [tp], segment_dist))

            
            if i < len(turning_points) - 1:
                turn_instr = f"Turn {tp.direction.upper()}"
                instructions.append(NavigationInstruction(turn_instr, [tp], 0.0))

        
        last_tp = turning_points[-1]
        final_dist = sum(self._distance(path[j], path[j + 1])
                        for j in range(last_tp.step_index, len(path) - 1))
        
        if final_dist > 0:
            instructions.append(NavigationInstruction(
                f"Go straight {final_dist:.2f}m to exit",
                [],
                final_dist
            ))

        return instructions

    def get_path_summary(self, path: List[Tuple[int, int]]) -> Dict:
  
        if not path:
            return {"error": "No path provided"}

        turning_points = self.identify_turning_points(path)
        instructions = self.generate_navigation_instructions(path)
        total_distance = sum(self._distance(path[i], path[i + 1]) 
                           for i in range(len(path) - 1))

        return {
            "total_distance": round(total_distance, 4),
            "total_steps": len(path),
            "turning_points_count": len(turning_points),
            "turning_points": [
                {
                    "position": tp.position,
                    "step": tp.step_index,
                    "direction": tp.direction,
                    "distance_from_start": round(tp.distance, 4)
                }
                for tp in turning_points
            ],
            "navigation_instructions": [
                {
                    "instruction": inst.instruction,
                    "distance": round(inst.segment_distance, 4)
                }
                for inst in instructions
            ]
        }

    
