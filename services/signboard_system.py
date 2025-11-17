import numpy as np
import heapq
import math
from typing import List, Tuple, Dict, Optional
from collections import defaultdict

class SignboardGuidanceSystem:
    """
    System to compute optimal signboard directions for evacuation.
    Signboards guide people from any location to the nearest safe exit.
    """
    
    def __init__(self, grid, fire_model, exits: List[Tuple[int, int]]):
        self.grid = grid
        self.fire = fire_model
        self.exits = exits
        
    def compute_signboard_directions(self, signboard_locations: List[Tuple[int, int]]) -> Dict:
        """
        For each signboard location, compute the optimal direction to display.
        Returns signboard_id -> {position, direction, next_hop, distance_to_exit}
        """
        signboard_directions = {}
        
        for idx, sign_pos in enumerate(signboard_locations):
            # Compute optimal path from this signboard to nearest exit
            path, distance = self._compute_path_from_position(sign_pos)
            
            if path and len(path) > 1:
                next_step = path[1]  # Next position after current
                direction = self._get_direction_arrow(sign_pos, next_step)
                turn = self._get_turn_direction(sign_pos, next_step)
                
                signboard_directions[f"SIGN_{idx+1}"] = {
                    "position": sign_pos,
                    "signal": direction,  # "→", "←", "↑", "↓", "↗", "↖", "↘", "↙"
                    "turn_signal": turn,  # "LEFT", "RIGHT", "STRAIGHT"
                    "next_position": next_step,
                    "distance_to_exit": round(distance, 2),
                    "path_length": len(path),
                    "is_safe": not self.fire.is_unsafe(*sign_pos, buffer=0)
                }
            else:
                # No safe path found or already at exit
                signboard_directions[f"SIGN_{idx+1}"] = {
                    "position": sign_pos,
                    "signal": "BLOCKED" if not path else "EXIT",
                    "turn_signal": "NONE",
                    "next_position": None,
                    "distance_to_exit": float('inf') if not path else 0,
                    "path_length": 0,
                    "is_safe": False
                }
        
        return signboard_directions
    
    def compute_room_guidance(self, rooms: Dict[str, List[Tuple[int, int]]]) -> Dict:
        """
        For each room, determine which signboard people should follow.
        rooms: {"ROOM_101": [(r1,c1), (r2,c2), ...], "ROOM_102": [...]}
        """
        room_guidance = {}
        
        for room_name, room_cells in rooms.items():
            # Find accessible cells in room (not walls, not fire)
            accessible_cells = [
                cell for cell in room_cells 
                if self.grid.mat[cell] != 1 and not self.fire.is_unsafe(*cell, buffer=0)
            ]
            
            if not accessible_cells:
                room_guidance[room_name] = {
                    "status": "BLOCKED",
                    "guidance": "Room is not safe - seek alternative route",
                    "nearest_signboards": []
                }
                continue
            
            # Find nearest signboard locations for this room
            # Use center of room as reference point
            center = self._get_room_center(accessible_cells)
            
            # Compute path from room center
            path, distance = self._compute_path_from_position(center)
            
            if path and len(path) > 1:
                # Determine exit direction from room
                exit_direction = self._get_direction_arrow(center, path[1])
                
                room_guidance[room_name] = {
                    "status": "SAFE",
                    "exit_direction": exit_direction,
                    "distance_to_exit": round(distance, 2),
                    "guidance": f"Exit {exit_direction} - {round(distance, 2)}m to safety",
                    "path_preview": path[:5]  # Show first 5 steps
                }
            else:
                room_guidance[room_name] = {
                    "status": "NO_PATH",
                    "guidance": "No safe path available - stay in room and await rescue",
                    "exit_direction": None
                }
        
        return room_guidance
    
    def compute_corridor_guidance(self, corridor_cells: List[Tuple[int, int]], 
                                  spacing: int = 5) -> List[Dict]:
        """
        Place virtual signboards along corridors at regular intervals.
        Returns list of signboard positions and their directions.
        """
        corridor_signboards = []
        
        # Place signboards at regular intervals
        for i in range(0, len(corridor_cells), spacing):
            pos = corridor_cells[i]
            
            if self.grid.mat[pos] == 1:  # Skip walls
                continue
                
            path, distance = self._compute_path_from_position(pos)
            
            if path and len(path) > 1:
                direction = self._get_direction_arrow(pos, path[1])
                turn = self._get_turn_direction(pos, path[1])
                
                corridor_signboards.append({
                    "position": pos,
                    "signal": direction,
                    "turn_signal": turn,
                    "distance_to_exit": round(distance, 2),
                    "is_safe": not self.fire.is_unsafe(*pos, buffer=0)
                })
        
        return corridor_signboards
    
    def _compute_path_from_position(self, start: Tuple[int, int]) -> Tuple[Optional[List], float]:
        """
        A* pathfinding from given position to nearest exit.
        Similar to ant_colony._a_star but optimized for single path computation.
        """
        if start in self.exits:
            return [start], 0.0
        
        buffer_by_stage = {"initial": 0, "growth": 1, "spread": 1}
        buf = buffer_by_stage.get(self.fire.current_stage, 0)
        
        def heuristic(pos: Tuple[int, int]) -> float:
            return min(self._distance(pos, exit_pos) for exit_pos in self.exits)
        
        open_set = [(0.0, start)]
        g_cost = {start: 0.0}
        parent = {}
        
        while open_set:
            _, current = heapq.heappop(open_set)
            
            if current in self.exits:
                # Reconstruct path
                path = [current]
                while current in parent:
                    current = parent[current]
                    path.append(current)
                path.reverse()
                return path, g_cost[path[-1]]
            
            for neighbor in self.grid.neighbors(*current):
                # Check if step is valid
                if not self._is_valid_step(current, neighbor, buf):
                    continue
                
                step_cost = self._distance(current, neighbor)
                fire_penalty = self.fire.get_fire_penalty(*neighbor)
                
                if fire_penalty == float('inf'):
                    continue
                
                tentative_g = g_cost[current] + step_cost * (1.0 + fire_penalty)
                
                if tentative_g < g_cost.get(neighbor, float('inf')):
                    g_cost[neighbor] = tentative_g
                    parent[neighbor] = current
                    f_cost = tentative_g + heuristic(neighbor)
                    heapq.heappush(open_set, (f_cost, neighbor))
        
        return None, float('inf')
    
    def _is_valid_step(self, curr: Tuple[int, int], nxt: Tuple[int, int], buf: int) -> bool:
        """Check if step from curr to nxt is valid (no walls, not unsafe)."""
        if self.grid.mat[nxt] == 1:
            return False
        if self.fire.is_unsafe(nxt[0], nxt[1], buffer=buf):
            return False
        
        # Check diagonal movement - both orthogonal cells must be free
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
    
    def _distance(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        """Euclidean distance."""
        return math.hypot(a[0] - b[0], a[1] - b[1])
    
    def _get_direction_arrow(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
        """Convert movement to arrow direction."""
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]
        
        # 8-directional arrows
        if dr == 0 and dc > 0:
            return "→"  # Right
        elif dr == 0 and dc < 0:
            return "←"  # Left
        elif dr > 0 and dc == 0:
            return "↑"  # Up
        elif dr < 0 and dc == 0:
            return "↓"  # Down
        elif dr > 0 and dc > 0:
            return "↗"  # Up-right
        elif dr > 0 and dc < 0:
            return "↖"  # Up-left
        elif dr < 0 and dc > 0:
            return "↘"  # Down-right
        elif dr < 0 and dc < 0:
            return "↙"  # Down-left
        else:
            return "•"  # Stay
    
    def _get_turn_direction(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> str:
        """Simplified turn direction (LEFT, RIGHT, STRAIGHT)."""
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]
        
        # Determine primary direction
        if abs(dc) > abs(dr):
            return "RIGHT" if dc > 0 else "LEFT"
        elif abs(dr) > abs(dc):
            return "STRAIGHT"
        else:
            # Diagonal
            return "RIGHT" if dc > 0 else "LEFT"
    
    def _get_room_center(self, cells: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Calculate center point of room."""
        if not cells:
            return (0, 0)
        avg_r = sum(r for r, c in cells) // len(cells)
        avg_c = sum(c for r, c in cells) // len(cells)
        return (avg_r, avg_c)


# Integration functions for API
def generate_signboard_plan(grid, fire_model, exits: List[Tuple[int, int]], 
                           signboard_locations: List[Tuple[int, int]],
                           rooms: Optional[Dict[str, List[Tuple[int, int]]]] = None) -> Dict:
    """
    Main function to generate complete signboard guidance plan.
    """
    system = SignboardGuidanceSystem(grid, fire_model, exits)
    
    # Compute signboard directions
    signboard_directions = system.compute_signboard_directions(signboard_locations)
    
    # Compute room guidance if rooms provided
    room_guidance = {}
    if rooms:
        room_guidance = system.compute_room_guidance(rooms)
    
    # Identify corridors (cells with value 0 that are not in rooms)
    all_room_cells = set()
    if rooms:
        for cells in rooms.values():
            all_room_cells.update(cells)
    
    corridor_cells = []
    for r in range(grid.h):
        for c in range(grid.w):
            if grid.mat[r, c] == 0 and (r, c) not in all_room_cells:
                corridor_cells.append((r, c))
    
    corridor_guidance = system.compute_corridor_guidance(corridor_cells, spacing=5)
    
    return {
        "signboards": signboard_directions,
        "rooms": room_guidance,
        "corridors": corridor_guidance,
        "summary": {
            "total_signboards": len(signboard_directions),
            "active_signboards": sum(1 for s in signboard_directions.values() if s["signal"] not in ["BLOCKED", "EXIT"]),
            "blocked_signboards": sum(1 for s in signboard_directions.values() if s["signal"] == "BLOCKED"),
            "safe_rooms": sum(1 for r in room_guidance.values() if r["status"] == "SAFE"),
            "blocked_rooms": sum(1 for r in room_guidance.values() if r["status"] == "BLOCKED"),
            "corridor_guidance_points": len(corridor_guidance)
        }
    }