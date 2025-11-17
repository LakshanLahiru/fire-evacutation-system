import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow, Circle
from matplotlib.lines import Line2D
from services.grid import Grid
from services.fire_model import FireModel
from typing import List, Optional
import pandas as pd
import numpy as np
import os

def detect_rooms(grid: Grid, fire: FireModel) -> dict:
    """
    Simple room detection - finds enclosed spaces.
    Returns dict of room_id -> list of cell positions.
    """
    visited = np.zeros((grid.h, grid.w), dtype=bool)
    rooms = {}
    room_counter = 1
    
    def flood_fill(start_r, start_c):
        """Flood fill to find connected free cells."""
        stack = [(start_r, start_c)]
        cells = []
        
        while stack:
            r, c = stack.pop()
            
            if visited[r, c] or grid.mat[r, c] == 1:
                continue
            
            visited[r, c] = True
            cells.append((r, c))
            
            # Check 4-connected neighbors for flood fill
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < grid.h and 0 <= nc < grid.w:
                    if not visited[nr, nc] and grid.mat[nr, nc] == 0:
                        stack.append((nr, nc))
        
        return cells
    
    # Find all connected regions
    for r in range(grid.h):
        for c in range(grid.w):
            if not visited[r, c] and grid.mat[r, c] == 0:
                room_cells = flood_fill(r, c)
                
                # Only consider as room if it has sufficient size
                if len(room_cells) >= 10:  # Minimum room size
                    rooms[f"ROOM_{room_counter:03d}"] = room_cells
                    room_counter += 1
    
    return rooms


def visualize_signboard_plan(grid: Grid, fire: FireModel, exits: List,
                             plan: dict, floor: int, fire_floor: int,
                             stage: str, consider_fire: bool) -> str:
    """
    Create visualization of signboard guidance system.
    """
    
    os.makedirs("output", exist_ok=True)
    filename = f"output/signboard_floor{floor}_{stage}_{'fire' if consider_fire else 'nofire'}.png"
    
    fig, ax = plt.subplots(figsize=(22, 14))
    
    # Create base display grid
    display_grid = np.ones((grid.h, grid.w, 3))
    for r in range(grid.h):
        for c in range(grid.w):
            if grid.mat[r, c] == 1:  # Wall
                display_grid[r, c] = [0.2, 0.2, 0.2]
            elif consider_fire and fire.intensity[r, c] > 0:  # Fire
                intensity = min(fire.intensity[r, c], 1.0)
                display_grid[r, c] = [1, 1 - intensity * 0.7, 1 - intensity * 0.7]
            else:  # Free space
                display_grid[r, c] = [0.95, 0.95, 0.95]
    
    ax.imshow(display_grid, origin='lower', extent=(0, grid.w, 0, grid.h))
    
    # Draw grid lines
    for i in range(grid.h + 1):
        ax.axhline(i, color='gray', linewidth=0.3, alpha=0.3)
    for i in range(grid.w + 1):
        ax.axvline(i, color='gray', linewidth=0.3, alpha=0.3)
    
    # Draw exits
    for i, (r, c) in enumerate(exits):
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor='green', 
                               edgecolor='darkgreen', linewidth=2, alpha=0.7))
        ax.text(c + 0.5, r + 0.5, f'EXIT\n{i+1}', ha='center', va='center',
                fontsize=10, fontweight='bold', color='white')
    
    # Draw signboards with arrows
    for sign_id, sign_data in plan["signboards"].items():
        r, c = sign_data["position"]
        signal = sign_data["signal"]
        is_safe = sign_data["is_safe"]
        
        # Signboard background
        color = 'lightgreen' if is_safe else 'orange'
        if signal == "BLOCKED":
            color = 'red'
        
        ax.add_patch(Circle((c + 0.5, r + 0.5), 0.35, 
                           facecolor=color, edgecolor='black', linewidth=2, alpha=0.8))
        
        # Draw arrow/signal
        if signal not in ["BLOCKED", "EXIT"]:
            ax.text(c + 0.5, r + 0.5, signal, ha='center', va='center',
                   fontsize=16, fontweight='bold', color='black')
        elif signal == "BLOCKED":
            ax.text(c + 0.5, r + 0.5, '✖', ha='center', va='center',
                   fontsize=16, fontweight='bold', color='white')
        else:
            ax.text(c + 0.5, r + 0.5, '✓', ha='center', va='center',
                   fontsize=16, fontweight='bold', color='white')
        
        # Add signboard ID label
        ax.text(c + 0.5, r - 0.3, sign_id.replace('SIGN_', 'S'),
               ha='center', va='top', fontsize=7, color='blue', fontweight='bold')
    
    # Draw corridor guidance (smaller markers)
    for corridor_sign in plan["corridors"][:50]:  # Limit to avoid clutter
        r, c = corridor_sign["position"]
        signal = corridor_sign["signal"]
        is_safe = corridor_sign["is_safe"]
        
        if not is_safe:
            continue
        
        # Small arrow marker for corridors
        ax.text(c + 0.5, r + 0.5, signal, ha='center', va='center',
               fontsize=10, color='blue', alpha=0.6, fontweight='bold')
    
    # Title and summary
    summary = plan["summary"]
    fire_status = f'{stage.upper()} Stage' if consider_fire else 'No Fire (Different Floor)'
    
    ax.set_title(
        f'Signboard Guidance System - Floor {floor} - {fire_status}\n'
        f'Fire Floor: {fire_floor} | Active Signs: {summary["active_signboards"]} | '
        f'Blocked: {summary["blocked_signboards"]} | Safe Rooms: {summary.get("safe_rooms", 0)}',
        fontsize=14, fontweight='bold', pad=20
    )
    
    # Legend
    legend_items = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen', 
               markeredgecolor='black', markersize=12, label='Active Signboard'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='orange',
               markeredgecolor='black', markersize=12, label='Warning Signboard'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red',
               markeredgecolor='black', markersize=12, label='Blocked Path'),
        Rectangle((0, 0), 1, 1, facecolor='green', label='Exit'),
        Rectangle((0, 0), 1, 1, facecolor='gray', label='Wall')
    ]
    
    if consider_fire:
        legend_items.insert(1, Rectangle((0, 0), 1, 1, facecolor='red', label='Fire'))
    
    ax.legend(handles=legend_items, loc='upper right', fontsize=10, framealpha=0.9)
    
    ax.set_xlim(0, grid.w)
    ax.set_ylim(0, grid.h)
    ax.set_xlabel('Column', fontsize=12, fontweight='bold')
    ax.set_ylabel('Row', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    return filename