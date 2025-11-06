import numpy as np
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Patch, Circle
from matplotlib.lines import Line2D
from services.grid import Grid
from services.fire_model import FireModel
from services.ant_colony import AntColony
import pandas as pd
import os

def generate_evacuation_image(matrix_path: str, start, exits, fire_locations, stage: str, consider_fire: bool = True, floor_number: int = 0, fire_floor: int = 0) -> dict:

    
    df = pd.read_csv(matrix_path, index_col=0)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    mat = df.to_numpy()

    grid = Grid(mat.tolist())
    fire = FireModel(grid)
    
   
    if consider_fire and fire_locations:
        fire.ignite(fire_locations)
        fire.stage_update(stage)

    aco = AntColony(grid, fire, start, exits, m_ants=30, alpha=1.0, beta=5.0, rho=0.3, Q=15.0, max_iter=50)
    path, length = aco.run()

    if not path:
        raise ValueError("No evacuation path found")

    os.makedirs("output", exist_ok=True)
    filename = f"output/route_{stage}_{'with_fire' if consider_fire else 'no_fire'}.png"

    fig, ax = plt.subplots(figsize=(20, 12))

   
    display_grid = np.ones((grid.h, grid.w, 3))
    for r in range(grid.h):
        for c in range(grid.w):
            if grid.mat[r, c] == 1: 
                display_grid[r, c] = [0, 0, 0]
            elif consider_fire and fire.intensity[r, c] > 0:  
                intensity = min(fire.intensity[r, c], 1.0)
                display_grid[r, c] = [1, 1 - intensity, 1 - intensity]
            else:  
                display_grid[r, c] = [1, 1, 1]

    ax.imshow(display_grid, origin='lower', extent=(0, grid.w, 0, grid.h))

   
    for i in range(grid.h + 1):
        ax.axhline(i, color='gray', linewidth=0.5, alpha=0.3)
    for i in range(grid.w + 1):
        ax.axvline(i, color='gray', linewidth=0.5, alpha=0.3)

   
    r, c = start
    ax.add_patch(Rectangle((c, r), 1, 1, facecolor='blue', edgecolor='blue', linewidth=3, alpha=0.7))
    ax.text(c + 0.5, r + 0.5, 'S', ha='center', va='center', fontsize=16, fontweight='bold', color='white')


    for i, (r, c) in enumerate(exits):
        ax.add_patch(Rectangle((c, r), 1, 1, facecolor='green', edgecolor='green', linewidth=3, alpha=0.7))
        ax.text(c + 0.5, r + 0.5, f'E{i+1}', ha='center', va='center', fontsize=16, fontweight='bold', color='white')

    
    path_x = [c + 0.5 for (r, c) in path]
    path_y = [r + 0.5 for (r, c) in path]
    ax.plot(path_x, path_y, 'g-', linewidth=3, alpha=0.8, label=f'Route ({len(path)} steps)')

   
    turning_points = aco.identify_turning_points(path)
    left_count = 0
    right_count = 0
    
    for tp in turning_points:
        r, c = tp.position
        
        if tp.direction == "left":
           
            circle = Circle((c + 0.5, r + 0.5), 0.35, facecolor='cyan', edgecolor='blue', linewidth=2, alpha=0.8)
            ax.add_patch(circle)
            left_count += 1
            ax.text(c + 0.5, r + 0.5, 'L', ha='center', va='center', fontsize=12, fontweight='bold', color='darkblue')
            
        elif tp.direction == "right":
           
            circle = Circle((c + 0.5, r + 0.5), 0.35, facecolor='yellow', edgecolor='red', linewidth=2, alpha=0.8)
            ax.add_patch(circle)
            right_count += 1
            ax.text(c + 0.5, r + 0.5, 'R', ha='center', va='center', fontsize=12, fontweight='bold', color='darkred')

    
    for i in range(0, len(path) - 1, 3):
        r1, c1 = path[i]
        r2, c2 = path[i + 1]
        dx, dy = c2 - c1, r2 - r1
        ax.arrow(
            c1 + 0.5, r1 + 0.5,
            dx * 0.3, dy * 0.3,
            head_width=0.3, head_length=0.2,
            fc='darkgreen', ec='darkgreen', alpha=0.6
        )

    ax.set_xlim(0, grid.w)
    ax.set_ylim(0, grid.h)
    ax.set_xlabel('Column', fontsize=12, fontweight='bold')
    ax.set_ylabel('Row', fontsize=12, fontweight='bold')
    
   
    fire_status = f'{stage.upper()} Stage' if consider_fire else 'No Fire (Different Floor)'
    ax.set_title(
        f'Optimal Evacuation Route - {fire_status}\n'
        f'Existing Floor: {floor_number} |   Fire Floor: {fire_floor}\n'
        f'Length: {length:.4f}m | Steps: {len(path)} | Left Turns: {left_count} | Right Turns: {right_count}',
        fontsize=14, fontweight='bold', pad=20
    )

   
    legend_items = [
        Patch(facecolor='black', label='Walls'),
        Patch(facecolor='blue', label='Start'),
        Patch(facecolor='green', label='Exit'),
        Line2D([0], [0], color='green', linewidth=3, label='Route'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='cyan', markeredgecolor='blue', markersize=12, label='Left Turn'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='yellow', markeredgecolor='red', markersize=12, label='Right Turn')
    ]
    
    
    if consider_fire:
        legend_items.insert(1, Patch(facecolor='red', label='Fire'))
    
    ax.legend(handles=legend_items, loc='upper left', fontsize=10)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close(fig)

   
    summary = aco.get_path_summary(path)

    return {
        "path": path,
        "length": length,
        "image_path": filename,
        "turning_points": summary["turning_points"],
        "navigation_instructions": summary["navigation_instructions"],
        "summary": summary
    }