
from fastapi import APIRouter, Query, Body
from fastapi.responses import JSONResponse, FileResponse
from typing import List, Optional
from pydantic import BaseModel, Field
import pandas as pd
import os

from services.grid import Grid
from services.fire_model import FireModel
from services.signboard_system import SignboardGuidanceSystem, generate_signboard_plan
from services.visualize_signboard import detect_rooms, visualize_signboard_plan

signboard_router = APIRouter()




@signboard_router.get("/signboard-guidance", summary="Get Signboard Guidance (GET)")
def get_signboard_guidance(
    floor: int = Query(..., description="Floor number (0, 1, 2)", example=1),
    fire_floor: int = Query(..., description="Fire floor", example=1),
    stage: str = Query("initial", regex="^(initial|growth|spread)$", description="Fire stage", example="growth"),
    include_visualization: bool = Query(True, description="Generate visualization image"),
    fire_locations: List[str] = Query(
        ..., 
        description="🔥 Fire positions (format: 'r,c'). Click 'Add string item' to add multiple locations.",
        example=["11,8", "12,9"]
    ),
    exits: List[str] = Query(
        ..., 
        description="🚪 Exit positions (format: 'r,c'). Click 'Add string item' to add multiple exits.",
        example=["8,18", "22,18"]
    ),
    signboard_locations: List[str] = Query(
        ..., 
        description="📍 Signboard positions (format: 'r,c'). Click 'Add string item' to add multiple signboards.",
        example=["3,5", "6,10", "10,15", "15,5", "20,10", "25,15",]
    )
):
    
    return _process_signboard_guidance(
        floor=floor,
        fire_locations=fire_locations,
        fire_floor=fire_floor,
        exits=exits,
        signboard_locations=signboard_locations,
        stage=stage,
        include_visualization=include_visualization
    )





@signboard_router.get("/download-signboard/{filename}", summary="Download Signboard Visualization")
def download_signboard_image(filename: str):
    """
    Download generated signboard visualization image.
    
    The filename is returned in the `visualization_url` field of the signboard-guidance response.
    """
    filepath = os.path.join("output", filename)
    if not os.path.exists(filepath):
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    return FileResponse(filepath, media_type="image/png", filename=filename)



def _process_signboard_guidance(
    floor: int,
    fire_locations: List[str],
    fire_floor: int,
    exits: List[str],
    signboard_locations: List[str],
    stage: str,
    include_visualization: bool
) -> dict:

    fire_locs = [tuple(map(int, f.split(','))) for f in fire_locations]
    exit_locs = [tuple(map(int, e.split(','))) for e in exits]
    sign_locs = [tuple(map(int, s.split(','))) for s in signboard_locations]
    
   
    floor_matrix = {0: "matrix/matrix.csv", 1: "matrix/matrix1.csv", 2: "matrix/matrix2.csv"}
    matrix_path = floor_matrix.get(floor, "matrix/matrix.csv")
    
    try:
       
        df = pd.read_csv(matrix_path, index_col=0)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
        mat = df.to_numpy()
        
        grid = Grid(mat.tolist())
        fire = FireModel(grid)
        
      
        consider_fire = (floor == fire_floor)
        if consider_fire and fire_locs:
            fire.ignite(fire_locs)
            fire.stage_update(stage)
    
        rooms = detect_rooms(grid, fire)

        plan = generate_signboard_plan(grid, fire, exit_locs, sign_locs, rooms)
        
    
        image_path = None
        if include_visualization:
            image_path = visualize_signboard_plan(
                grid, fire, exit_locs, plan, 
                floor, fire_floor, stage, consider_fire
            )
        
        return {
            "floor": floor,
            "fire_floor": fire_floor,
            "fire_stage": stage,
            "fire_active": consider_fire,
            "signboards": plan["signboards"],
            "rooms": plan["rooms"],
            "corridors": plan["corridors"],
            "summary": plan["summary"],
            "visualization_url": f"/download-signboard/{os.path.basename(image_path)}" if image_path else None
        }
        
    except Exception as e:
        return JSONResponse(content={"error": str(e), "details": str(type(e).__name__)}, status_code=400)


