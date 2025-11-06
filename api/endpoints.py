# api/endpoints.py
from fastapi import APIRouter, Query
from fastapi.responses import FileResponse, JSONResponse
from typing import List
import os
from services.visualize import generate_evacuation_image


router = APIRouter()



@router.get("/evacuation")
def get_evacuation_path(
    start_row: int,
    start_col: int,
    strating_floor: int = Query(..., description="Floor number"),
    fire_locations: List[str] = Query(..., description="Format: r,c (multiple allowed)"),
    fire_floor: int = Query(..., description="Floor number where fire starts"),
    exits: List[str] = Query(..., description="Format: r,c (multiple allowed)"),
    stage: str = Query("initial", regex="^(initial|growth|spread)$")
):
    start = (start_row, start_col)
    fire_locs = [tuple(map(int, f.split(','))) for f in fire_locations]
    exit_locs = [tuple(map(int, e.split(','))) for e in exits]
    floor_metrix = {0: "matrix/matrix.csv", 1: "matrix/matrix1.csv", 2: "matrix/matrix2.csv"}
    matrix = floor_metrix[strating_floor]

    try:
       
        consider_fire = (strating_floor == fire_floor)
        
        
        result = generate_evacuation_image(
            matrix, 
            start, 
            exit_locs, 
            fire_locs if consider_fire else [], 
            stage,
            consider_fire=consider_fire,
            floor_number=strating_floor,
            fire_floor=fire_floor
        )
        
        return {
            "path": result["path"],
            "length": result["length"],
            "turning_points_count": result["summary"]["turning_points_count"],
            "turning_points": result["turning_points"],
            "navigation_instructions": result["navigation_instructions"],
            "download_url": f"/download/{os.path.basename(result['image_path'])}",
            "fire_considered": consider_fire
        }
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=400)


@router.get("/download/{filename}")
def download_image(filename: str):
    filepath = os.path.join("output", filename)
    if not os.path.exists(filepath):
        return JSONResponse(content={"error": "File not found"}, status_code=404)
    return FileResponse(filepath, media_type="image/png", filename=filename)


