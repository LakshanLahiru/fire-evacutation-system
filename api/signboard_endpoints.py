
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


# ========== Pydantic Models for POST Requests ==========
class SignboardRequest(BaseModel):
    """Request model for signboard guidance with JSON body"""
    floor: int = Field(..., description="Floor number (0, 1, 2)", ge=0, le=2, example=1)
    fire_locations: List[str] = Field(
        ..., 
        description="List of fire positions in format 'r,c'", 
        example=["11,8", "12,9"]
    )
    fire_floor: int = Field(..., description="Floor where fire is located", ge=0, le=2, example=1)
    exits: List[str] = Field(
        ..., 
        description="List of exit positions in format 'r,c'", 
        example=["8,18", "22,18"]
    )
    signboard_locations: List[str] = Field(
        ..., 
        description="List of signboard positions in format 'r,c'", 
        example=["3,5", "6,10", "10,15", "15,5", "20,10", "25,15"]
    )
    stage: str = Field(
        "initial", 
        description="Fire stage: initial, growth, or spread",
        pattern="^(initial|growth|spread)$",
        example="growth"
    )
    include_visualization: bool = Field(True, description="Generate visualization image")

    class Config:
        schema_extra = {
            "example": {
                "floor": 1,
                "fire_locations": ["11,8", "12,9"],
                "fire_floor": 1,
                "exits": ["8,18", "22,18"],
                "signboard_locations": ["3,5", "6,10", "10,15", "15,5", "20,10", "25,15"],
                "stage": "growth",
                "include_visualization": True
            }
        }


class RoomGuidanceRequest(BaseModel):
    """Request model for room guidance with JSON body"""
    room_id: str = Field(..., description="Room identifier (e.g., ROOM_001)", example="ROOM_001")
    floor: int = Field(..., description="Floor number", ge=0, le=2, example=1)
    fire_locations: List[str] = Field(..., description="Fire positions", example=["11,8"])
    fire_floor: int = Field(..., description="Fire floor", ge=0, le=2, example=1)
    exits: List[str] = Field(..., description="Exit positions", example=["8,18", "22,18"])
    stage: str = Field("initial", description="Fire stage", pattern="^(initial|growth|spread)$", example="growth")

    class Config:
        schema_extra = {
            "example": {
                "room_id": "ROOM_001",
                "floor": 1,
                "fire_locations": ["11,8"],
                "fire_floor": 1,
                "exits": ["8,18", "22,18"],
                "stage": "growth"
            }
        }


# ========== POST Endpoints (RECOMMENDED) ==========

@signboard_router.post("/signboard-guidance", summary="Get Signboard Guidance (POST - Recommended)")
def post_signboard_guidance(request: SignboardRequest):
    """
    **POST version - Get signboard guidance for evacuation (RECOMMENDED)**
    
    This endpoint accepts a JSON body and is easier to use than the GET version.
    
    ### Request Body Example:
    ```json
    {
        "floor": 1,
        "fire_locations": ["11,8", "12,9"],
        "fire_floor": 1,
        "exits": ["8,18", "22,18"],
        "signboard_locations": ["3,5", "6,10", "10,15", "15,5", "20,10"],
        "stage": "growth",
        "include_visualization": true
    }
    ```
    
    ### Returns:
    - Signboard directions for each location
    - Room guidance for all detected rooms
    - Corridor guidance points
    - Summary statistics
    - Visualization image URL (if requested)
    """
    return _process_signboard_guidance(
        floor=request.floor,
        fire_locations=request.fire_locations,
        fire_floor=request.fire_floor,
        exits=request.exits,
        signboard_locations=request.signboard_locations,
        stage=request.stage,
        include_visualization=request.include_visualization
    )


@signboard_router.post("/room-guidance", summary="Get Room Guidance (POST - Recommended)")
def post_room_guidance(request: RoomGuidanceRequest):
    """
    **POST version - Get guidance for a specific room (RECOMMENDED)**
    
    Returns evacuation guidance for occupants of a specific room.
    
    ### Request Body Example:
    ```json
    {
        "room_id": "ROOM_001",
        "floor": 1,
        "fire_locations": ["11,8"],
        "fire_floor": 1,
        "exits": ["8,18", "22,18"],
        "stage": "growth"
    }
    ```
    """
    return _process_room_guidance(
        room_id=request.room_id,
        floor=request.floor,
        fire_locations=request.fire_locations,
        fire_floor=request.fire_floor,
        exits=request.exits,
        stage=request.stage
    )


# ========== GET Endpoints (Alternative) ==========

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
        example=["3,5", "6,10", "10,15", "15,5", "20,10", "25,15"]
    )
):
    """
    **GET version - Get signboard guidance for evacuation**
    
    ### How to use in Swagger UI:
    1. Click **"Try it out"** button
    2. Fill in floor, fire_floor, stage, include_visualization
    3. For array fields (fire_locations, exits, signboard_locations):
       - Click **"Add string item"** button to add each location
       - Enter each location as **"row,column"** (e.g., "11,8")
       - Add multiple locations by clicking the button multiple times
    
    ### Example URL format:
    ```
    /signboard-guidance?floor=1&fire_floor=1&stage=growth
      &fire_locations=11,8&fire_locations=12,9
      &exits=8,18&exits=22,18
      &signboard_locations=3,5&signboard_locations=10,15&signboard_locations=20,10
    ```
    
    ### 💡 Tip: 
    For easier testing, use **POST /signboard-guidance** instead (see above endpoint)
    """
    return _process_signboard_guidance(
        floor=floor,
        fire_locations=fire_locations,
        fire_floor=fire_floor,
        exits=exits,
        signboard_locations=signboard_locations,
        stage=stage,
        include_visualization=include_visualization
    )


@signboard_router.get("/room-guidance", summary="Get Room Guidance (GET)")
def get_room_guidance(
    room_id: str = Query(..., description="Room identifier (e.g., ROOM_001)", example="ROOM_001"),
    floor: int = Query(..., description="Floor number", example=1),
    fire_floor: int = Query(..., description="Fire floor", example=1),
    stage: str = Query("initial", description="Fire stage", example="growth"),
    fire_locations: List[str] = Query(
        ..., 
        description="🔥 Fire positions. Click 'Add string item' to add multiple.",
        example=["11,8"]
    ),
    exits: List[str] = Query(
        ..., 
        description="🚪 Exit positions. Click 'Add string item' to add multiple.",
        example=["8,18", "22,18"]
    )
):
    """
    **GET version - Get guidance for a specific room**
    
    ### How to use in Swagger UI:
    1. Click **"Try it out"**
    2. Enter room_id, floor, fire_floor, stage
    3. Click **"Add string item"** for fire_locations and exits arrays
    
    ### 💡 Tip: 
    Use **POST /room-guidance** for easier testing (see above endpoint)
    """
    return _process_room_guidance(
        room_id=room_id,
        floor=floor,
        fire_locations=fire_locations,
        fire_floor=fire_floor,
        exits=exits,
        stage=stage
    )


# ========== Download Endpoint ==========

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


# ========== Core Processing Functions ==========

def _process_signboard_guidance(
    floor: int,
    fire_locations: List[str],
    fire_floor: int,
    exits: List[str],
    signboard_locations: List[str],
    stage: str,
    include_visualization: bool
) -> dict:
    """Core function to process signboard guidance request"""
    
    # Parse inputs
    fire_locs = [tuple(map(int, f.split(','))) for f in fire_locations]
    exit_locs = [tuple(map(int, e.split(','))) for e in exits]
    sign_locs = [tuple(map(int, s.split(','))) for s in signboard_locations]
    
    # Load floor matrix
    floor_matrix = {0: "matrix/matrix.csv", 1: "matrix/matrix1.csv", 2: "matrix/matrix2.csv"}
    matrix_path = floor_matrix.get(floor, "matrix/matrix.csv")
    
    try:
        # Load grid
        df = pd.read_csv(matrix_path, index_col=0)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
        mat = df.to_numpy()
        
        grid = Grid(mat.tolist())
        fire = FireModel(grid)
        
        # Initialize fire if on same floor
        consider_fire = (floor == fire_floor)
        if consider_fire and fire_locs:
            fire.ignite(fire_locs)
            fire.stage_update(stage)
        
        # Auto-detect rooms
        rooms = detect_rooms(grid, fire)
        
        # Generate signboard plan
        plan = generate_signboard_plan(grid, fire, exit_locs, sign_locs, rooms)
        
        # Generate visualization if requested
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


def _process_room_guidance(
    room_id: str,
    floor: int,
    fire_locations: List[str],
    fire_floor: int,
    exits: List[str],
    stage: str
) -> dict:
    """Core function to process room guidance request"""
    
    fire_locs = [tuple(map(int, f.split(','))) for f in fire_locations]
    exit_locs = [tuple(map(int, e.split(','))) for e in exits]
    
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
        
        # Detect all rooms
        rooms = detect_rooms(grid, fire)
        
        # Create system and get guidance
        system = SignboardGuidanceSystem(grid, fire, exit_locs)
        room_guidance = system.compute_room_guidance(rooms)
        
        if room_id in room_guidance:
            return {
                "room_id": room_id,
                "floor": floor,
                "fire_active": consider_fire,
                "guidance": room_guidance[room_id]
            }
        else:
            return JSONResponse(
                content={
                    "error": f"Room {room_id} not found",
                    "available_rooms": list(rooms.keys()),
                    "total_rooms": len(rooms)
                },
                status_code=404
            )
            
    except Exception as e:
        return JSONResponse(content={"error": str(e), "details": str(type(e).__name__)}, status_code=400)