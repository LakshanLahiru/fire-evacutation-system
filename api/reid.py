from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket,  Body, WebSocketDisconnect, Form
from fastapi.responses import JSONResponse
import threading
import os
import time
import json
import uuid
import asyncio
from typing import List, Optional

from services.detector import YoloDetector
from services.reid import ReIDExtractor
from services.tracker import DeepSortWrapper
from services.worker import CameraWorker
from services.identity_manager import IdentityManager


router = APIRouter()
# Static files are now mounted in main.py

CAM_WORKERS = {}      # cam_name -> CameraWorker
STATUS_LOCK = threading.Lock()
SHARED_STATUS = {}    # cam_name -> status dict

# Default models (shared across all videos)
DETECTOR = YoloDetector()
REID = ReIDExtractor()

# GLOBAL IDENTITY MANAGER - This is the key component for cross-video Re-ID
IDENTITY_MANAGER = IdentityManager(similarity_threshold=0.6)

# Each video gets its own tracker instance
TRACKERS = {}  # cam_name -> DeepSortWrapper

@router.post("/start_camera")
async def start_camera(cam_name: str = Body(...), source: str = Body(...), area_m2: float = Body(50.0)):
    """
    Start processing for a camera/video.
    source: video path or device index string (e.g., '0' for webcam)
    """
    if cam_name in CAM_WORKERS:
        return {"error": "camera already running", "cam_name": cam_name}
    
    # Create a new tracker for this video
    tracker = DeepSortWrapper()
    TRACKERS[cam_name] = tracker
    
    worker = CameraWorker(
        cam_name=cam_name, 
        source=source, 
        shared_status=SHARED_STATUS,
        lock=STATUS_LOCK, 
        area_m2=area_m2,
        detector=DETECTOR, 
        reid=REID, 
        tracker=tracker,
        identity_manager=IDENTITY_MANAGER  # Pass global identity manager
    )
    CAM_WORKERS[cam_name] = worker
    worker.start()
    return {"started": cam_name, "message": "Video processing started with global Re-ID"}

@router.post("/stop_camera")
async def stop_camera(cam_name: str = Body(...)):
    if cam_name not in CAM_WORKERS:
        return {"error": "camera not found"}
    CAM_WORKERS[cam_name].stop()
    del CAM_WORKERS[cam_name]
    if cam_name in TRACKERS:
        del TRACKERS[cam_name]
    return {"stopped": cam_name}

# upload video and start as camera
@router.post("/upload_and_start")
async def upload_and_start(file: UploadFile = File(...), cam_name: str = Form(None), area_m2: float = Form(50.0)):
    """
    Upload a video file and start processing it.
    If cam_name not provided, generates one automatically.
    """
    os.makedirs("uploads", exist_ok=True)
    fname = file.filename
    
    # Generate unique camera name if not provided
    if cam_name is None or cam_name == "":
        cam_name = f"video_{int(time.time())}_{fname.split('.')[0]}"
    
    path = os.path.join("uploads", f"{int(time.time())}_{fname}")
    with open(path, "wb") as f:
        f.write(await file.read())
    
    return await start_camera(cam_name=cam_name, source=path, area_m2=area_m2)

# NEW: Upload multiple videos at once
@router.post("/upload_multiple")
async def upload_multiple(files: List[UploadFile] = File(...), area_m2: float = Form(50.0)):
    """
    Upload multiple video files and start processing all of them.
    Each video will be automatically assigned a unique name.
    """
    os.makedirs("uploads", exist_ok=True)
    results = []
    
    for file in files:
        fname = file.filename
        cam_name = f"video_{int(time.time())}_{fname.split('.')[0]}"
        path = os.path.join("uploads", f"{int(time.time())}_{fname}")
        
        with open(path, "wb") as f:
            f.write(await file.read())
        
        result = await start_camera(cam_name=cam_name, source=path, area_m2=area_m2)
        results.append({
            "filename": fname,
            "cam_name": cam_name,
            "result": result
        })
        
        # Small delay to ensure unique timestamps
        await asyncio.sleep(0.1)
    
    return {
        "uploaded": len(results),
        "videos": results,
        "message": f"Started processing {len(results)} videos with global Re-ID"
    }

# websocket endpoint: broadcast SHARED_STATUS periodically
@router.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            with STATUS_LOCK:
                snapshot = json.loads(json.dumps(SHARED_STATUS))
            await ws.send_json(snapshot)
            await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print("WS error:", e)
    finally:
        try:
            await ws.close()
        except:
            pass

# Root route moved to main.py

# quick cleanup endpoint
@router.post("/stop_all")
async def stop_all():
    cams = list(CAM_WORKERS.keys())
    for c in cams:
        try:
            CAM_WORKERS[c].stop()
        except Exception:
            pass
        del CAM_WORKERS[c]
        if c in TRACKERS:
            del TRACKERS[c]
    with STATUS_LOCK:
        for k in list(SHARED_STATUS.keys()):
            SHARED_STATUS[k]["running"] = False
    return {"stopped": cams}

# NEW: Get all global identities
@router.get("/identities")
async def get_identities():
    """
    Get all registered global identities and their metadata.
    Shows which people have been seen across which videos.
    """
    identities = IDENTITY_MANAGER.get_all_identities()
    stats = IDENTITY_MANAGER.get_statistics()
    
    return {
        "identities": identities,
        "statistics": stats
    }

# NEW: Get identities for a specific video
@router.get("/identities/{video_name}")
async def get_video_identities(video_name: str):
    """
    Get all identities seen in a specific video.
    """
    ids = IDENTITY_MANAGER.get_video_identities(video_name)
    identities_info = {}
    for gid in ids:
        info = IDENTITY_MANAGER.get_identity_info(gid)
        if info:
            identities_info[gid] = info
    
    return {
        "video_name": video_name,
        "identity_count": len(ids),
        "identities": identities_info
    }

# NEW: Reset identity database
@router.post("/reset_identities")
async def reset_identities():
    """
    Clear all stored identities. Useful for starting fresh.
    Warning: This will reset all person IDs!
    """
    IDENTITY_MANAGER.reset()
    return {
        "message": "All identities have been reset",
        "status": "success"
    }

# NEW: Get current status with identity information
@router.get("/status")
async def get_status():
    """
    Get current status of all videos with identity information.
    """
    with STATUS_LOCK:
        status_snapshot = json.loads(json.dumps(SHARED_STATUS))
    
    stats = IDENTITY_MANAGER.get_statistics()
    
    return {
        "videos": status_snapshot,
        "identity_stats": stats,
        "active_videos": len([v for v in status_snapshot.values() if v.get("running", False)])
    }
