from fastapi import APIRouter, UploadFile, File, HTTPException, Body, Form
from fastapi.responses import JSONResponse
import threading
import os
import time
import uuid
from typing import List, Optional
from staire_case.density_monitor import StaircaseDensityMonitor

router = APIRouter(prefix="/staircase", tags=["staircase"])

MONITORS = {}
STATUS_LOCK = threading.Lock()
SHARED_STATUS = {}

@router.post("/upload_video")
async def upload_video(
    file: UploadFile = File(...),
    staircase_area_m2: float = Form(10.0),
    density_threshold: float = Form(0.5)
):
    os.makedirs("staire_case/uploads", exist_ok=True)
    fname = file.filename
    monitor_id = str(uuid.uuid4())
    
    path = os.path.join("staire_case", "uploads", f"{int(time.time())}_{fname}")
    with open(path, "wb") as f:
        f.write(await file.read())
    
    monitor = StaircaseDensityMonitor(
        staircase_area_m2=staircase_area_m2,
        density_threshold=density_threshold
    )
    
    monitor_status = {}
    SHARED_STATUS[monitor_id] = monitor_status
    
    monitor.start(path, monitor_status)
    MONITORS[monitor_id] = monitor
    
    return {
        "monitor_id": monitor_id,
        "filename": fname,
        "staircase_area_m2": staircase_area_m2,
        "density_threshold": density_threshold,
        "message": "Video processing started"
    }

@router.post("/live_camera")
async def live_camera(
    camera_index: int = Body(0),
    staircase_area_m2: float = Body(10.0),
    density_threshold: float = Body(0.5)
):
    monitor_id = str(uuid.uuid4())
    
    monitor = StaircaseDensityMonitor(
        staircase_area_m2=staircase_area_m2,
        density_threshold=density_threshold
    )
    
    monitor_status = {}
    SHARED_STATUS[monitor_id] = monitor_status
    
    monitor.start(str(camera_index), monitor_status)
    MONITORS[monitor_id] = monitor
    
    return {
        "monitor_id": monitor_id,
        "camera_index": camera_index,
        "staircase_area_m2": staircase_area_m2,
        "density_threshold": density_threshold,
        "message": "Live camera processing started"
    }

@router.get("/status/{monitor_id}")
async def get_status(monitor_id: str):
    if monitor_id not in MONITORS:
        raise HTTPException(status_code=404, detail="Monitor not found")
    
    status = MONITORS[monitor_id].get_status()
    return {
        "monitor_id": monitor_id,
        **status
    }

@router.get("/status")
async def get_all_status():
    with STATUS_LOCK:
        result = {}
        for monitor_id, monitor in MONITORS.items():
            result[monitor_id] = monitor.get_status()
        return result

@router.post("/stop/{monitor_id}")
async def stop_monitor(monitor_id: str):
    if monitor_id not in MONITORS:
        raise HTTPException(status_code=404, detail="Monitor not found")
    
    MONITORS[monitor_id].stop()
    del MONITORS[monitor_id]
    
    with STATUS_LOCK:
        if monitor_id in SHARED_STATUS:
            del SHARED_STATUS[monitor_id]
    
    return {"message": "Monitor stopped", "monitor_id": monitor_id}

@router.post("/stop_all")
async def stop_all():
    monitor_ids = list(MONITORS.keys())
    for monitor_id in monitor_ids:
        try:
            MONITORS[monitor_id].stop()
            del MONITORS[monitor_id]
        except Exception:
            pass
    
    with STATUS_LOCK:
        SHARED_STATUS.clear()
    
    return {"stopped": monitor_ids}

