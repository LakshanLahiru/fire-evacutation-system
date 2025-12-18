from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket, Body
from fastapi.responses import JSONResponse
import threading
import os
import uuid
import asyncio
from services.thermal_detection import ThermalHumanDetector
from typing import List
from fastapi import WebSocketDisconnect
import copy

router = APIRouter(tags=["human_detection"])
detector = ThermalHumanDetector()
threads = {"video": None, "webcam": None}

active_videos = {}  
video_status = {}
status_lock = threading.Lock()  # Add lock for thread safety

@router.post("/detect/videos")
async def detect_multiple_videos(files: List[UploadFile] = File(...)):
    try:
        os.makedirs("videos", exist_ok=True)
        video_ids = []

        for file in files:
            video_path = f"videos/{file.filename}"
            
            # Save file
            with open(video_path, "wb") as f:
                f.write(await file.read())

            # Generate unique ID
            video_id = str(uuid.uuid4())
            video_ids.append(video_id)
            
            # Initialize status with lock
            with status_lock:
                video_status[video_id] = {
                    "count": 0,
                    "fps": 0.0,
                    "running": True,
                    "video_path": video_path,
                    "filename": file.filename
                }
            
            # Initialize stop flag
            detector.stop_flag[video_id] = False
            
            # Start detection thread
            thread = threading.Thread(
                target=detector.detect_in_video_multi,
                args=(video_id, video_path, video_status),
                daemon=True  # Make thread daemon so it closes when main program exits
            )
            active_videos[video_id] = thread
            thread.start()

        return {
            "message": "Video processing started", 
            "video_ids": video_ids,
            "count": len(video_ids)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/webcam")
async def detect_webcam():
    try:
        detector.stop_flag["webcam"] = False
        threads["webcam"] = threading.Thread(
            target=detector.detect_in_webcam,
            daemon=True
        )
        threads["webcam"].start()

        return {"status": "Thermal detection started on webcam"}
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Webcam detection failed: {str(e)}"
        )


@router.post("/stop/videos")
async def stop_videos(video_ids: List[str] = Body(...)):
    stopped = []
    not_found = []

    for vid in video_ids:
        if vid in active_videos:
            detector.stop_flag[vid] = True
            stopped.append(vid)
            
            # Wait for thread to finish
            thread = active_videos[vid]
            if thread.is_alive():
                thread.join(timeout=2.0)
            
            # Clean up
            with status_lock:
                if vid in video_status:
                    video_status[vid]["running"] = False
        else:
            not_found.append(vid)

    return {
        "stopped_videos": stopped,
        "not_found": not_found
    }


@router.get("/status")
async def get_status():
    """Get current status of all videos"""
    with status_lock:
        return copy.deepcopy(video_status)


@router.websocket("/ws")
async def detection_stream(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket client connected")
    
    try:
        while True:
            # Create a deep copy to avoid race conditions
            with status_lock:
                current_status = copy.deepcopy(video_status)
            
            # Send the status
            await websocket.send_json(current_status)
            
            # Wait before next update
            await asyncio.sleep(1)  # Update twice per second
            
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        print("WebSocket connection closed")