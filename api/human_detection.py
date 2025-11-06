from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket
from fastapi.responses import JSONResponse
import threading
import os
import asyncio
from services.thermal_detection import ThermalHumanDetector

router = APIRouter()
detector = ThermalHumanDetector()
threads = {"video": None, "webcam": None}

@router.post("/detect/video")
async def detect_video(file: UploadFile = File(...)):
    try:
        os.makedirs("vedios", exist_ok=True)
        video_path = f"vedios/{file.filename}"
        with open(video_path, "wb") as f:
            f.write(await file.read())

        detector.stop_flag["video"] = False
        threads["video"] = threading.Thread(target=detector.detect_in_video, args=(video_path,))
        threads["video"].start()

        return {"status": "Thermal detection started on uploaded video"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Video detection failed: {str(e)}"})

@router.post("/detect/webcam")
async def detect_webcam():
    try:
        detector.stop_flag["webcam"] = False
        threads["webcam"] = threading.Thread(target=detector.detect_in_webcam)
        threads["webcam"].start()

        return {"status": "Thermal detection started on webcam"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Webcam detection failed: {str(e)}"})

@router.post("/stop")
async def stop_all():
    try:
        detector.stop_all()
        return {"status": "All detection processes stopped"}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to stop detection: {str(e)}"})
    
@router.websocket("/ws")
async def detection_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            status = detector.get_status()
            await websocket.send_json(status)
            await asyncio.sleep(1)  # Send update every second
    except Exception:
        await websocket.close()