# main.py
from fastapi import FastAPI
from api.endpoints import router as api_router
from api.human_detection import router as human_detection_router

app = FastAPI(title="Fire Evacuation Route API")

app.include_router(api_router)
app.include_router(human_detection_router)
