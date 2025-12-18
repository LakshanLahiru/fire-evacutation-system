# main.py
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from api.endpoints import router as api_router
from api.human_detection import router as human_detection_router
from api.signboard_endpoints import signboard_router
from api.reid import router as reid_router
from api.stair_case import router as stair_case_router

app = FastAPI(title="Fire Evacuation Route API - Multi-Video Person Re-ID")

# Mount static files FIRST before routers
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include routers
app.include_router(api_router)
app.include_router(human_detection_router)
app.include_router(signboard_router)
app.include_router(reid_router)
app.include_router(stair_case_router)

# Root endpoint - redirect to dashboard
@app.get("/")
async def root():
    """Redirect to the Re-ID dashboard"""
    return RedirectResponse(url="/static/index.html")