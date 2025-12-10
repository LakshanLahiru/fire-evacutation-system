# app.py
import streamlit as st
import threading
import asyncio
import websockets
import json
import requests
import time
import queue
from datetime import datetime
import plotly.graph_objects as go
import pandas as pd


# ----------------------------------------------------
# CONFIG
# ----------------------------------------------------
API_BASE_URL = "http://localhost:8000"
WS_URL = "ws://localhost:8000/ws"

# ----------------------------------------------------
# GLOBAL QUEUE (IMPORTANT!!!)
# DO NOT PUT THIS IN session_state!!!
# ----------------------------------------------------
WS_QUEUE = queue.Queue()


# ----------------------------------------------------
# STREAMLIT SESSION INIT (safe)
# ----------------------------------------------------
if "video_status" not in st.session_state:
    st.session_state.video_status = {}

if "history" not in st.session_state:
    st.session_state.history = {}

if "ws_thread_started" not in st.session_state:
    st.session_state.ws_thread_started = False


# ----------------------------------------------------
# ASYNC WEBSOCKET CLIENT (background thread)
# ----------------------------------------------------
async def ws_client():
    while True:
        try:
            async with websockets.connect(WS_URL) as ws:
                print("Streamlit WebSocket connected.")

                while True:
                    try:
                        msg = await ws.recv()
                        data = json.loads(msg)

                        # PUT DATA IN GLOBAL QUEUE
                        WS_QUEUE.put(data)

                    except Exception as e:
                        print("WebSocket recv error:", e)
                        break

        except Exception as e:
            print("WebSocket connection lost, retrying...", e)
            time.sleep(1)


def start_ws_thread():
    if not st.session_state.ws_thread_started:
        st.session_state.ws_thread_started = True

        def thread_target():
            asyncio.run(ws_client())

        t = threading.Thread(target=thread_target, daemon=True)
        t.start()


# ----------------------------------------------------
# DRAIN QUEUE & UPDATE UI (only in main thread)
# ----------------------------------------------------
def process_ws_messages():
    updated = False

    while not WS_QUEUE.empty():
        msg = WS_QUEUE.get()

        # update session_state safely
        st.session_state.video_status = msg
        update_history(msg)
        updated = True

    return updated


# ----------------------------------------------------
# HISTORY MANAGEMENT
# ----------------------------------------------------
def update_history(status):
    now = datetime.now()
    for vid, info in status.items():
        if vid not in st.session_state.history:
            st.session_state.history[vid] = {
                "timestamps": [],
                "persons": [],
                "fps": []
            }

        hist = st.session_state.history[vid]
        hist["timestamps"].append(now)
        hist["persons"].append(info.get("count", 0))
        hist["fps"].append(info.get("fps", 0))

        # limit to last 60 points
        if len(hist["timestamps"]) > 60:
            hist["timestamps"].pop(0)
            hist["persons"].pop(0)
            hist["fps"].pop(0)


# ----------------------------------------------------
# API HELPERS
# ----------------------------------------------------
def upload_videos(files):
    try:
        files_data = [
            ("files", (f.name, f.getvalue(), f.type))
            for f in files
        ]

        res = requests.post(f"{API_BASE_URL}/detect/videos", files=files_data)
        return res.json()
    except Exception as e:
        return {"error": str(e)}


def stop_videos(vids):
    try:
        res = requests.post(f"{API_BASE_URL}/stop/videos", json=vids)
        return res.json()
    except Exception as e:
        return {"error": str(e)}


# ----------------------------------------------------
# UI
# ----------------------------------------------------
def main():

    start_ws_thread()

    st.title("🔥 Thermal Human Detection Dashboard")
    st.markdown("---")

    # PROCESS WS MESSAGES EACH RUN
    process_ws_messages()

    # ---- SIDEBAR ----
    with st.sidebar:
        st.header("Controls")

        files = st.file_uploader(
            "Upload Videos",
            type=["mp4", "avi", "mov"],
            accept_multiple_files=True
        )

        if st.button("Start Detection"):
            if files:
                res = upload_videos(files)
                st.success("Started detection")
                st.json(res)
            else:
                st.warning("Select videos first!")

        if st.button("Stop All"):
            ids = list(st.session_state.video_status.keys())
            st.json(stop_videos(ids))

    # ---- STATUS ----
    status = st.session_state.video_status

    st.header("📡 Live Status")

    total_persons = sum(v.get("count", 0) for v in status.values())
    active_videos = len([v for v in status.values() if v.get("running")])
    avg_fps = (
        sum(v.get("fps", 0) for v in status.values()) / max(len(status), 1)
        if status else 0
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Persons", total_persons)
    col2.metric("Active Videos", active_videos)
    col3.metric("Avg FPS", round(avg_fps, 2))

    st.markdown("---")

    # ---- GRAPHS ----
    if st.session_state.history:
        st.header("📈 Real-Time Charts")
        tabs = st.tabs(["Persons", "FPS"])

        with tabs[0]:
            fig = go.Figure()
            for vid, hist in st.session_state.history.items():
                fig.add_trace(go.Scatter(
                    x=hist["timestamps"],
                    y=hist["persons"],
                    name=vid[:6],
                    mode="lines+markers"
                ))
            st.plotly_chart(fig, use_container_width=True)

        with tabs[1]:
            fig = go.Figure()
            for vid, hist in st.session_state.history.items():
                fig.add_trace(go.Scatter(
                    x=hist["timestamps"],
                    y=hist["fps"],
                    name=vid[:6],
                    mode="lines+markers"
                ))
            st.plotly_chart(fig, use_container_width=True)

    # ---- DETAILS ----
    st.header("🎥 Video Details")
    for vid, info in status.items():
        with st.expander(f"{vid} - {'Running' if info.get('running') else 'Stopped'}"):
            st.write(info)
            if st.button(f"Stop {vid}", key=f"stop_{vid}"):
                st.json(stop_videos([vid]))

    time.sleep(0.1)


if __name__ == "__main__":
    main()
