import websockets
import asyncio
import json

async def listen():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as websocket:
        while True:
            data = await websocket.recv()
            print(json.loads(data))

asyncio.run(listen())
