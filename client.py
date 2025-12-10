import websockets
import asyncio
import json
from datetime import datetime

async def listen():
    uri = "ws://localhost:8000/ws"
    
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Connecting to {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Connected! Listening for updates...\n")
            
            while True:
                try:
                    # Receive data from server
                    data = await websocket.recv()
                    status = json.loads(data)
                    
                    # Clear screen for better visualization (optional)
                    # print("\033[2J\033[H", end="")  # Uncomment to clear screen
                    
                    timestamp = datetime.now().strftime('%H:%M:%S')
                    print(f"\n{'='*60}")
                    print(f"[{timestamp}] Status Update")
                    print(f"{'='*60}")
                    
                    if not status:
                        print("No active videos")
                    else:
                        for video_id, info in status.items():
                            print(f"\nVideo ID: {video_id[:8]}...")
                            print(f"  Filename: {info.get('filename', 'N/A')}")
                            print(f"  Persons Detected: {info.get('count', 0)}")
                            print(f"  FPS: {info.get('fps', 0):.2f}")
                            print(f"  Running: {info.get('running', False)}")
                            print(f"  Frames Processed: {info.get('frames_processed', 0)}")
                            
                            if 'error' in info:
                                print(f"  ⚠️  Error: {info['error']}")
                    
                except websockets.exceptions.ConnectionClosed:
                    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Connection closed by server")
                    break
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON: {e}")
                except Exception as e:
                    print(f"Error: {e}")
                    
    except ConnectionRefusedError:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] ❌ Could not connect to server at {uri}")
        print("Make sure the FastAPI server is running!")
    except Exception as e:
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(listen())
    except KeyboardInterrupt:
        print("\n\nClient stopped by user")