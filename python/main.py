import asyncio
import json
import websockets
import sounddevice as sd
import numpy as np

HOST = "127.0.0.1"
PORT = 8765

connected_clients = set()

async def handler(websocket):
    connected_clients.add(websocket)
    print(f"[Zer0Voices] Client connected")
    try:
        async for message in websocket:
            data = json.loads(message)
            await handle_message(websocket, data)
    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        connected_clients.discard(websocket)
        print(f"[Zer0Voices] Client disconnected")

async def handle_message(websocket, data):
    msg_type = data.get("type")

    if msg_type == "ping":
        await websocket.send(json.dumps({"type": "pong"}))

    elif msg_type == "get_devices":
        devices = get_audio_devices()
        await websocket.send(json.dumps({
            "type": "devices",
            "payload": devices
        }))

    elif msg_type == "get_gpu_info":
        info = get_gpu_info()
        await websocket.send(json.dumps({
            "type": "gpu_info",
            "payload": info
        }))

def get_audio_devices():
    devices = sd.query_devices()
    inputs = []
    outputs = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            inputs.append({"id": i, "name": d["name"]})
        if d["max_output_channels"] > 0:
            outputs.append({"id": i, "name": d["name"]})
    return {"inputs": inputs, "outputs": outputs}

def get_gpu_info():
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory // (1024**3)
            return {
                "available": True,
                "name": gpu_name,
                "vram_gb": vram,
                "cuda_version": torch.version.cuda
            }
        else:
            return {"available": False}
    except Exception as e:
        return {"available": False, "error": str(e)}

async def main():
    print(f"[Zer0Voices] Sidecar starting on ws://{HOST}:{PORT}")
    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())