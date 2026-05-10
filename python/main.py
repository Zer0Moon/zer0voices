import asyncio
import json
import websockets
import sounddevice as sd
import numpy as np

HOST = "127.0.0.1"
PORT = 8765

connected_clients = set()
audio_stream = None
current_input_device = None
current_output_device = None
is_converting = False

async def broadcast(message: dict):
    if connected_clients:
        data = json.dumps(message)
        await asyncio.gather(*[c.send(data) for c in connected_clients], return_exceptions=True)

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

    elif msg_type == "set_input_device":
        global current_input_device
        current_input_device = data.get("payload")
        await websocket.send(json.dumps({"type": "input_device_set", "payload": current_input_device}))

    elif msg_type == "set_output_device":
        global current_output_device
        current_output_device = data.get("payload")
        await websocket.send(json.dumps({"type": "output_device_set", "payload": current_output_device}))

    elif msg_type == "start_stream":
        await start_audio_stream(websocket)

    elif msg_type == "stop_stream":
        await stop_audio_stream()
        await websocket.send(json.dumps({"type": "stream_stopped"}))

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
            return {
                "available": True,
                "name": torch.cuda.get_device_name(0),
                "vram_gb": torch.cuda.get_device_properties(0).total_memory // (1024**3),
                "cuda_version": torch.version.cuda
            }
        return {"available": False}
    except Exception as e:
        return {"available": False, "error": str(e)}

async def start_audio_stream(websocket):
    global audio_stream, current_input_device
    loop = asyncio.get_event_loop()

    await stop_audio_stream()

    def audio_callback(indata, frames, time, status):
        rms = float(np.sqrt(np.mean(indata**2)))
        waveform = indata[::4, 0].tolist()
        asyncio.run_coroutine_threadsafe(
            broadcast({
                "type": "audio_data",
                "payload": {
                    "waveform": waveform,
                    "rms": rms,
                    "converting": is_converting
                }
            }),
            loop
        )

    try:
        audio_stream = sd.InputStream(
            device=current_input_device,
            channels=1,
            samplerate=44100,
            blocksize=2048,
            callback=audio_callback
        )
        audio_stream.start()
        await websocket.send(json.dumps({"type": "stream_started"}))
        print(f"[Zer0Voices] Audio stream started on device {current_input_device}")
    except Exception as e:
        await websocket.send(json.dumps({"type": "stream_error", "payload": str(e)}))
        print(f"[Zer0Voices] Stream error: {e}")

async def stop_audio_stream():
    global audio_stream
    if audio_stream:
        audio_stream.stop()
        audio_stream.close()
        audio_stream = None

async def main():
    print(f"[Zer0Voices] Sidecar starting on ws://{HOST}:{PORT}")
    async with websockets.serve(handler, HOST, PORT):
        await asyncio.Future()

if __name__ == "__main__":
    asyncio.run(main())