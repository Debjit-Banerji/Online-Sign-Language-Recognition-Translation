"""
Sign Language Translation — Inference Server

FastAPI backend with WebSocket endpoint for real-time sign language
video frame processing and translation.

Usage:
    python inference_server.py                    # mock mode (no GPU)
    python inference_server.py --mode slrt        # SLRT model
    python inference_server.py --mode custom      # user's own model
    python inference_server.py --host 0.0.0.0     # expose to network

The server:
1. Accepts WebSocket connections at /ws/translate
2. Receives binary JPEG frames from the mobile app
3. Buffers frames in a sliding window
4. Runs inference when enough frames accumulate
5. Sends back JSON: {"glosses": "...", "text": "...", "latency_ms": ...}
"""

import argparse
import asyncio
import io
import json
import time
from typing import Dict

import numpy as np
import cv2
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from gtts import gTTS

from frame_buffer import FrameBuffer
from model_loader import load_model
from preprocessing import decode_jpeg, preprocess_numpy_frames


# ────────────────────────────────────────────────────────────────
# App setup
# ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sign Language Translation API",
    description="Real-time sign language recognition and translation via WebSocket",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance (loaded at startup)
model = None
# Server config (set via CLI args)
server_config = {
    "window_size": 16,
    "stride": 8,
    "img_size": 224,
}


# ────────────────────────────────────────────────────────────────
# REST endpoints
# ────────────────────────────────────────────────────────────────

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return JSONResponse({
        "status": "ok",
        "model_loaded": model is not None,
        "config": server_config,
    })


@app.get("/")
async def root():
    """Root endpoint with API info."""
    return JSONResponse({
        "name": "Sign Language Translation API",
        "version": "0.1.0",
        "endpoints": {
            "health": "GET /health",
            "translate": "WebSocket /ws/translate",
        },
        "usage": "Connect via WebSocket and send JPEG frames as binary messages.",
    })


@app.get("/tts")
async def tts_get(text: str = Query(..., description="Text to convert to speech"),
                  lang: str = Query("en", description="Language code (en, hi, ta, etc.)")):
    """Generate MP3 audio from text via GET (used by mobile download)."""
    return _generate_tts(text, lang)


@app.post("/tts")
async def tts_post(body: dict):
    """Generate MP3 audio from text via POST."""
    text = body.get("text", "")
    lang = body.get("lang", "en")
    if not text.strip():
        return JSONResponse({"error": "No text provided"}, status_code=400)
    return _generate_tts(text, lang)


def _generate_tts(text: str, lang: str = "en"):
    """Common TTS generation logic."""
    try:
        tts = gTTS(text=text, lang=lang)
        mp3_buffer = io.BytesIO()
        tts.write_to_fp(mp3_buffer)
        mp3_buffer.seek(0)
        return StreamingResponse(
            mp3_buffer,
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": 'attachment; filename="translation.mp3"',
            },
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)


# ────────────────────────────────────────────────────────────────
# WebSocket endpoint
# ────────────────────────────────────────────────────────────────

@app.websocket("/ws/translate")
async def websocket_translate(websocket: WebSocket):
    """
    WebSocket endpoint for real-time translation.
    
    Protocol:
    - Client sends: binary JPEG frames
    - Client can send: JSON text messages for control
      {"action": "reset"}  — clear frame buffer
      {"action": "config", "window_size": 16, "stride": 8}  — reconfigure
    - Server sends: JSON text messages with results
      {"type": "result", "glosses": "...", "text": "...", "latency_ms": ..., "frame_count": ...}
      {"type": "status", "message": "...", "buffered": ..., "needed": ...}
    """
    await websocket.accept()
    
    # Per-connection frame buffer
    buffer = FrameBuffer(
        window_size=server_config["window_size"],
        stride=server_config["stride"],
    )
    
    client_host = websocket.client.host if websocket.client else "unknown"
    print(f"🔌 Client connected: {client_host}")

    # Per-connection inference queue (max 2 windows of backpressure)
    inference_queue = asyncio.Queue(maxsize=2)
    
    async def inference_worker():
        """Background task to run inference without blocking the receiver."""
        try:
            while True:
                window = await inference_queue.get()
                try:
                    t_start = time.time()
                    
                    # Run blocking inference in a separate thread to keep the event loop free
                    # model.predict is safe to run in a thread as Torch/MediaPipe release the GIL
                    if server_config.get("mode") == "custom":
                        result = await asyncio.to_thread(model.predict, window)
                    else:
                        preprocessed = await asyncio.to_thread(
                            preprocess_numpy_frames,
                            window,
                            img_size=server_config["img_size"],
                        )
                        result = await asyncio.to_thread(model.predict, preprocessed)
                    
                    latency_ms = (time.time() - t_start) * 1000
                    
                    print(f"\n{'─'*50}")
                    print(f"🔤 Glosses:    {result['glosses']}")
                    if 'text_de' in result:
                        print(f"🇩🇪 German:     {result['text_de']}")
                    print(f"🇬🇧 English:    {result['text']}")
                    print(f"⏱  Latency:    {latency_ms:.0f}ms | Total Frames: {buffer.frame_count}")
                    print(f"{'─'*50}")
                    
                    await websocket.send_json({
                        "type": "result",
                        "glosses": result["glosses"],
                        "text": result["text"],
                        "latency_ms": round(latency_ms, 1),
                        "frame_count": buffer.frame_count,
                        "window_size": len(window),
                    })
                except Exception as e:
                    print(f"⚠️ Inference Worker Error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Inference error: {str(e)}",
                    })
                finally:
                    inference_queue.task_done()
        except asyncio.CancelledError:
            pass

    worker_task = asyncio.create_task(inference_worker())
    
    # Send initial status
    await websocket.send_json({
        "type": "connected",
        "message": "Connected to Sign Language Translation server (Async Ready)",
        "config": {
            "window_size": buffer.window_size,
            "stride": buffer.stride,
        },
    })

    try:
        while True:
            message = await websocket.receive()
            
            # Handle text messages (control commands + base64 frames)
            if "text" in message:
                try:
                    data = json.loads(message["text"])
                    
                    # ── Base64 frame (fast path from mobile) ──
                    if data.get("type") == "frame" and "base64" in data:
                        import base64
                        img_bytes = base64.b64decode(data["base64"])
                        nparr = np.frombuffer(img_bytes, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            ready = buffer.add_frame(frame)
                            if ready:
                                window = buffer.get_window()
                                if window is not None:
                                    try:
                                        # Use put_nowait to push to queue without blocking
                                        # If queue is full (maxsize=2), this frame window is dropped
                                        inference_queue.put_nowait(window)
                                    except asyncio.QueueFull:
                                        print("⚠️ Inference queue full, dropping window (Server busy)")
                            else:
                                if buffer.buffered_count % 4 == 0:
                                    await websocket.send_json({
                                        "type": "buffering",
                                        "buffered": buffer.buffered_count,
                                        "needed": buffer.window_size,
                                        "message": f"Buffering... {buffer.buffered_count}/{buffer.window_size}",
                                    })
                        continue
                    
                    # ── Action commands ──
                    action = data.get("action", "")
                    
                    if action == "reset":
                        buffer.clear()
                        await websocket.send_json({
                            "type": "status",
                            "message": "Buffer cleared",
                            "buffered": 0,
                            "needed": buffer.window_size,
                        })
                    
                    elif action == "config":
                        buffer.window_size = data.get("window_size", buffer.window_size)
                        buffer.stride = data.get("stride", buffer.stride)
                        buffer.clear()
                        await websocket.send_json({
                            "type": "status",
                            "message": "Reconfigured",
                            "config": {
                                "window_size": buffer.window_size,
                                "stride": buffer.stride,
                            },
                        })
                    
                    elif action == "ping":
                        await websocket.send_json({
                            "type": "pong",
                            "timestamp": time.time(),
                        })
                
                except json.JSONDecodeError:
                    pass
                continue
            
            # Handle binary messages (JPEG frames)
            if "bytes" in message:
                jpeg_data = message["bytes"]
                
                try:
                    # Decode JPEG to numpy
                    frame = decode_jpeg(jpeg_data)
                    
                    # Add to buffer
                    ready = buffer.add_frame(frame)
                    
                    if ready:
                        # Get inference window
                        window = buffer.get_window()
                        if window is not None:
                            try:
                                # Push to queue for worker to handle
                                inference_queue.put_nowait(window)
                            except asyncio.QueueFull:
                                print("⚠️ Inference queue full, dropping window")
                    else:
                        # Periodically send buffer status (every 4 frames)
                        if buffer.buffered_count % 4 == 0:
                            await websocket.send_json({
                                "type": "buffering",
                                "buffered": buffer.buffered_count,
                                "needed": buffer.window_size,
                                "message": f"Buffering... {buffer.buffered_count}/{buffer.window_size}",
                            })
                
                except Exception as e:
                    await websocket.send_json({
                        "type": "error",
                        "message": f"Frame processing error: {str(e)}",
                    })

    except WebSocketDisconnect:
        print(f"🔌 Client disconnected: {client_host}")
    except Exception as e:
        print(f"❌ WebSocket error: {e}")
    finally:
        # Cleanup: Cancel the worker task when connection closes
        if 'worker_task' in locals():
            worker_task.cancel()
        try:
            await websocket.close()
        except:
            pass


# ────────────────────────────────────────────────────────────────
# Entry point
# ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sign Language Translation Server")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--mode", default="mock", choices=["mock", "slrt", "custom"],
                        help="Model mode: 'mock' for testing, 'slrt' for SLRT, 'custom' for user's model")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to SLRT YAML config (required for --mode slrt)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (required for --mode slrt)")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device for inference")
    parser.add_argument("--window-size", type=int, default=16,
                        help="Number of frames per inference window")
    parser.add_argument("--stride", type=int, default=8,
                        help="Frames between consecutive inferences")
    args = parser.parse_args()

    # Update server config
    server_config["window_size"] = args.window_size
    server_config["stride"] = args.stride
    server_config["mode"] = args.mode

    # Load model
    global model
    model = load_model(
        mode=args.mode,
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        device=args.device,
    )

    print(f"🚀 Starting server on {args.host}:{args.port}")
    print(f"   Mode: {args.mode}")
    print(f"   Window: {args.window_size} frames, stride: {args.stride}")
    print(f"   WebSocket: ws://{args.host}:{args.port}/ws/translate")
    print(f"   Health:    http://{args.host}:{args.port}/health")

    uvicorn.run(app, host=args.host, port=args.port, log_level="info")


if __name__ == "__main__":
    main()
