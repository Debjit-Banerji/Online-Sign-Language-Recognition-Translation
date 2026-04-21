"""Quick test client for the inference server WebSocket."""
import asyncio
import json
import io
import numpy as np
from PIL import Image

try:
    import websockets
except ImportError:
    print("Install: pip install websockets")
    exit(1)


def make_fake_jpeg(width=320, height=240) -> bytes:
    """Create a random JPEG image for testing."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=50)
    return buf.getvalue()


async def test():
    uri = "ws://localhost:8765/ws/translate"
    print(f"Connecting to {uri}...")

    async with websockets.connect(uri) as ws:
        # Read connection message
        msg = json.loads(await ws.recv())
        print(f"✅ Connected: {msg}")

        # Send 20 fake frames
        for i in range(20):
            jpeg = make_fake_jpeg()
            await ws.send(jpeg)
            print(f"  Sent frame {i+1} ({len(jpeg)} bytes)")

            # Check for responses (non-blocking)
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=0.3)
                data = json.loads(resp)
                if data["type"] == "result":
                    print(f"  🎯 RESULT: glosses='{data['glosses']}' | text='{data['text']}' | {data['latency_ms']}ms")
                elif data["type"] == "buffering":
                    print(f"  ⏳ {data['message']}")
            except asyncio.TimeoutError:
                pass

            await asyncio.sleep(0.05)

        # Drain remaining messages
        for _ in range(5):
            try:
                resp = await asyncio.wait_for(ws.recv(), timeout=0.5)
                data = json.loads(resp)
                if data["type"] == "result":
                    print(f"  🎯 RESULT: glosses='{data['glosses']}' | text='{data['text']}' | {data['latency_ms']}ms")
            except asyncio.TimeoutError:
                break

    print("\n✅ Test passed!")


if __name__ == "__main__":
    asyncio.run(test())
