import asyncio
import base64
import json
import os

import cv2 as cv
import numpy as np
import websockets
from dotenv import load_dotenv
from ultralytics import YOLO

import logging
logging.basicConfig(level=logging.INFO)

load_dotenv()

# Load model
model = YOLO("yolo11n.pt")
device = "cpu"  # Or "cpu"


async def authenticate(websocket) -> bool:
    """Validate the bearer token from websocket headers"""
    try:
        headers = websocket.request.headers
        auth_header = headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return False
        token = auth_header.split(" ")[1]
        expected_token = os.getenv("SECRET_KEY")
        return token == expected_token
    except Exception:
        return False


async def process_frame(frame_data):
    try:
        decoded_frame = base64.b64decode(frame_data)
        frame_np = np.frombuffer(decoded_frame, dtype=np.uint8)
        frame = cv.imdecode(frame_np, cv.IMREAD_COLOR)

        result = model.predict(frame, device=device)[0]
        detections = []
        for detection in result.boxes:  # Access boxes directly
            box = detection.xyxy[0].tolist()  # Convert to list
            confidence = detection.conf[0].item()  # Convert to float
            class_id = int(detection.cls[0].item())  # Convert to int
            class_name = result.names[class_id]

            detections.append(
                {
                    "box": box,
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_name": class_name,
                }
            )
        return detections
    except Exception as e:
        print(f"Error processing frame: {e}")
        return []


async def handler(websocket):
    if not await authenticate(websocket):
        await websocket.close(1008, "Unauthorized")  # Close with unauthorized status code
        return

    while True:
        try:
            frame_data = await websocket.recv()
            while True:
                try:
                    # Use recv with a very small timeout to drain the queue
                    frame_data = await asyncio.wait_for(websocket.recv(), timeout=0.001)
                except asyncio.TimeoutError:
                    break  # No more messages in queue
            detections = await process_frame(frame_data)
            await websocket.send(json.dumps(detections))  # Send detections as JSON
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"Connection closed: {e}")
            break
        except Exception as e:
            print(f"Error receiving or processing frame: {e}")
            break


async def main():
    async with websockets.serve(handler, "0.0.0.0", int(os.environ.get("PORT", 8080))):
        await asyncio.Future()  # Run forever


if __name__ == "__main__":
    import os

    asyncio.run(main())
