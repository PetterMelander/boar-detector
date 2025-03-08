import asyncio
import base64
import json
import logging
import os

import cv2 as cv
import numpy as np
import screeninfo
import websockets
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logging.getLogger("websockets").setLevel(logging.INFO)

load_dotenv()

# Setup window and video capture stream
screen_id = 0  # You can change this if you want to use a different screen
screen = screeninfo.get_monitors()[screen_id]
screen_resolution = (screen.width, screen.height)
cv.namedWindow("Object tracker", cv.WINDOW_NORMAL)
# cv.setWindowProperty("Object tracker", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, screen_resolution[0])
cap.set(cv.CAP_PROP_FRAME_HEIGHT, screen_resolution[1])

object_colors = list(np.random.rand(80, 3) * 255)
inference_size = (640, 480)
scale_x = screen_resolution[0] / inference_size[0]  # scaling factor for bounding boxes
scale_y = screen_resolution[1] / inference_size[1]  # scaling factor for bounding boxes
latest_detections: list[dict] = []  # Global variable to store latest results


async def receiver(websocket):
    global latest_detections
    while True:
        try:
            detections_str = await websocket.recv()
            latest_detections = json.loads(detections_str)
        except websockets.exceptions.ConnectionClosedError as e:
            print(f"Receiver connection closed: {e}")
            break
        except Exception as e:
            print(f"Error in receiver: {e}")
            break


def display_results(frame, detections: list[dict]) -> None:
    for detection in detections:
        bbox = detection["box"]
        p1 = (int(bbox[0] * scale_x), int(bbox[1] * scale_y))
        p2 = (int(bbox[2] * scale_x), int(bbox[3] * scale_y))
        p_text = (int(bbox[0] * scale_x), int(bbox[1] * scale_y) - 5)

        frame = cv.rectangle(
            frame,
            pt1=p1,
            pt2=p2,
            color=object_colors[detection["class_id"]],
            thickness=2,
        )
        frame = cv.putText(
            frame,
            text=detection["class_name"],
            org=p_text,
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=1.5,
            thickness=2,
            color=object_colors[detection["class_id"]],
        )
    cv.imshow("Object tracker", frame)
    if cv.waitKey(1) == ord("q"):
        cap.release()
        cv.destroyAllWindows()
        exit(0)


async def stream_frames():
    # uri = "ws://your-cloud-run-service-url/ws"  # Replace with your Cloud Run URL
    uri = "ws://localhost:8080/ws"
    secret_key = os.getenv("SECRET_KEY")
    if not secret_key:
        raise ValueError("SECRET_KEY environment variable is not set")
    headers = {
        "Authorization": f"Bearer {secret_key}"
    }
    async with websockets.connect(uri, additional_headers=headers) as websocket:
        # Start the receiver in the background
        recv_task = asyncio.create_task(receiver(websocket))
        while True:
            ret, frame = cap.read()
            _, buffer = cv.imencode(
                ".jpg", cv.resize(frame, inference_size), [cv.IMWRITE_JPEG_QUALITY, 80]
            )  # Encode as JPEG
            frame = cv.resize(frame, screen_resolution, interpolation=cv.INTER_NEAREST)
            frame_bytes = buffer.tobytes()
            encoded_frame = base64.b64encode(frame_bytes).decode("utf-8")  # Encode to base64 string

            try:
                await websocket.send(encoded_frame)  # Does NOT wait for confirmation from server
                display_results(frame, latest_detections)

            except websockets.exceptions.ConnectionClosedError as e:
                print(f"Connection closed: {e}")
                break
            except Exception as e:
                print(f"Error sending frame: {e}")
                break

        recv_task.cancel()


if __name__ == "__main__":
    asyncio.run(stream_frames())
