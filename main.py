import os

import cv2 as cv
import numpy as np
import screeninfo
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

# Load model
model = YOLO("yolo11n.pt")
device = "cpu"
object_colors = list(np.random.rand(80, 3) * 255)

# Setup window and video capture stream
screen_id = 0  # You can change this if you want to use a different screen
screen = screeninfo.get_monitors()[screen_id]
screen_resolution = (screen.width, screen.height)
cv.namedWindow("Object tracker", cv.WINDOW_NORMAL)
cv.setWindowProperty("Object tracker", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)
cap = cv.VideoCapture(os.getenv("VIDEO_SOURCE"))
cap.set(cv.CAP_PROP_FRAME_WIDTH, screen_resolution[0])
cap.set(cv.CAP_PROP_FRAME_HEIGHT, screen_resolution[1])

while True:
    ret, frame = cap.read()
    frame = cv.resize(frame, screen_resolution)
    result = model.predict(frame, device=device)[0]
    for detection in result.summary():
        bbox = detection["box"]
        p1 = (int(bbox["x1"]), int(bbox["y1"]))
        p2 = (int(bbox["x2"]), int(bbox["y2"]))
        p_text = (int(bbox["x1"]), int(bbox["y1"]) - 5)

        frame = cv.rectangle(
            frame,
            pt1=p1,
            pt2=p2,
            color=object_colors[detection["class"]],
            thickness=2,
        )
        frame = cv.putText(
            frame,
            text=detection["name"],
            org=p_text,
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=1.5,
            thickness=2,
            color=object_colors[detection["class"]],
        )

    cv.imshow("Object tracker", frame)
    if cv.waitKey(1) == ord("q"):
        break

cv.destroyAllWindows()
