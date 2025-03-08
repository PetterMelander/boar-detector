import asyncio
import os
import threading
import time
import cv2 as cv
import argparse
from dotenv import load_dotenv
from ultralytics import YOLO

load_dotenv()

THRESHOLD = float(os.getenv("DECISION_THRESHOLD"))
MIN_SCARE_INTERVAL = float(os.getenv("MIN_SCARE_INTERVAL"))
SCARE_FORGET_TIME = float(os.getenv("SCARE_FORGET_TIME"))
MAX_SCARES = int(os.getenv("MAX_SCARES"))
DETECTION_INTERVAL = int(os.getenv("DETECTION_INTERVAL"))

scare_counter = 0


def detect_boar(model, frame, device, verbose):
    result = model.predict(frame, device=device, verbose=verbose)[0]
    return result.probs.data[341:344].sum() >= THRESHOLD


async def scare_boar():
    global scare_counter
    if scare_counter >= MAX_SCARES:
        return
    print("boo!")
    scare_counter += 1
    try:
        await asyncio.sleep(SCARE_FORGET_TIME)
    finally:
        scare_counter -= 1


class FrameGrabber:
    def __init__(self, source):
        self.cap = cv.VideoCapture(source)
        self.cap.set(cv.CAP_PROP_FRAME_WIDTH, 1920)
        self.cap.set(cv.CAP_PROP_FRAME_HEIGHT, 1080)
        self.running = False
        self.latest_grabbed = False
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._grab_frames, daemon=True)

    def start(self):
        self.running = True
        self.thread.start()

    def stop(self):
        self.running = False
        self.thread.join()
        self.cap.release()

    def _grab_frames(self):
        while self.running:
            with self.lock:
                self.latest_grabbed = self.cap.grab()
            time.sleep(0.016)

    def retrieve_frame(self):
        with self.lock:
            if not self.latest_grabbed:
                return None
            ret, frame = self.cap.retrieve()
            self.latest_grabbed = False
            return frame if ret else None


def parse_args():
    parser = argparse.ArgumentParser(description="Boar detection system")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with verbose output and image display",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    model = YOLO("yolo11n-cls.pt", task="classify")
    device = "cpu"

    if args.debug:
        cv.namedWindow("Boar detector", cv.WINDOW_NORMAL)

    grabber = FrameGrabber(os.getenv("VIDEO_SOURCE"))
    grabber.start()

    try:
        while True:
            frame = grabber.retrieve_frame()
            if frame is None:
                print("Failed to get frame")
                continue

            if args.debug:
                cv.imshow("Boar detector", frame)

            if detect_boar(model, frame, device, args.debug):
                asyncio.create_task(scare_boar())
                await asyncio.sleep(MIN_SCARE_INTERVAL)
            else:
                await asyncio.sleep(DETECTION_INTERVAL)

            if args.debug and cv.waitKey(1) == ord("q"):
                break

    except Exception as e:
        print(f"Error: {e}")
    finally:
        grabber.stop()
        if args.debug:
            cv.destroyAllWindows()


if __name__ == "__main__":
    asyncio.run(main())
