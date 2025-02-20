# Single-threaded motion detection
import os
import sys
import cv2
import time

from motion_detector import motion_detection_ctx, motion_detection_process
from utils import FPS_60_MS, RectMatcher, draw_results


def main():
    """
    Main motion detection loop which captures frames from the main camera,
    and processes synchronously. Since OpenCV implements concurrency internally
    for it's image processing functions, this actually runs at a higher FPS
    than a multiprocess based approach which suffers from cross-process memory
    copying overhead. Multi threading will suffer the same fate due to the GIL.

    On my MBP 15" 2018, this code runs at ~200 FPS on the provided video file.
    """
    if len(sys.argv) < 2:
        print('Usage: pipenv run python3 main-multi-process.py <input_location>')
        print('Continuing with sample video...')
        input_file = os.path.abspath('motion-detection-computer-room-door-1920x1080.mp4')
    else:
      input_file = os.path.abspath(sys.argv[1])
    cap = cv2.VideoCapture(input_file)
    detected_rects = RectMatcher()
    motion_detector = motion_detection_ctx()
    last_frame = time.time()
    try:
        while True:
            if not cap.isOpened():
                print("Error loading video file.")
                return
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                return None
            detected_rects.set(motion_detection_process(motion_detector, frame))
            frame = draw_results(frame,
                                 detected_rects.rects(),
                                 last_frame)
            cv2.imshow("Frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            last_frame = time.time()
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()