# Multi-process motion detection. The main process is the UI and coordinator
# process. It spawns two other processes to handle video streaming and motion
# detection. A simple dictionary is passed between the processes to communicate
# the current frame and detected motion rectangles (which is what creates the
# overhead in this implementation alongside process signaling).
import sys
import cv2
import time
import os
import multiprocessing

from motion_detector import motion_detection_ctx, motion_detection_process
from pipeline import PipelineStep
from utils import RectMatcher, draw_results

g_video_capture = None
def streamer_main(ctx):
    """Stream video frames from an input file.
    
    Opens a video file specified in the context and yields frames one at a time.
    Uses a global VideoCapture object to maintain state between calls.
    
    Args:
        ctx: Dictionary containing:
            input_file: Path to video file to stream from
            
    Yields:
        Dictionary containing:
            frame: The next video frame as a numpy array
    """
    global g_video_capture
    if g_video_capture is None:
        g_video_capture = cv2.VideoCapture(ctx['input_file'])
    try:
        while True:
            if not g_video_capture.isOpened():
                print("Error loading video file.")
                return
            ret, frame = g_video_capture.read()
            if not ret:
                print("Error: Could not read frame.")
                return None
            yield {
                'frame': frame,
                'fps': g_video_capture.get(cv2.CAP_PROP_FPS),
            }
    finally:
        g_video_capture.release()

g_detector_ctx = None
def detector_main(ctx):
    """Process video frames to detect motion.
    
    Uses a global motion detector context to maintain state between calls.
    Processes each input frame to detect regions of motion.
    
    Args:
        ctx: Dictionary containing:
            frame: Video frame to process as numpy array
            
    Yields:
        Dictionary containing:
            frame: The input video frame
            detected_rects: List of rectangles in (x1,y1,x2,y2) format defining 
                regions where motion was detected
    """
    global g_detector_ctx
    if g_detector_ctx is None:
        g_detector_ctx = motion_detection_ctx()
    ctx['detected_rects'] = motion_detection_process(g_detector_ctx, ctx['frame'])
    yield ctx

def presenter_main():
    """Main UI process that displays the video feed with motion detection.
    
    Sets up the video processing pipeline and displays results in a window.
    Press 'q' to exit.

    This multiprocess implementation is much less efficient than the single
    thread one due to the overhead of inter-process communication. If we really
    needed multiprocessing we'd have to shared memory to avoid copying which is
    outside the scope of this project.
    """
    if len(sys.argv) < 2:
        print('Usage: pipenv run python3 main-multi-process.py <input_location>')
        print('Continuing with sample video...')
        input_file = os.path.abspath('motion-detection-computer-room-door-1920x1080.mp4')
    else:
      input_file = os.path.abspath(sys.argv[1])

    detected_rects = RectMatcher()
    display_queue = multiprocessing.Queue()
    streamer_process = PipelineStep(streamer_main)
    detector_process = PipelineStep(detector_main,
                                    input_queue=streamer_process.output_queue,
                                    output_queue=display_queue)
    streamer_process.start()
    detector_process.start()
    try:
      streamer_process.input({
          'input_file': input_file
      })
      last_frame = time.time()
      while True:
          ctx = display_queue.get()
          detected_rects.set(ctx['detected_rects'])
          frame = draw_results(ctx['frame'],
                               detected_rects.rects(),
                               last_frame)
          cv2.imshow("Frame", frame)
          last_frame = time.time()
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
    finally:
      detector_process.stop()
      streamer_process.stop()
      cv2.destroyAllWindows()
if __name__ == "__main__":
    presenter_main()