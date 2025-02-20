from functools import cmp_to_key
import math
import cv2
import imutils

from utils import rect_dist

# Maximum size in pixels for the longest axis of the frame when doing detection.
# Frames larger than this will be scaled down to improve performance while
# maintaining reasonable accuracy.
DETECTION_SIZE = 500

def motion_detection_ctx(min_area=50):
    """Create a context dictionary for motion detection.
    
    Initializes a dictionary containing the state needed for motion detection
    between video frames. This includes reference and current frames in both
    color and grayscale, minimum contour area for motion detection, and list
    of detected motion rectangles.

    Args:
        min_area: Minimum contour area in pixels to consider as motion.
                  Smaller areas are filtered out as noise. Default is 50.

    Returns:
        Dictionary containing motion detection state with keys:
            ref_frame: Previous color frame used as reference
            cur_frame: Current color frame being processed  
            ref_frame_gray: Grayscale version of reference frame
            cur_frame_gray: Grayscale version of current frame
            min_area: Minimum contour area threshold
    """
    return {
				'avg_frame_gray': None,
        'min_area': min_area,
    }

def fixup_rect(rect, scaleFactor):
    """Scale a rectangle by a given factor.
    
    Takes a rectangle in (x, y, w, h) format and returns a scaled rectangle
    in (x1, y1, x2, y2) format.

    Args:
        rect: Tuple of (x, y, width, height) defining the input rectangle
        scaleFactor: Factor to scale the rectangle by

    Returns:
        Tuple of (x1, y1, x2, y2) defining the scaled rectangle with:
            x1, y1: Top-left corner coordinates
            x2, y2: Bottom-right corner coordinates
    """
    return (round(rect[0] * scaleFactor),
            round(rect[1] * scaleFactor), 
            round((rect[0] + rect[2]) * scaleFactor),
            round((rect[1] + rect[3]) * scaleFactor))

def rect_diagonal(rect):
    """Calculate the diagonal length of a rectangle.
    
    Takes a rectangle in (x1, y1, x2, y2) format and returns the length of its
    diagonalusing the Pythagorean theorem.

    Args:
        rect: Tuple of (x1, y1, x2, y2) defining the rectangle coordinates

    Returns:
        Float value representing the diagonal length in pixels
    """
    return math.sqrt((rect[2] - rect[0]) ** 2 + (rect[3] - rect[1]) ** 2)

def bounding_rect(rects):
    """Calculate the minimum bounding rect containing all input rects.
    
    Takes a list of rects in (x1, y1, x2, y2) format and returns the smallest
    rects that contains all of them.

    Args:
        rects: List of tuples (x1, y1, x2, y2) defining input rectangles

    Returns:
        Tuple (x1, y1, x2, y2) defining the bounding rectangle coordinates
    """
    return (min([r[0] for r in rects]),
            min([r[1] for r in rects]), 
            max([r[2] for r in rects]),
            max([r[3] for r in rects]))

def merge_rects(rects):
    """Merge nearby rectangles into larger bounding rectangles.
    
    Takes a list of rectangles and iteratively merges pairs that are closer than
    their diagonal length (think bubble sort). This helps consolidate multiple
    detections of the same motion region into a single bounding rectangle.

    Args:
        rects: List of tuples (x1, y1, x2, y2) defining input rectangles

    Returns:
        List of merged rectangles in (x1, y1, x2, y2) format, with overlapping
        or nearby rectangles combined into larger bounding rectangles
    """
    rects.sort(key=cmp_to_key(lambda a, b: rect_dist(a, b)))
    did_merge = True
    while did_merge:
        did_merge = False
        for i in range(len(rects) - 2):
            dist = max(rect_diagonal(rects[i]), rect_diagonal(rects[i + 1]))    
            if rect_dist(rects[i], rects[i + 1]) < dist:
                rects[i] = bounding_rect([rects[i], rects[i + 1]])
                rects.pop(i + 1)
                did_merge = True
                break
    return rects

def motion_detection_process(ctx, frame):
	"""Process a video frame for motion detection.

	Takes a frame and motion detection context dictionary, updates the context
	with the new frame, and detects motion by comparing against the previous
	reference frame.
	For more info refer to `this link <https://pyimagesearch.com/2015/05/25/basic-motion-detection-and-tracking-with-python-and-opencv/>`

	Args:
		ctx: Motion detection context dictionary containing frame history and
		parameters frame: Current color video frame to process

	Returns:
		List of tuples (x1,y1,x2,y2) defining rectangles around detected motion regions,
		or the updated context dict if this is the first frame
	"""
	maxAxis = max(frame.shape[0], frame.shape[1])
	scaleFactor = 1 if maxAxis < DETECTION_SIZE else DETECTION_SIZE / maxAxis
	gray = cv2.cvtColor(cv2.resize(frame,
                                 (round(scaleFactor * frame.shape[1]),
                                 round(scaleFactor * frame.shape[0]))),
																 cv2.COLOR_BGR2GRAY)
	gray = cv2.GaussianBlur(gray, (21, 21), 0)
	# Push the current frame to the reference frame
	if ctx['avg_frame_gray'] is None is None:
		ctx['avg_frame_gray'] = gray.copy().astype("float")
		return []
	cv2.accumulateWeighted(gray, ctx['avg_frame_gray'], 0.5)
	# Compute the absolute difference between the current frame and the ref frame
	frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(ctx['avg_frame_gray']))
	thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
	# Dilate the thresholded image to fill in holes, then find contours on
	# thresholded image
	thresh = cv2.dilate(thresh, None, iterations=2)
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	# Loop over the contours
	rects = []
	for c in cnts:
		# If the contour is too small, ignore it
		if cv2.contourArea(c) < ctx["min_area"]:
			continue
		# Compute the bounding box for the contour and add it to the list of
		# detected rects
		(x, y, w, h) = cv2.boundingRect(c)
		rects.append(fixup_rect((x, y, w, h), 1 / scaleFactor))
	return merge_rects(rects)