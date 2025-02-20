import datetime
import time
import cv2

# 60 FPS
FPS_60_MS = 1 / (1000 / 60)

class MovingAverage:
    """A class that maintains a moving average over a fixed window of values.
    
    Keeps track of the most recent N values and provides their average, where N
    is the window size specified at initialization. When the window is full,
    adding new values causes the oldest values to be removed.
    """

    def __init__(self, window_size):
        """Initialize a MovingAverage with the given window size.

        Args:
            window_size: Integer size of the moving average window
        """
        self.window_size = window_size
        self.values = []

    def add(self, value):
        """Add a new value to the moving average window.
        
        Appends the value to the window and removes the oldest value if the
        window is full.

        Args:
            value: Numeric value to add to the window
        """
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

    def average(self):
        """Calculate the current average of values in the window.
        
        Returns:
            Float average of the current values, or 0 if the window is empty
        """
        return 0 if len(self.values) == 0 else sum(self.values) / len(self.values)

class SmoothRect:
   """A class that smooths rectangle coordinates using moving averages.
   
   Maintains separate moving averages for each corner coordinate of a rectangle
   to provide smoothed motion over time. This helps reduce jitter in rectangle
   tracking.
   """

   def __init__(self, rect, window_size=30):
      """Initialize a SmoothRect with the given rectangle and window size."""
      self.x1 = MovingAverage(window_size)
      self.y1 = MovingAverage(window_size)
      self.x2 = MovingAverage(window_size)
      self.y2 = MovingAverage(window_size)
      self.update(rect)

   def update(self, rect):
      """Update the smoothed coordinates with a new rectangle.
      
      Args:
          rect: New rectangle coordinates in (x1,y1,x2,y2) format
      """
      self.x1.add(rect[0])
      self.y1.add(rect[1])
      self.x2.add(rect[2])
      self.y2.add(rect[3])
   
   def rect(self):
      """Get the current smoothed rectangle coordinates.
      
      Returns:
          Tuple of (x1,y1,x2,y2) containing the latest coordinates.
      """
      return (round(self.x1.average()),
              round(self.y1.average()),
              round(self.x2.average()),
              round(self.y2.average()))

def rect_dist(r1, r2):
   """Calculate the distance between the centers of two rectangles.
   
   Takes two rectangles in (x1,y1,x2,y2) format and computes the distance
   between their center points.

   Args:
       r1: First rectangle in (x1,y1,x2,y2) format where:
           x1,y1: Top-left corner coordinates
           x2,y2: Bottom-right corner coordinates
       r2: Second rectangle in same format as r1

   Returns:
       Float value representing the Euclidean distance between the centers
       of the two rectangles.
   """
   x1 = (r1[0] + r1[2]) / 2
   y1 = (r1[1] + r1[3]) / 2
   x2 = (r2[0] + r2[2]) / 2
   y2 = (r2[1] + r2[3]) / 2
   return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

class RectMatcher:
   """Matches and smooths rectangles across frames.
   
   Tracks detected rectangles across multiple frames by matching them based on
   their proximity. Each tracked rectangle has its coordinates smoothed over
   time to reduce jitter.
   """
   def __init__(self, dist=10):
      self.smoothRects = []
      self.dist = dist

   def set(self, rects):
      """Update tracked rectangles with new detections.
      
      Matches new rectangles with existing tracked rectangles based on their
      proximity. Creates new tracked rectangles for any unmatched detections
      and deletes old unmatched ones.

      Args:
          rects: List of rectangles in (x1,y1,x2,y2)
      """
      new_smoothRects = []
      for new_rect in rects:
         found = False
         for r in self.smoothRects:
            if rect_dist(new_rect, r.rect()) < self.dist:
              r.update(new_rect)
              new_smoothRects.append(r)
              found = True
              break
         if not found:
            new_smoothRects.append(SmoothRect(new_rect))
      self.smoothRects = new_smoothRects
   
   def rects(self):
      """Get the current smoothed rectangles.
      
      Returns:
          List of smoothed rectangles in (x1,y1,x2,y2)
      """
      return [r.rect() for r in self.smoothRects]

def blur_rects(image, rects, factor=3):
    """Apply blur to rectangular regions in an image.
    
    For each rectangle in the input list, extracts that region from the image
    and blurs it. The blur kernel size is automatically determined based on the
    factor parameter.
    
    Args:
        image: Input image (numpy array) to blur regions of
        rects: List of rectangles in (x1,y1,x2,y2) format
        factor: Integer factor to divide rectangle dimensions by to determine
            blur kernel size. Larger values produce less blurring.
            
    Returns:
        None. The input image is modified in-place.
    """
    for r in rects:
        x1, y1, x2, y2 = r
        blur_frame = image[y1:y2, x1:x2]
        # automatically determine the size of the blurring kernel based
        # on the spatial dimensions of the input image
        (h, w) = blur_frame.shape[:2]
        if w <= 0 or h <= 0:
            continue
        kW = int(w / factor)
        kH = int(h / factor)
        if kW % 2 == 0:
            kW -= 1
        if kH % 2 == 0:
            kH -= 1
        image[y1:y2, x1:x2] = cv2.blur(blur_frame, (kW, kH), 0)

# Moving average of FPS over the last 60 frames
fps_avg = MovingAverage(60)

def draw_results(frame, rects, capture_time):
    """Draw detection results on a frame.
    
    Draws rectangles around detected motion regions, blurs those regions,
    and adds timestamp and FPS overlay text.
    
    Args:
        frame: Input image (numpy array) to draw on
        rects: List of rectangles in (x1,y1,x2,y2) format defining detected
            motion regions
        capture_time: Timestamp when frame was captured, used for FPS
            calculation
            
    Returns:
        The input frame with detection results drawn on it
    """
    blur_rects(frame, rects)
    for x1, y1, x2, y2 in rects:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    timestamp_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fps_avg.add(1 / (time.time() - capture_time))
    cv2.putText(frame,
                f"{timestamp_str} FPS: {fps_avg.average():.2f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2)
    return frame