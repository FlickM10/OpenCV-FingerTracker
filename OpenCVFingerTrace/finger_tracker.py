import cv2
import numpy as np
import collections

# Define the thicker line width (Increased to 25 for a denser, clearer trace)
LINE_THICKNESS = 25 

class FingerTracker:
    """
    Handles hand detection, trace recording, smoothing, and trace image preparation.
    """
    def __init__(self, trace_buffer=100, smoothing_window=7):
        self.trace_buffer = trace_buffer
        self.smoothing_window = smoothing_window
        
        self.trace = collections.deque(maxlen=self.trace_buffer)
        self.raw_trace_buffer = collections.deque(maxlen=self.smoothing_window)
        
        self.is_drawing = False
        self.current_frame_dims = (0, 0)
        self.trace_bbox = None 

    def _get_hand_center(self, frame):
        """Finds the center of the largest skin-colored contour."""
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Wider skin color range (HSV) for more robust detection
        lower_skin = np.array([0, 10, 60], dtype=np.uint8) 
        upper_skin = np.array([25, 255, 255], dtype=np.uint8)
        
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)
        mask = cv2.erode(mask, kernel, iterations=1)
        mask = cv2.GaussianBlur(mask, (7, 7), 0)
        
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            # Filter small noise
            if cv2.contourArea(max_contour) > 2500:
                M = cv2.moments(max_contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    
                    cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)
                    
                    self.is_drawing = True 
                    return (center_x, center_y)
        
        self.is_drawing = False
        return None

    def _smooth_point(self, point):
        """Applies a simple moving average filter."""
        self.raw_trace_buffer.append(point)

        # Relaxed check: allows drawing sooner
        if len(self.raw_trace_buffer) < 2: 
            return point 

        x_avg = int(sum(p[0] for p in self.raw_trace_buffer) / len(self.raw_trace_buffer))
        y_avg = int(sum(p[1] for p in self.raw_trace_buffer) / len(self.raw_trace_buffer))
        return (x_avg, y_avg)

    def process_frame(self, frame):
        """Updates the trace and draws it on the frame."""
        self.current_frame_dims = (frame.shape[1], frame.shape[0])
        
        center_point = self._get_hand_center(frame)

        if self.is_drawing and center_point:
            smoothed_point = self._smooth_point(center_point)
            self.trace.append(smoothed_point)
                
            # Update the bounding box for cropping later
            if not self.trace_bbox:
                self.trace_bbox = (smoothed_point[0], smoothed_point[1], smoothed_point[0], smoothed_point[1])
            else:
                x_min, y_min, x_max, y_max = self.trace_bbox
                self.trace_bbox = (
                    min(x_min, smoothed_point[0]),
                    min(y_min, smoothed_point[1]),
                    max(x_max, smoothed_point[0]),
                    max(y_max, smoothed_point[1])
                )
        
        # Draw the magenta trace using the thicker line width (LINE_THICKNESS = 25)
        for i in range(1, len(self.trace)):
            cv2.line(frame, self.trace[i-1], self.trace[i], (255, 0, 255), LINE_THICKNESS)
            
        return frame

    def get_trace_image(self):
        """Renders the trace onto a minimal, cropped, black-and-white canvas."""
        if not self.trace or not self.trace_bbox:
            return None

        W, H = self.current_frame_dims
        x1, y1, x2, y2 = self.trace_bbox

        # Add padding
        padding = 30
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(W, x2 + padding)
        y2 = min(H, y2 + padding)

        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w <= 0 or crop_h <= 0:
            return None

        # Create a blank canvas (black background, white drawing - 255)
        canvas = np.zeros((crop_h, crop_w), dtype=np.uint8)

        # Draw the trace relative to the new origin using the thicker line width
        for i in range(1, len(self.trace)):
            p1 = (self.trace[i-1][0] - x1, self.trace[i-1][1] - y1)
            p2 = (self.trace[i][0] - x1, self.trace[i][1] - y1)
            cv2.line(canvas, p1, p2, 255, LINE_THICKNESS) 

        return canvas

    def clear_trace(self):
        """Clears all drawing data."""
        self.trace.clear()
        self.raw_trace_buffer.clear()
        self.trace_bbox = None
        self.is_drawing = False
