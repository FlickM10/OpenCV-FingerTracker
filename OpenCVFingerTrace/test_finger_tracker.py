import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time

class FingerTracker:
    def __init__(self, trace_buffer=50, smoothing_window=5):
        """
        Initialize finger tracking with MediaPipe
        trace_buffer: number of points to keep in trace history
        smoothing_window: window size for moving average smoothing
        """
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Trace storage
        self.index_trace = deque(maxlen=trace_buffer)
        self.middle_trace = deque(maxlen=trace_buffer)
        
        # Smoothing buffer
        self.smoothing_window = smoothing_window
        self.index_smooth_buffer = deque(maxlen=smoothing_window)
        self.middle_smooth_buffer = deque(maxlen=smoothing_window)
        
        # Drawing state
        self.is_drawing = False
        self.drawing_canvas = None
        
        # Gesture detection
        self.prev_pinch_distance = None
        self.pinch_threshold = 0.05  # Threshold for detecting two-finger pinch
        
    def get_smoothed_point(self, buffer):
        """Apply moving average smoothing to reduce jitter"""
        if len(buffer) == 0:
            return None
        
        # Calculate average of points in buffer
        avg_x = sum(p[0] for p in buffer) / len(buffer)
        avg_y = sum(p[1] for p in buffer) / len(buffer)
        return (int(avg_x), int(avg_y))
    
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def detect_two_fingers_extended(self, hand_landmarks, image_shape):
        """
        Detect if index and middle fingers are extended and close together
        Returns: (is_two_fingers, index_tip, middle_tip)
        """
        h, w = image_shape[:2]
        
        # Get finger tip and base positions
        index_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_TIP]
        index_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.INDEX_FINGER_PIP]
        middle_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
        middle_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.MIDDLE_FINGER_PIP]
        ring_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_TIP]
        ring_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.RING_FINGER_PIP]
        pinky_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_TIP]
        pinky_pip = hand_landmarks.landmark[self.mp_hands.HandLandmark.PINKY_PIP]
        thumb_tip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_TIP]
        thumb_ip = hand_landmarks.landmark[self.mp_hands.HandLandmark.THUMB_IP]
        
        # Check if index and middle are extended (tip higher than pip)
        index_extended = index_tip.y < index_pip.y
        middle_extended = middle_tip.y < middle_pip.y
        
        # Check if ring and pinky are NOT extended (folded)
        ring_folded = ring_tip.y > ring_pip.y
        pinky_folded = pinky_tip.y > pinky_pip.y
        
        # Check if thumb is folded
        thumb_folded = abs(thumb_tip.x - thumb_ip.x) < 0.05
        
        # Calculate distance between index and middle tips
        index_pos = (int(index_tip.x * w), int(index_tip.y * h))
        middle_pos = (int(middle_tip.x * w), int(middle_tip.y * h))
        distance = self.calculate_distance(index_pos, middle_pos)
        
        # Two fingers detected if:
        # 1. Index and middle extended
        # 2. Ring and pinky folded
        # 3. Tips are close together (drawing mode)
        two_fingers = (index_extended and middle_extended and 
                      ring_folded and pinky_folded and 
                      distance < 50)  # pixels
        
        return two_fingers, index_pos, middle_pos
    
    def process_frame(self, frame):
        """
        Process frame and detect finger tracking
        Returns: processed frame with overlays
        """
        h, w = frame.shape[:2]
        
        # Initialize canvas if needed
        if self.drawing_canvas is None:
            self.drawing_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        # Create output frame
        output_frame = frame.copy()
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand skeleton
                self.mp_draw.draw_landmarks(
                    output_frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2)
                )
                
                # Detect two-finger gesture
                two_fingers, index_pos, middle_pos = self.detect_two_fingers_extended(
                    hand_landmarks, frame.shape
                )
                
                if two_fingers:
                    self.is_drawing = True
                    
                    # Calculate center point between two fingers
                    center_x = (index_pos[0] + middle_pos[0]) // 2
                    center_y = (index_pos[1] + middle_pos[1]) // 2
                    center_pos = (center_x, center_y)
                    
                    # Add to smoothing buffer
                    self.index_smooth_buffer.append(center_pos)
                    
                    # Get smoothed point
                    smoothed_point = self.get_smoothed_point(self.index_smooth_buffer)
                    
                    if smoothed_point:
                        # Add to trace
                        self.index_trace.append(smoothed_point)
                        
                        # Draw circle at current position
                        cv2.circle(output_frame, smoothed_point, 8, (0, 255, 255), -1)
                        cv2.circle(self.drawing_canvas, smoothed_point, 5, (255, 255, 255), -1)
                    
                    # Draw indicator circles on fingertips
                    cv2.circle(output_frame, index_pos, 8, (255, 0, 255), 2)
                    cv2.circle(output_frame, middle_pos, 8, (255, 0, 255), 2)
                    
                    # Draw line between fingers
                    cv2.line(output_frame, index_pos, middle_pos, (0, 255, 255), 2)
                    
                else:
                    self.is_drawing = False
        
        # Draw the complete trace with smoothing
        if len(self.index_trace) > 1:
            for i in range(1, len(self.index_trace)):
                if self.index_trace[i-1] is None or self.index_trace[i] is None:
                    continue
                
                # Varying thickness for better visualization
                thickness = int(np.sqrt(64 / (i + 1)) + 2)
                cv2.line(output_frame, self.index_trace[i-1], self.index_trace[i], 
                        (0, 255, 0), thickness)
                cv2.line(self.drawing_canvas, self.index_trace[i-1], self.index_trace[i],
                        (255, 255, 255), 3)
        
        # Overlay canvas on frame
        mask = cv2.cvtColor(self.drawing_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        
        output_frame = cv2.bitwise_and(output_frame, output_frame, mask=mask_inv)
        canvas_colored = cv2.cvtColor(self.drawing_canvas, cv2.COLOR_BGR2RGB)
        output_frame = cv2.add(output_frame, canvas_colored)
        
        return output_frame
    
    def get_drawing_canvas(self):
        """Return the current drawing canvas"""
        return self.drawing_canvas.copy()
    
    def clear_trace(self):
        """Clear the current trace"""
        self.index_trace.clear()
        self.middle_trace.clear()
        self.index_smooth_buffer.clear()
        self.middle_smooth_buffer.clear()
        if self.drawing_canvas is not None:
            self.drawing_canvas.fill(0)
    
    def get_trace_image(self, size=(28, 28)):
        """
        Get the trace as a resized binary image for digit recognition
        Returns preprocessed image ready for model input
        """
        if self.drawing_canvas is None:
            return None
        
        # Convert to grayscale
        gray = cv2.cvtColor(self.drawing_canvas, cv2.COLOR_BGR2GRAY)
        
        # Find bounding box of the drawing
        coords = cv2.findNonZero(gray)
        if coords is None:
            return None
        
        x, y, w, h = cv2.boundingRect(coords)
        
        # Add padding
        padding = 20
        x = max(0, x - padding)
        y = max(0, y - padding)
        w = min(self.drawing_canvas.shape[1] - x, w + 2*padding)
        h = min(self.drawing_canvas.shape[0] - y, h + 2*padding)
        
        # Crop to bounding box
        cropped = gray[y:y+h, x:x+w]
        
        # Make it square by padding
        if w > h:
            diff = w - h
            pad_top = diff // 2
            pad_bottom = diff - pad_top
            cropped = cv2.copyMakeBorder(cropped, pad_top, pad_bottom, 0, 0, 
                                        cv2.BORDER_CONSTANT, value=0)
        elif h > w:
            diff = h - w
            pad_left = diff // 2
            pad_right = diff - pad_left
            cropped = cv2.copyMakeBorder(cropped, 0, 0, pad_left, pad_right,
                                        cv2.BORDER_CONSTANT, value=0)
        
        # Resize to target size
        resized = cv2.resize(cropped, size, interpolation=cv2.INTER_AREA)
        
        # Normalize
        normalized = resized.astype(np.float32) / 255.0
        
        return normalized
    
    def close(self):
        """Release resources"""
        self.hands.close()


def main():
    # Initialize camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # Initialize tracker
    tracker = FingerTracker(trace_buffer=100, smoothing_window=7)
    
    print("=" * 60)
    print("FINGER TRACE DIGIT RECOGNITION")
    print("=" * 60)
    print("\nInstructions:")
    print("  - Show TWO FINGERS (index + middle) close together to draw")
    print("  - Keep other fingers folded")
    print("  - Move your hand to draw digits in the air")
    print("\nControls:")
    print("  'c' - Clear the current trace")
    print("  's' - Save trace image (for training/testing)")
    print("  'p' - Process digit (future: send to neural network)")
    print("  'q' - Quit")
    print("=" * 60)
    
    frame_count = 0
    fps_time = time.time()
    fps = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip frame horizontally for mirror view
            frame = cv2.flip(frame, 1)
            
            # Process frame
            output_frame = tracker.process_frame(frame)
            
            # Calculate FPS
            frame_count += 1
            if frame_count % 10 == 0:
                fps = 10 / (time.time() - fps_time)
                fps_time = time.time()
            
            # Add UI text
            cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            status = "DRAWING" if tracker.is_drawing else "IDLE"
            color = (0, 255, 0) if tracker.is_drawing else (0, 0, 255)
            cv2.putText(output_frame, f"Status: {status}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            cv2.putText(output_frame, "Show 2 fingers close together to draw", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            
            # Show controls
            cv2.putText(output_frame, "C:Clear | S:Save | P:Process | Q:Quit", 
                       (10, output_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
            
            # Display
            cv2.imshow('Finger Trace - Digit Recognition', output_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("\nQuitting...")
                break
            
            elif key == ord('c'):
                print("Clearing trace...")
                tracker.clear_trace()
            
            elif key == ord('s'):
                trace_img = tracker.get_trace_image()
                if trace_img is not None:
                    filename = f'digit_trace_{int(time.time())}.png'
                    cv2.imwrite(filename, (trace_img * 255).astype(np.uint8))
                    print(f"Saved trace to {filename}")
                else:
                    print("No trace to save!")
            
            elif key == ord('p'):
                trace_img = tracker.get_trace_image()
                if trace_img is not None:
                    print("Processing digit...")
                    # Show the preprocessed image
                    cv2.imshow('Preprocessed Digit', cv2.resize(trace_img, (280, 280)))
                    print("Shape:", trace_img.shape)
                    print("Ready to send to neural network!")
                    # Here you would call: model.predict(trace_img)
                else:
                    print("No trace to process!")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        print("Cleanup complete")


if __name__ == "__main__":
    main()
