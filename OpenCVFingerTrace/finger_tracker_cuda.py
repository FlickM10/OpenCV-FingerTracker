import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from collections import deque
import time

# --- GLOBAL VARIABLES ---
MODEL = None
predicted_digit = "N/A"
confidence = 0.0
tracker = None 
AUTO_CLEAR_DELAY = 1.0 # Time (in seconds) to show the result before clearing
clear_time = 0.0 # Tracks when the trace should be cleared

# --- FINGER TRACKER CLASS (MediaPipe Implementation) ---

class FingerTracker:    
    def __init__(self, trace_buffer=50, smoothing_window=5):        
        self.mp_hands = mp.solutions.hands        
        # --- OPTIMIZATION: model_complexity=0 for lowest resource usage ---
        self.hands = self.mp_hands.Hands(            
            static_image_mode=False,
            model_complexity=0, # Use the lowest complexity model (fastest)            
            max_num_hands=1,            
            min_detection_confidence=0.5, # Slightly lowered for faster lock-on
            min_tracking_confidence=0.5 # Slightly lowered for faster lock-on        
        )        
        self.mp_draw = mp.solutions.drawing_utils                
        self.index_trace = deque(maxlen=trace_buffer)        
        self.middle_trace = deque(maxlen=trace_buffer)                
        self.smoothing_window = smoothing_window        
        self.index_smooth_buffer = deque(maxlen=smoothing_window)        
        self.middle_smooth_buffer = deque(maxlen=smoothing_window)                
        self.is_drawing = False        
        self.drawing_canvas = None                
        self.prev_pinch_distance = None        
        self.pinch_threshold = 0.05
            
    def get_smoothed_point(self, buffer):        
        if len(buffer) == 0:            
            return None                
        avg_x = sum(p[0] for p in buffer) / len(buffer)        
        avg_y = sum(p[1] for p in buffer) / len(buffer)        
        return (int(avg_x), int(avg_y))        

    def calculate_distance(self, point1, point2):        
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)        

    def detect_two_fingers_extended(self, hand_landmarks, image_shape):        
        h, w = image_shape[:2]                
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
        index_extended = index_tip.y < index_pip.y        
        middle_extended = middle_tip.y < middle_pip.y                
        ring_folded = ring_tip.y > ring_pip.y        
        pinky_folded = pinky_tip.y > pinky_pip.y                
        thumb_folded = abs(thumb_tip.x - thumb_ip.x) < 0.05                
        index_pos = (int(index_tip.x * w), int(index_tip.y * h))        
        middle_pos = (int(middle_tip.x * w), int(middle_tip.y * h))        
        distance = self.calculate_distance(index_pos, middle_pos)                
        two_fingers = (index_extended and middle_extended and                       
                       ring_folded and pinky_folded and                       
                       distance < 50)  # pixels                
        return two_fingers, index_pos, middle_pos        

    def process_frame(self, frame):        
        h, w = frame.shape[:2]                
        if self.drawing_canvas is None:            
            self.drawing_canvas = np.zeros((h, w, 3), dtype=np.uint8)                
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)        
        results = self.hands.process(rgb_frame)                
        output_frame = frame.copy()                

        self.is_drawing = False # Reset status for the current frame
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
                two_fingers, index_pos, middle_pos = self.detect_two_fingers_extended(                    
                    hand_landmarks, frame.shape                
                )                                
                if two_fingers:                    
                    self.is_drawing = True                                        
                    center_x = (index_pos[0] + middle_pos[0]) // 2                    
                    center_y = (index_pos[1] + middle_pos[1]) // 2                    
                    center_pos = (center_x, center_y)                                        
                    self.index_smooth_buffer.append(center_pos)                                        
                    smoothed_point = self.get_smoothed_point(self.index_smooth_buffer)                                        
                    if smoothed_point:                        
                        self.index_trace.append(smoothed_point)                                                
                        cv2.circle(output_frame, smoothed_point, 8, (0, 255, 255), -1)
                        cv2.circle(self.drawing_canvas, smoothed_point, 10, (255, 255, 255), -1)                                        
                        cv2.circle(output_frame, index_pos, 8, (255, 0, 255), 2)
                        cv2.circle(output_frame, middle_pos, 8, (255, 0, 255), 2)                                        
                        cv2.line(output_frame, index_pos, middle_pos, (0, 255, 255), 2)                                    

        if len(self.index_trace) > 1:            
            for i in range(1, len(self.index_trace)):                
                if self.index_trace[i-1] is None or self.index_trace[i] is None:                    
                    continue                                
                thickness = int(np.sqrt(64 / (i + 1)) + 2)
                
                cv2.line(output_frame, self.index_trace[i-1], self.index_trace[i],                         
                         (0, 255, 0), thickness)
                
                cv2.line(self.drawing_canvas, self.index_trace[i-1], self.index_trace[i],                        
                         (255, 255, 255), 10) 

        mask = cv2.cvtColor(self.drawing_canvas, cv2.COLOR_BGR2GRAY)        
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)        
        mask_inv = cv2.bitwise_not(mask)                
        output_frame = cv2.bitwise_and(output_frame, output_frame, mask=mask_inv)        
        canvas_colored = cv2.cvtColor(self.drawing_canvas, cv2.COLOR_BGR2RGB)        
        output_frame = cv2.add(output_frame, canvas_colored)                

        return output_frame        

    def get_drawing_canvas(self):        
        return self.drawing_canvas.copy()        

    def clear_trace(self):        
        self.index_trace.clear()        
        self.middle_trace.clear()        
        self.index_smooth_buffer.clear()        
        self.middle_smooth_buffer.clear()        
        if self.drawing_canvas is not None:            
            self.drawing_canvas.fill(0)        

    def get_trace_image(self, size=(28, 28)):        
        if self.drawing_canvas is None:            
            return None                
        gray = cv2.cvtColor(self.drawing_canvas, cv2.COLOR_BGR2GRAY)                
        coords = cv2.findNonZero(gray)        
        if coords is None:            
            return None                
        x, y, w, h = cv2.boundingRect(coords)                
        padding = 20        
        x = max(0, x - padding)        
        y = max(0, y - padding)        
        w = min(self.drawing_canvas.shape[1] - x, w + 2*padding)        
        h = min(self.drawing_canvas.shape[0] - y, h + 2*padding)                
        cropped = gray[y:y+h, x:x+w]                
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
        resized = cv2.resize(cropped, size, interpolation=cv2.INTER_AREA)                
        normalized = resized.astype(np.float32) / 255.0                

        return normalized        

    def close(self):        
        self.hands.close()

# --- HELPER FUNCTIONS FOR TENSORFLOW ---

def preprocess_image_for_model(img):
    """
    Ensures the image is ready for the model's predict function.
    """
    if img is None:
        return None
    
    # Reshape for the CNN: (1, 28, 28, 1)
    return img.reshape(1, 28, 28, 1).astype('float32')

def predict_digit():
    """Performs digit recognition using the TensorFlow model, explicitly on the CPU."""
    global MODEL, predicted_digit, confidence
    
    trace_img = tracker.get_trace_image()
    if trace_img is None:
        predicted_digit = "N/A"
        confidence = 0.0
        return
        
    processed_img = preprocess_image_for_model(trace_img)

    device = '/CPU:0' 
    
    try:
        with tf.device(device):
            prediction = MODEL.predict(processed_img, verbose=0) 
            
        digit = np.argmax(prediction)
        confidence = np.max(prediction)
        predicted_digit = str(digit)
        # print(f"Prediction: {predicted_digit}, Confidence: {confidence:.2f}")

    except Exception as e:
        # print(f"RUNTIME PREDICTION ERROR (TF_FAIL): {e}")
        predicted_digit = "TF_FAIL"
        confidence = 0.0

def setup_model_and_device():
    """Configures TensorFlow, forcing model loading onto the CPU due to known CUDA errors."""
    global MODEL
    print("--- Setting up TensorFlow ---")
    
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print("✅ GPU detected but running on CPU due to compatibility.")
        else:
            print("⚠️ No GPU detected. Running on CPU.")
    except Exception as e:
        print(f"❌ Failed to check GPU configuration: {e}")
        
    # --- Model Loading (Crucial: Force device to /CPU:0) ---
    try:
        device = '/CPU:0' 
        with tf.device(device):
            MODEL = tf.keras.models.load_model('digit_model.h5')
        print(f"✅ Model 'digit_model.h5' loaded successfully on {device}.")
    except Exception as e:
        print(f"❌ FATAL ERROR: Could not load model 'digit_model.h5'. Check file path. Error: {e}")
        MODEL = None
        
# --- MAIN APPLICATION LOOP ---

def main():
    global predicted_digit, confidence, tracker, clear_time

    # 1. Setup Model
    setup_model_and_device()
    if MODEL is None:
        print("Application exiting due to failed model load.")
        return

    # 2. Initialize Camera
    cap = cv2.VideoCapture(0)
    
    # --- STABLE RESOLUTION (640x480) ---
    TARGET_WIDTH = 640
    TARGET_HEIGHT = 480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
    # Re-request 60 FPS (may or may not work depending on driver/hardware)
    cap.set(cv2.CAP_PROP_FPS, 60) 

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # 3. Initialize Tracker
    tracker = FingerTracker(trace_buffer=100, smoothing_window=7)
    
    prev_is_drawing = False

    print("=" * 60)
    print("FINGER TRACE DIGIT RECOGNITION (640x480 Stable)")
    print("=" * 60)
    print("Instructions: Use extended index and middle fingers (others folded) to draw.")
    print(f"Running at {TARGET_WIDTH}x{TARGET_HEIGHT} resolution.")
    print("The trace clears automatically after showing the prediction.")
    print("Controls: 'c': Clear | 's': Save | 'q': Quit")
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

            current_time = time.time()
            
            frame = cv2.flip(frame, 1) 
            output_frame = tracker.process_frame(frame)

            # --- AUTO-PREDICTION & AUTO-CLEAR LOGIC ---
            
            # Check 1: Did drawing just stop? (True -> False transition)
            if prev_is_drawing and not tracker.is_drawing:
                # print("Drawing complete. Auto-predicting digit...")
                predict_digit()
                
                # Set the clear time
                if predicted_digit not in ["N/A", "TF_FAIL"]:
                    clear_time = current_time + AUTO_CLEAR_DELAY
            
            # Check 2: Should we clear the trace? (Time expired OR a new drawing starts)
            if (current_time >= clear_time and clear_time != 0.0) or (clear_time != 0.0 and tracker.is_drawing):
                # print("Auto-clearing trace.")
                tracker.clear_trace()
                predicted_digit = "N/A"
                confidence = 0.0
                clear_time = 0.0 # Reset timer

            # Update the previous state
            prev_is_drawing = tracker.is_drawing 
            # --- END AUTO-PREDICTION & AUTO-CLEAR LOGIC ---

            # Calculate FPS
            frame_count += 1
            if frame_count % 10 == 0:
                fps = 10 / (current_time - fps_time)
                fps_time = current_time

            # --- UI RENDERING ---

            # FPS
            cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Status
            status = "DRAWING" if tracker.is_drawing else "IDLE"
            status_color = (0, 255, 0) if tracker.is_drawing else (0, 0, 255)
            cv2.putText(output_frame, f"Status: {status}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            
            # Prediction confirmation
            if predicted_digit != "N/A":
                # Display the prediction number in large text
                prediction_text = f"Prediction = {predicted_digit}"
                # Use a larger font/thickness for emphasis
                cv2.putText(output_frame, prediction_text, 
                            (output_frame.shape[1] - 400, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 0), 3)
                
                # Still display confidence
                cv2.putText(output_frame, f"Conf: {confidence:.2f}", 
                            (output_frame.shape[1] - 180, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            else:
                 cv2.putText(output_frame, f"Pred: {predicted_digit}", (output_frame.shape[1] - 180, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


            # Controls 
            cv2.putText(output_frame, "C:Clear | S:Save | Q:Quit",
                        (10, output_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Display frame
            cv2.imshow('Finger Trace - Digit Recognition', output_frame)

            # --- KEYBOARD HANDLER (Manual Clear takes precedence) ---
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('c'):
                # print("Manual clear initiated.")
                tracker.clear_trace()
                predicted_digit = "N/A"
                confidence = 0.0
                clear_time = 0.0 # Cancel any pending auto-clear
            elif key == ord('s'):
                trace_img = tracker.get_trace_image()
                if trace_img is not None:
                    filename = f'digit_trace_{int(time.time())}.png'
                    # Scale back to 255 for saving as PNG
                    cv2.imwrite(filename, (trace_img * 255).astype(np.uint8)) 
                    print(f"Saved trace to {filename}")
                else:
                    print("No trace to save!")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close() # Close detector here
        print("Cleanup complete")


if __name__ == "__main__":
    main()
