import cv2
import numpy as np
import tensorflow as tf
import os

# --- CUDA CONFIGURATION AND GPU SETUP ---
def setup_gpu():
    """Configures TensorFlow to use the GPU with dynamic memory growth."""
    print("--- Setting up TensorFlow for CUDA ---")
    try:
        # Check available GPUs
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            # Restrict TensorFlow to only use the first GPU
            try:
                tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
                # Enable memory growth to prevent TensorFlow from allocating all GPU memory at once
                tf.config.experimental.set_memory_growth(gpus[0], True)
                print(f"✅ GPU detected and configured: {gpus[0].name}")
                return True
            except RuntimeError as e:
                # Memory growth must be set before GPUs have been initialized
                print(f"❌ Error configuring GPU: {e}")
                return False
        else:
            print("⚠️ No GPU detected. Running on CPU.")
            return False
    except Exception as e:
        print(f"❌ Failed to set up GPU configuration: {e}")
        return False

# Setup GPU before loading the model
setup_gpu()

# Load the pre-trained MNIST model
# Assuming 'digit_model.h5' is in the current directory.
# We explicitly place the model loading on the device that was configured (GPU or CPU)
print("Loading model 'digit_model.h5'...")
try:
    with tf.device('/GPU:0' if tf.config.experimental.list_physical_devices('GPU') else '/CPU:0'):
        # Ensure 'digit_model.h5' exists for this line to run successfully.
        model = tf.keras.models.load_model('digit_model.h5')
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Could not load model 'digit_model.h5'. Please ensure the file is present. Error: {e}")
    # Create a dummy model if load fails to prevent crash, allowing the rest of the app to start
    model = tf.keras.Sequential([tf.keras.layers.Dense(10)])


class FingerTracker:
    def __init__(self):
        self.trace = []
        # State to control when the application should record the finger position
        self.is_drawing = False 
        # State to store the bounding box reference for trace image cropping
        self.trace_bbox = None

    def update(self, frame):
        """Processes the frame to find the finger tip/hand and updates the trace."""
        center = None
        
        # Simple background subtraction (could be optimized with MOG2 or CV CUDA methods)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Using a fixed threshold; better real-world apps would use adaptive or background subtraction
        _, thresh = cv2.threshold(blur, 100, 255, cv2.THRESH_BINARY_INV) 

        # Find contours (this is typically a CPU operation in standard OpenCV)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            max_contour = max(contours, key=cv2.contourArea)
            # Filter contours that are too small (noise)
            if cv2.contourArea(max_contour) > 1500:  
                x, y, w, h = cv2.boundingRect(max_contour)
                
                # Draw green box on the live frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Calculate the center point for the trace
                center = (x + w // 2, y + h // 2)

                if self.is_drawing and center:
                    self.trace.append(center)

    def get_trace_image(self):
        """Renders the recorded trace onto a blank canvas for prediction."""
        if not self.trace:
            return None

        # 1. Determine bounding box of the entire trace
        trace_array = np.array(self.trace)
        x_min, y_min = np.min(trace_array, axis=0)
        x_max, y_max = np.max(trace_array, axis=0)
        
        # Add padding to ensure the digit isn't cut off
        padding = 20 
        
        # Dimensions of the live frame (assuming 640x480 standard webcam)
        H, W, _ = frame_placeholder.shape 

        # Define the cropping area
        x1 = max(0, x_min - padding)
        y1 = max(0, y_min - padding)
        x2 = min(W, x_max + padding)
        y2 = min(H, y_max + padding)

        crop_w = x2 - x1
        crop_h = y2 - y1

        if crop_w <= 0 or crop_h <= 0:
            return None

        # 2. Create a blank canvas the size of the cropped area
        # Note: We need to draw relative to the canvas origin (0, 0), not the frame origin (x1, y1)
        canvas = np.zeros((crop_h, crop_w), dtype=np.uint8)

        # 3. Draw the trace on the canvas, adjusted for the new origin
        for i in range(1, len(self.trace)):
            p1 = (self.trace[i-1][0] - x1, self.trace[i-1][1] - y1)
            p2 = (self.trace[i][0] - x1, self.trace[i][1] - y1)
            cv2.line(canvas, p1, p2, 255, 10) # Line thickness increased for MNIST look

        return canvas

    def clear(self):
        """Clears the drawing trace."""
        self.trace = []
        self.is_drawing = False


def preprocess_image(img):
    """Preprocesses the image for MNIST model inference (28x28, normalized)."""
    # Resize to 28x28 (MNIST required size)
    img_resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    
    # Invert colors (MNIST digits are white on black)
    # Note: cv2.bitwise_not is not strictly needed if we ensure the canvas is black (0) and trace is white (255)
    # but it's a safe step if the contour area wasn't perfect.
    
    # Normalize to [0, 1]
    img_normalized = img_resized.astype('float32') / 255.0
    
    # Add channel dimension (28, 28) -> (28, 28, 1)
    img_expanded = np.expand_dims(img_normalized, axis=-1)
    
    # Add batch dimension (1, 28, 28, 1)
    img_final = np.expand_dims(img_expanded, axis=0)
    return img_final

def predict_digit(img):
    """Performs digit recognition using the TensorFlow model on the GPU/CPU."""
    # Ensure the prediction runs on the configured device
    with tf.device('/GPU:0' if tf.config.experimental.list_physical_devices('GPU') else '/CPU:0'):
        processed_img = preprocess_image(img)
        # Using model.predict for prediction is the step accelerated by CUDA
        prediction = model.predict(processed_img, verbose=0) 
        
    digit = np.argmax(prediction)
    confidence = np.max(prediction)
    return digit, confidence

# Placeholder for the frame dimensions (will be set in main)
frame_placeholder = np.zeros((480, 640, 3), dtype=np.uint8)

def main():
    global frame_placeholder
    cap = cv2.VideoCapture(0)
    tracker = FingerTracker()
    
    # Try to set a common resolution (may fail depending on webcam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Initialize prediction variables
    last_digit = "N/A"
    last_confidence = 0.0

    print("\n--- Controls ---")
    print("Press 's' to START drawing (records trace).")
    print("Press 'e' to END drawing (stops recording).")
    print("Press 'p' to PROCESS the current drawing (prediction).")
    print("Press 'c' to CLEAR the drawing and prediction.")
    print("Press 'q' to QUIT.")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Flip the frame for a more intuitive drawing experience
            frame = cv2.flip(frame, 1)
            frame_placeholder = frame # Update global placeholder

            # Update tracker (find hand position)
            tracker.update(frame)

            # Draw status on the frame
            status_text = "DRAWING" if tracker.is_drawing else "IDLE"
            color = (0, 255, 0) if tracker.is_drawing else (0, 0, 255)
            cv2.putText(frame, f"Status: {status_text}", (10, 450), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            # Draw the recorded trace onto the live frame
            for i in range(1, len(tracker.trace)):
                cv2.line(frame, tracker.trace[i-1], tracker.trace[i], (255, 0, 255), 3) # Magenta line

            # Display prediction
            cv2.putText(frame, f"Pred: {last_digit}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(frame, f"Conf: {last_confidence:.2f}", (10, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                        
            cv2.imshow('Finger Digit Recognizer (Live Feed)', frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s'):
                tracker.is_drawing = True
                print("--- Drawing STARTED ---")
            
            elif key == ord('e'):
                tracker.is_drawing = False
                print("--- Drawing STOPPED ---")

            elif key == ord('c'):
                tracker.clear()
                last_digit = "N/A"
                last_confidence = 0.0
                print("Trace and Prediction cleared!")
                cv2.destroyWindow('Preprocessed Digit') # Close the digit preview window
            
            elif key == ord('p'):
                if tracker.is_drawing:
                    print("Please press 'e' to stop drawing before processing.")
                    continue
                    
                trace_img = tracker.get_trace_image()
                
                if trace_img is not None and np.sum(trace_img) > 1000: # Ensure there is actual drawing
                    print("Processing digit with CUDA/TensorFlow...")
                    # Show the image sent to the model (scaled up for visualization)
                    cv2.imshow('Preprocessed Digit', cv2.resize(trace_img, (280, 280), interpolation=cv2.INTER_NEAREST))

                    # Predict the digit (CUDA-accelerated step)
                    digit, confidence = predict_digit(trace_img)
                    
                    last_digit = str(digit)
                    last_confidence = confidence
                    print(f"Prediction: {last_digit}, Confidence: {last_confidence:.2f}")

                else:
                    print("No significant trace to process!")
                    last_digit = "N/A"
                    last_confidence = 0.0
                    
            elif key == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("\nCleanup complete. Application closed.")

if __name__ == "__main__":
    main()
