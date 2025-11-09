import cv2
import numpy as np
import tensorflow as tf
from finger_tracker import FingerTracker
from led_controller import LEDController
import time

# ============ CONFIGURATION ============
# UPDATE THESE WITH YOUR ARDUINO PORTS!
ARDUINO_PORTS = [
    '/dev/ttyACM0',  # Arduino 1 - Input layer
    '/dev/ttyACM1',  # Arduino 2 - Hidden layer 1
    '/dev/ttyACM2',  # Arduino 3 - Hidden layer 2
    '/dev/ttyACM3'   # Arduino 4 - Output layer
]

# ============ LOAD MODEL ============
print("Loading model...")
model = tf.keras.models.load_model('digit_model.h5')

# Create activation model
layer_outputs = [layer.output for layer in model.layers if 'dense' in layer.name.lower()]
activation_model = tf.keras.models.Model(inputs=model.input, outputs=layer_outputs)

# ============ INITIALIZE ============
print("Initializing camera...")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Initializing finger tracker...")
tracker = FingerTracker(trace_buffer=100, smoothing_window=7)

print("Connecting to Arduinos...")
try:
    led_controller = LEDController(ARDUINO_PORTS)
except Exception as e:
    print(f"Warning: Could not connect to Arduinos: {e}")
    print("Running in CAMERA-ONLY mode (no LEDs)")
    led_controller = None

# ============ FUNCTIONS ============
def visualize_neural_network(image):
    """Visualize forward pass on LEDs"""
    if led_controller is None:
        return
    
    activations = activation_model.predict(image, verbose=0)
    led_controller.clear_all_arduinos()
    time.sleep(0.2)
    
    for i, activation in enumerate(activations):
        if i >= len(ARDUINO_PORTS):
            break
        
        layer_activation = activation.flatten()
        
        if len(layer_activation) > 10:
            top_indices = np.argsort(layer_activation)[-10:]
            sampled_activation = layer_activation[top_indices]
        else:
            sampled_activation = layer_activation
        
        led_controller.send_layer_to_arduino(i, sampled_activation, delay=0.5)
    
    time.sleep(2)

# ============ MAIN LOOP ============
print("\n" + "="*60)
print("NEURAL NETWORK DIGIT RECOGNIZER WITH LED VISUALIZATION")
print("="*60)
print("\nControls:")
print("  'p' - Predict digit and visualize on LEDs")
print("  'c' - Clear trace")
print("  'q' - Quit")
print("="*60 + "\n")

frame_count = 0
fps_time = time.time()
fps = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        output_frame = tracker.process_frame(frame)
        
        # Calculate FPS
        frame_count += 1
        if frame_count % 10 == 0:
            fps = 10 / (time.time() - fps_time)
            fps_time = time.time()
        
        # Add UI
        cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        status = "DRAWING" if tracker.is_drawing else "IDLE"
        color = (0, 255, 0) if tracker.is_drawing else (0, 0, 255)
        cv2.putText(output_frame, f"Status: {status}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        cv2.putText(output_frame, "P:Predict | C:Clear | Q:Quit", 
                   (10, output_frame.shape[0] - 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        cv2.imshow('Neural Network Digit Recognizer', output_frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        
        elif key == ord('c'):
            print("Clearing...")
            tracker.clear_trace()
            if led_controller:
                led_controller.clear_all_arduinos()
        
        elif key == ord('p'):
            trace_img = tracker.get_trace_image()
            if trace_img is not None:
                model_input = trace_img.reshape(1, 28, 28, 1)
                
                prediction = model.predict(model_input, verbose=0)
                predicted_digit = np.argmax(prediction)
                confidence = np.max(prediction) * 100
                
                print(f"\n{'='*40}")
                print(f"PREDICTED: {predicted_digit}")
                print(f"CONFIDENCE: {confidence:.1f}%")
                print(f"{'='*40}\n")
                
                # Visualize on LEDs
                print("Visualizing on LEDs...")
                visualize_neural_network(model_input)
                
                # Show preprocessed
                cv2.imshow('Preprocessed Digit', cv2.resize(trace_img, (280, 280)))
            else:
                print("No trace to predict!")

finally:
    cap.release()
    cv2.destroyAllWindows()
    tracker.close()
    if led_controller:
        led_controller.close()
    print("\nShutdown complete!")
