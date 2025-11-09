import cv2
import numpy as np
import mediapipe as mp
from collections import deque
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

# --- CONFIGURATION & GLOBAL STATE ---
predicted_digit_final = "..."
predicted_confidence_final = "0.0%"
prediction_display_time = 0.0
DISPLAY_DURATION = 3.0 # Show the prediction pop-up for 3.0 seconds

# --- PYTORCH MODEL DEFINITIONS (UNCHANGED) ---

# 1. SVHN Model (Requires 32x32 input)
class SVHNDigitClassifier(nn.Module):
    """High-Capacity 3-Layer CNN for SVHN."""
    def __init__(self):
        super(SVHNDigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1) 
        self.fc_input_size = 256 * 4 * 4
        self.dropout = nn.Dropout(p=0.5) 
        self.fc1 = nn.Linear(self.fc_input_size, 256) 
        self.fc2 = nn.Linear(256, 10)
    def forward(self, x):
        x = F.relu(self.conv1(x)); x = F.max_pool2d(x, 2) 
        x = F.relu(self.conv2(x)); x = F.max_pool2d(x, 2) 
        x = F.relu(self.conv3(x)); x = F.max_pool2d(x, 2)    
        x = x.view(-1, self.fc_input_size); x = self.dropout(x)                
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x)); return x # Fixed here to have F.relu for non-linear activation

# 2. DIGITAL Model (Placeholder for your second model, 28x28 input)
class SimpleDigitClassifier(nn.Module):
    """Placeholder model architecture for 'digital_model.pth'."""
    def __init__(self):
        super(SimpleDigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(-1, 32 * 7 * 7); x = F.relu(self.fc1(x)); x = self.fc2(x); return x

# --- MODEL LOADING AND INFERENCE SETUP (UNCHANGED) ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH_SVHN = 'svhn_digit_model_max_capacity.pth'
MODEL_PATH_SIMPLE = 'digital_model.pth'

model_svhn = SVHNDigitClassifier().to(DEVICE)
model_simple = SimpleDigitClassifier().to(DEVICE)

svhn_normalize = transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))

def load_models():
    global model_svhn, model_simple
    
    try:
        model_svhn.load_state_dict(torch.load(MODEL_PATH_SVHN, map_location=DEVICE))
        model_svhn.eval() 
        print(f"Loaded SVHN Model from: {MODEL_PATH_SVHN}")
    except FileNotFoundError:
        print(f"FATAL ERROR: SVHN Model file '{MODEL_PATH_SVHN}' not found.")
        return False
        
    try:
        model_simple.load_state_dict(torch.load(MODEL_PATH_SIMPLE, map_location=DEVICE))
        model_simple.eval() 
        print(f"Loaded DIGITAL Model from: {MODEL_PATH_SIMPLE}")
    except FileNotFoundError:
        print(f"WARNING: Digital Model file '{MODEL_PATH_SIMPLE}' not found. Prediction for this model will be 'N/A'.")

    return True

def predict_single_model(normalized_trace_array, model_obj, input_size):
    if model_obj.training: 
        return "N/A", 0.0, np.zeros(10)

    rgb_trace_array = np.stack([normalized_trace_array]*3, axis=-1)
    pil_image = Image.fromarray((rgb_trace_array * 255).astype(np.uint8), mode='RGB')
    
    input_tensor = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(), 
        svhn_normalize
    ])(pil_image)
    
    input_batch = input_tensor.unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        output = model_obj(input_batch)
    
    probabilities = F.softmax(output, dim=1)
    confidence, predicted_class_index = torch.max(probabilities, 1)

    return (
        str(predicted_class_index.item()), 
        confidence.item() * 100, 
        output.squeeze(0).cpu().numpy()
    )

def run_dual_prediction(normalized_trace_array):
    global predicted_digit_final, predicted_confidence_final, prediction_display_time
    
    svhn_pred, svhn_conf, svhn_logits = predict_single_model(normalized_trace_array, model_svhn, 32)
    digital_pred, digital_conf, _ = predict_single_model(normalized_trace_array, model_simple, 28)
    
    print("-" * 60)
    print(f"SVHN LOGITS (Final Layer Output Matrix):")
    print(" ".join(f"[{i}:{l:.2f}]" for i, l in enumerate(svhn_logits)))
    print(f"SVHN Prediction: {svhn_pred} ({svhn_conf:.1f}%) | Digital Prediction: {digital_pred} ({digital_conf:.1f}%)")
    print("-" * 60)
    
    predicted_digit_final = svhn_pred
    predicted_confidence_final = f"{svhn_conf:.1f}%"
    
    prediction_display_time = time.time()

# --- FINGER TRACKER CLASS (UNCHANGED) ---
class FingerTracker:
    def __init__(self, trace_buffer=50, smoothing_window=5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
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
        if len(buffer) == 0: return None
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
        two_fingers = (index_extended and middle_extended and ring_folded and pinky_folded and distance < 50) 
        return two_fingers, index_pos, middle_pos
    def process_frame(self, frame):
        h, w = frame.shape[:2]
        if self.drawing_canvas is None: self.drawing_canvas = np.zeros((h, w, 3), dtype=np.uint8)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        output_frame = frame.copy()
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(output_frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS, self.mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2), self.mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2))
                two_fingers, index_pos, middle_pos = self.detect_two_fingers_extended(hand_landmarks, frame.shape)
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
                        cv2.circle(self.drawing_canvas, smoothed_point, 5, (255, 255, 255), -1)
                        cv2.circle(output_frame, index_pos, 8, (255, 0, 255), 2)
                        cv2.circle(output_frame, middle_pos, 8, (255, 0, 255), 2)
                        cv2.line(output_frame, index_pos, middle_pos, (0, 255, 255), 2)
                else: self.is_drawing = False
        if len(self.index_trace) > 1:
            for i in range(1, len(self.index_trace)):
                if self.index_trace[i-1] is None or self.index_trace[i] is None: continue
                thickness = int(np.sqrt(64 / (i + 1)) + 2)
                cv2.line(output_frame, self.index_trace[i-1], self.index_trace[i], (0, 255, 0), thickness)
                cv2.line(self.drawing_canvas, self.index_trace[i-1], self.index_trace[i], (255, 255, 255), 3)
        mask = cv2.cvtColor(self.drawing_canvas, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(mask, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        output_frame = cv2.bitwise_and(output_frame, output_frame, mask=mask_inv)
        canvas_colored = cv2.cvtColor(self.drawing_canvas, cv2.COLOR_BGR2RGB)
        output_frame = cv2.add(output_frame, canvas_colored)
        return output_frame
    def get_drawing_canvas(self): return self.drawing_canvas.copy()
    def clear_trace(self):
        self.index_trace.clear(); self.middle_trace.clear()
        self.index_smooth_buffer.clear(); self.middle_smooth_buffer.clear()
        if self.drawing_canvas is not None: self.drawing_canvas.fill(0)
    def get_trace_image(self, size=(32, 32)):
        if self.drawing_canvas is None: return None
        gray = cv2.cvtColor(self.drawing_canvas, cv2.COLOR_BGR2GRAY)
        coords = cv2.findNonZero(gray)
        if coords is None: return None
        x, y, w, h = cv2.boundingRect(coords)
        padding = 20
        x = max(0, x - padding); y = max(0, y - padding)
        w = min(self.drawing_canvas.shape[1] - x, w + 2*padding)
        h = min(self.drawing_canvas.shape[0] - y, h + 2*padding)
        cropped = gray[y:y+h, x:x+w]
        if w > h:
            diff = w - h; pad_top = diff // 2; pad_bottom = diff - pad_top
            cropped = cv2.copyMakeBorder(cropped, pad_top, pad_bottom, 0, 0, cv2.BORDER_CONSTANT, value=0)
        elif h > w:
            diff = h - w; pad_left = diff // 2; pad_right = diff - pad_left
            cropped = cv2.copyMakeBorder(cropped, 0, 0, pad_left, pad_right, cv2.BORDER_CONSTANT, value=0)
        resized = cv2.resize(cropped, size, interpolation=cv2.INTER_AREA)
        normalized = resized.astype(np.float32) / 255.0
        return normalized
    def close(self): self.hands.close()

# --- CUSTOM UI DRAWING FUNCTION ---
def draw_rounded_rect_with_blur(image, rect_start, rect_end, radius, color, alpha=0.5, blur_strength=5):
    x1, y1 = rect_start
    x2, y2 = rect_end
    
    # Ensure coordinates are within image bounds
    x1 = max(0, x1); y1 = max(0, y1)
    x2 = min(image.shape[1], x2); y2 = min(image.shape[0], y2)

    # Create a mask for the rounded rectangle
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Draw circles for corners
    cv2.circle(mask, (x1 + radius, y1 + radius), radius, 255, -1)
    cv2.circle(mask, (x2 - radius, y1 + radius), radius, 255, -1)
    cv2.circle(mask, (x1 + radius, y2 - radius), radius, 255, -1)
    cv2.circle(mask, (x2 - radius, y2 - radius), radius, 255, -1)

    # Draw rectangles to connect the circles and fill the center
    cv2.rectangle(mask, (x1 + radius, y1), (x2 - radius, y2), 255, -1) # Top/bottom connectors
    cv2.rectangle(mask, (x1, y1 + radius), (x2, y2 - radius), 255, -1) # Left/right connectors

    # Apply blur to the background under the mask
    blurred_background_section = image[y1:y2, x1:x2]
    if blurred_background_section.shape[0] > blur_strength and blurred_background_section.shape[1] > blur_strength:
        blurred_background_section = cv2.GaussianBlur(blurred_background_section, (blur_strength, blur_strength), 0)
    
    # Create the colored overlay (glassy tint)
    color_overlay = np.full(image[y1:y2, x1:x2].shape, color, dtype=np.uint8)

    # Combine blurred background with color overlay using alpha blending
    result_section = cv2.addWeighted(blurred_background_section, 1 - alpha, color_overlay, alpha, 0)
    
    # Apply the mask to the result_section
    mask_cropped = mask[y1:y2, x1:x2]
    # Resize mask_cropped if it doesn't match result_section size (can happen with max/min bounds)
    if mask_cropped.shape[:2] != result_section.shape[:2]:
        mask_cropped = cv2.resize(mask_cropped, (result_section.shape[1], result_section.shape[0]), interpolation=cv2.INTER_LINEAR)

    mask_3_channel = cv2.cvtColor(mask_cropped, cv2.COLOR_GRAY2BGR)
    
    # Use the mask to blend the original background and the new glassy section
    # Create a final composite where the original image parts outside the mask are preserved
    # and the glassy effect is applied inside.
    
    # 1. Extract the region of interest from the original image
    roi = image[y1:y2, x1:x2]

    # 2. Blend the blurred/colored section using the mask
    # We need to make sure mask_cropped is 3-channel for bitwise operations with color images
    
    # Make mask_cropped into a boolean mask for blending
    mask_bool = mask_cropped > 0

    # Ensure result_section has the same dimensions as roi
    if result_section.shape[:2] != roi.shape[:2]:
        result_section = cv2.resize(result_section, (roi.shape[1], roi.shape[0]), interpolation=cv2.INTER_LINEAR)

    # Apply the blended result to the ROI
    roi[mask_bool] = result_section[mask_bool]

    return image


# --- MAIN APPLICATION LOOP ---
def main():
    global predicted_digit_final, predicted_confidence_final, prediction_display_time
    
    if not load_models(): return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    tracker = FingerTracker(trace_buffer=100, smoothing_window=7)
    
    frame_count = 0
    fps_time = time.time()
    fps = 0
    was_drawing = False

    try:
        while True:
            current_time = time.time()
            ret, frame = cap.read()
            if not ret: break

            frame = cv2.flip(frame, 1)
            output_frame = tracker.process_frame(frame)

            # --- AUTO-PREDICTION LOGIC ---
            if was_drawing and not tracker.is_drawing:
                trace_img_normalized = tracker.get_trace_image()
                
                if trace_img_normalized is not None:
                    run_dual_prediction(trace_img_normalized)
                    tracker.clear_trace()
                else:
                    prediction_display_time = 0.0 # Clear pop-up
                    
            was_drawing = tracker.is_drawing

            # FPS calculation
            frame_count += 1
            if frame_count % 10 == 0:
                fps = 10 / (current_time - fps_time)
                fps_time = current_time

            # --- MODERN UI ELEMENTS ---
            
            # 1. Status Indicator (Minimal)
            status = "DRAWING" if tracker.is_drawing else "IDLE"
            status_color = (0, 255, 0) if tracker.is_drawing else (100, 100, 255)
            cv2.putText(output_frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
            
            # 2. FPS/Controls (Bottom Corner)
            cv2.putText(output_frame, f"FPS: {fps:.1f}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
            cv2.putText(output_frame, "Lift fingers to predict. C: Clear | Q: Quit", 
                        (10, output_frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

            # 3. Prediction Pop-up (Top Right Corner) - Now with Glassy Effect
            if current_time < prediction_display_time + DISPLAY_DURATION:
                h, w, _ = output_frame.shape
                
                text_content = f"Prediction: {predicted_digit_final} ({predicted_confidence_final})"
                
                # Sizing for top-right placement
                font_scale = 1.0 # Adjusted for a slightly larger but still compact look
                font_thickness = 2
                padding = 15
                corner_radius = 15 # Radius for rounded corners
                blur_strength = 25 # Higher value for more blur, must be odd (e.g., 15, 25)
                
                (text_w, text_h), baseline = cv2.getTextSize(text_content, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                
                # Calculate Top Right Box Coordinates
                # Ensuring padding on both sides
                rect_start_x = w - text_w - 2 * padding
                rect_start_y = padding // 2
                rect_end_x = w - padding // 2
                rect_end_y = 2 * padding + text_h + baseline
                
                rect_start = (rect_start_x, rect_start_y)
                rect_end = (rect_end_x, rect_end_y)

                # Call the new drawing function for the glassy effect
                output_frame = draw_rounded_rect_with_blur(
                    output_frame, 
                    rect_start, rect_end, 
                    corner_radius, 
                    (0, 0, 0), # Color of the tint (black in this case)
                    alpha=0.6, # Translucency (0.0 - 1.0)
                    blur_strength=blur_strength
                )
                
                # Draw Text (Cyan/Yellow) at the correct position
                text_pos = (rect_start_x + padding, rect_start_y + padding + text_h)
                cv2.putText(output_frame, text_content, text_pos, cv2.FONT_HERSHEY_SIMPLEX, 
                            font_scale, (255, 255, 0), font_thickness, cv2.LINE_AA) 

            cv2.imshow('Finger Trace Recognition', output_frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('c'):
                tracker.clear_trace()
                prediction_display_time = 0.0
                
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        print("Cleanup complete")


if __name__ == "__main__":
    main()
