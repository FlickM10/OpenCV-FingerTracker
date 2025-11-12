# 👆 Real-Time Finger Tracker & PyTorch SVHN Classifier
Advanced ML Backend: Utilizes a PyTorch CNN trained on over 600,000 real-world SVHN images for classification, offering superior generalization compared to standard MNIST models.

Real-time Interaction: Leverages OpenCV to track a designated marker's position (e.g., a colored cap or finger) in a live video feed.
Drawing & Preprocessing: The drawn path is captured, resized, and preprocessed (e.g., grayscale, normalization) to match the 3×32×32 input requirements of the SVHN model.

Instant Inference: Predicts the classified digit (0-9) immediately after the user signals the end of the drawing.

# 🛠️ Prerequisites
