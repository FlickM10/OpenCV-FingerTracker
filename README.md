# 👆 Real-Time Finger Tracker & PyTorch SVHN Classifier
Advanced ML Backend: Utilizes a PyTorch CNN trained on over 600,000 real-world SVHN images for classification, offering superior generalization compared to standard MNIST models.

Real-time Interaction: Leverages OpenCV to track a designated marker's position (e.g., a colored cap or finger) in a live video feed.
Drawing & Preprocessing: The drawn path is captured, resized, and preprocessed (e.g., grayscale, normalization) to match the 3×32×32 input requirements of the SVHN model.

Instant Inference: Predicts the classified digit (0-9) immediately after the user signals the end of the drawing.

# 🛠️ Prerequisites
You need Python 3.x and the following libraries. The PyTorch installation requires careful attention to your CUDA version if you plan to use a GPU.

#
    1. Create and activate a virtual environment
    
    python -m venv venv
    source venv/bin/activate  # Linux/macOS
    .\venv\Scripts\activate # Windows

    2. Install core dependencies
    
    pip install opencv-python numpy

    3. Install PyTorch (Choose the command based on your system/CUDA version)
    
    Example for CPU-only:
    pip install torch torchvision
    
    Example for CUDA 12.1 (check PyTorch website for the exact command):
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    4. Save your requirements (Good practice)
    pip freeze > requirements.txt

# 🚀 Getting Started
1. PyTorch Model Setup

The system requires a trained PyTorch model.

Training: Run the training script to generate the model checkpoint:

    python3 model_trainer_svhn.py

Placement: Ensure your final trained model file (e.g., svhn_cnn_model.pth) is located in the root directory. The main tracker script will attempt to load this model.

2. Run the Main Application

Start the real-time finger tracking and classification application:

    python finger_tracker_main.py

3. Usage Instructions

       Start: The camera window will open and begin tracking the designated color/marker.

       Press 'S': Save and Classify. This captures the drawn path, runs it through the PyTorch model, and displays the prediction.

       Press 'C': Clear. Clears the drawn path from the current frame.

       Press 'Q': Quit. Closes the application windows.

# Project Structure

finger_tracker_main.py:    Main script with OpenCV tracking, drawing, and prediction logic.

model_trainer_svhn.py: Script for loading SVHN data, defining, and training the CNN.

svhn_cnn_model.pth: Trained PyTorch model weights (the checkpoint file).

utils/cnn_architecture.py: Defines the PyTorch CNN class.

README.md: This documentation file.

requirements.txt: List of required dependencies.
