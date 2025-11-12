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
