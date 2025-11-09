import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, UnidentifiedImageError
import os

# --- CRITICAL CONFIGURATION (MUST MATCH TRAINING SCRIPT) ---
# IMPORTANT: This path must point to the folder containing the validation set
DATA_ROOT = '/home/avaneesh/neural_net_visualizer/kaggle_augmented_data' 
KAGGLE_VALID_DIR = os.path.join(DATA_ROOT, "MNIST Validation Set (4k)")

MODEL_PATH = 'digit_model.pth' 
# Loading onto CPU to test cross-platform compatibility
DEVICE = torch.device("cpu") 

# --- Custom Loader and Robust Class for Validation ---

# Custom loader function
def pil_loader_rgb(path):
    """Loads image and converts to RGB before any transformations."""
    with open(path, 'rb') as f:
        # We explicitly convert to RGB to handle P/RGBA modes gracefully
        img = Image.open(f).convert('RGB') 
        return img

class RobustImageFolder(datasets.ImageFolder):
    """
    Custom ImageFolder that filters out corrupted or unidentifiable images
    upon initialization to prevent DataLoader crashes.
    """
    def __init__(self, root, transform=None):
        # Initialize the standard ImageFolder structure first
        super().__init__(root, transform=transform, loader=pil_loader_rgb)
        
        valid_samples = []
        corrupted_count = 0
        total_samples = len(self.samples)

        print(f"--- Running pre-check on Validation Set ({total_samples} files) ---")
        
        # Filter out invalid/corrupted samples
        for path, target in self.samples:
            try:
                # Attempt to load the image immediately to check for corruption
                pil_loader_rgb(path)
                valid_samples.append((path, target))
            except UnidentifiedImageError:
                corrupted_count += 1
            except Exception:
                corrupted_count += 1

        if corrupted_count > 0:
            print(f"✅ Filtered {corrupted_count} corrupted or invalid files from validation set.")
            self.samples = valid_samples
            self.imgs = self.samples
        else:
            print("✅ No corrupted files found in validation set.")


# --- Model Architecture (Must be identical to the training model) ---
class DigitClassifier(nn.Module):
    def __init__(self):
        super(DigitClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 64 * 7 * 7) # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Validation Logic ---
def validate_model():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at '{MODEL_PATH}'.")
        print("Please ensure your training script ran successfully and created 'digit_model.pth'.")
        return

    # 1. Setup Data Loader
    data_transforms = transforms.Compose([
        # These transforms must match the training transforms
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    try:
        # Use the Robust class to load and filter the validation set
        valid_dataset = RobustImageFolder(KAGGLE_VALID_DIR, transform=data_transforms)
        valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
    except Exception as e:
        print(f"Error setting up validation data: {e}")
        return

    # 2. Initialize Model and Load Weights
    model = DigitClassifier().to(DEVICE)
    
    # Load state dict, mapping trained weights to the CPU
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval() # Set model to evaluation mode

    # 3. Perform Evaluation
    correct = 0
    total = 0
    
    print(f"Starting validation on {len(valid_dataset)} clean images...")

    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(DEVICE), target.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    accuracy = 100 * correct / total
    
    print("\n--- Validation Results ---")
    print(f"Total Images Tested: {total}")
    print(f"Correct Predictions: {correct}")
    print(f"Final Model Accuracy: {accuracy:.4f}%")
    print("--------------------------\n")
    
if __name__ == '__main__':
    validate_model()
