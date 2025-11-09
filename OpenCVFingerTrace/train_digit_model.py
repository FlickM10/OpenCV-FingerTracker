import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, UnidentifiedImageError
import os

# --- CRITICAL CONFIGURATION ---
# IMPORTANT: This path must be correct:
DATA_ROOT = '/home/avaneesh/neural_net_visualizer/kaggle_augmented_data' 
KAGGLE_TRAIN_DIR = os.path.join(DATA_ROOT, "Augmented MNIST Training Set (400k)")
KAGGLE_VALID_DIR = os.path.join(DATA_ROOT, "MNIST Validation Set (4k)")

if not os.path.exists(KAGGLE_TRAIN_DIR):
    print(f"ERROR: Training directory not found at {KAGGLE_TRAIN_DIR}")
    
# --- 1. Robust Dataset Class for Pre-Filtering ---

# Custom loader function
def pil_loader_rgb(path):
    """Loads image and converts to RGB before any transformations."""
    with open(path, 'rb') as f:
        img = Image.open(f).convert('RGB')
        return img

class RobustImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        # Initialize the standard ImageFolder
        super().__init__(root, transform=transform, loader=pil_loader_rgb)
        
        # Now, filter out invalid/corrupted samples
        valid_samples = []
        corrupted_count = 0
        total_samples = len(self.samples)

        print(f"--- Running pre-check on {root.split('/')[-1]} ({total_samples} files) ---")
        
        for path, target in self.samples:
            try:
                # Attempt to open and load the file immediately
                pil_loader_rgb(path)
                valid_samples.append((path, target))
            except UnidentifiedImageError:
                # If PIL can't identify the file (corruption, wrong format), skip it.
                corrupted_count += 1
                # print(f"Skipped corrupted file: {path}") # Uncomment to see every skipped file
            except Exception:
                # Catch general I/O errors or other unexpected issues
                corrupted_count += 1

        if corrupted_count > 0:
            print(f"✅ Filtered {corrupted_count} corrupted or invalid files.")
            self.samples = valid_samples
            # Update the paths/targets tuple list in the parent class
            self.imgs = self.samples
        else:
            print("✅ No corrupted files found.")

# --- 2. Model Architecture (Identical) ---
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
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- 3. Training Setup ---
def train_model():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
        print("WARNING: CUDA not available. Falling back to CPU.")
        
    print(f"Training device selected: {device}")
    
    # Define transforms
    data_transforms = transforms.Compose([
        transforms.Grayscale(num_output_channels=1), 
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # Load the datasets using the RobustImageFolder
    train_dataset = RobustImageFolder(KAGGLE_TRAIN_DIR, transform=data_transforms)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)
    
    valid_dataset = RobustImageFolder(KAGGLE_VALID_DIR, transform=data_transforms)
    valid_loader = DataLoader(valid_dataset, batch_size=128, shuffle=False)
    
    print(f"Loaded {len(train_dataset)} clean training images and {len(valid_dataset)} clean validation images.")
    
    model = DigitClassifier().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    NUM_EPOCHS = 3 
    
    print("Starting PyTorch model training...")

    # --- Training Loop ---
    model.train() 
    for epoch in range(NUM_EPOCHS):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            
            # The DataLoader should now only feed valid tensors, so no try/except needed here.
            data, target = data.to(device), target.to(device) 
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            if batch_idx % 1000 == 999:
                print(f'Epoch {epoch+1}/{NUM_EPOCHS}, Batch {batch_idx+1}: Loss: {running_loss/1000:.4f}')
                running_loss = 0.0
        
        evaluate_model(model, valid_loader, device)
        print(f'*** Epoch {epoch+1} complete. ***')


    print("Training finished.")
    
    # --- 4. Save the Model Weights ---
    MODEL_PATH = 'digit_model.pth'
    torch.save(model.state_dict(), MODEL_PATH) 
    print(f"\n✅ Model weights successfully saved to {MODEL_PATH}")

# --- Validation Helper ---
def evaluate_model(model, loader, device):
    """Evaluates the model on the validation set."""
    model.eval() 
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
            
    accuracy = 100 * correct / total
    print(f'Validation Accuracy: {accuracy:.2f}%')
    model.train()

if __name__ == '__main__':
    train_model()
