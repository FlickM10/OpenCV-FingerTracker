import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import datasets, transforms
import os

# --- HYPERPARAMETERS ---
BATCH_SIZE = 2048
NUM_EPOCHS = 15  # Extended epochs for deeper model convergence
LEARNING_RATE = 0.001
DATA_ROOT = './data/svhn' # Ensure this path is correct for your SVHN data
MODEL_PATH = 'svhn_digit_model_max_capacity.pth' 

# Use GPU if available, otherwise CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- MODEL DEFINITION (WIDER ARCHITECTURE + DROPOUT) ---
class DigitClassifier(nn.Module):
    """
    High-Capacity 3-Layer CNN for SVHN. Uses increased filter counts (Wider Network).
    """
    def __init__(self):
        super(DigitClassifier, self).__init__()
        # Increased filter counts (Wider): 32 -> 64 -> 128 -> 256
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1) # Third Convolutional Layer

        # Image size reduction: 32 -> 16 -> 8 -> 4
        # Flattened size: 256 channels * 4 * 4 = 4096
        self.fc_input_size = 256 * 4 * 4
        
        self.dropout = nn.Dropout(p=0.5) 
        self.fc1 = nn.Linear(self.fc_input_size, 256) 
        self.fc2 = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2) 
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2) 
        x = F.relu(self.conv3(x)) 
        x = F.max_pool2d(x, 2)    
        
        x = x.view(-1, self.fc_input_size) # Flatten
        x = self.dropout(x)                
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- TRAINING FUNCTION ---
def train_model():
    print(f"--- Starting Training (High Capacity) on {DEVICE} ---")
    
    # SVHN Normalization Constants
    svhn_normalize = transforms.Normalize(
        (0.4377, 0.4438, 0.4728), 
        (0.1980, 0.2010, 0.1970)  
    )

    # Data Augmentation (Applied during training only)
    train_transforms = transforms.Compose([
        # FIX: Removed transforms.ToPILImage() which caused the TypeError, 
        # as SVHN already returns a PIL Image.
        transforms.RandomRotation(5),          
        transforms.ColorJitter(brightness=0.1, contrast=0.1), 
        transforms.RandomPerspective(distortion_scale=0.1, p=0.5), 
        transforms.ToTensor(),                 
        svhn_normalize
    ])

    # Test set gets no augmentation, only normalization
    test_transforms = transforms.Compose([
        transforms.ToTensor(), 
        svhn_normalize
    ])

    # 1. Load SVHN Data
    train_dataset_main = datasets.SVHN(
        root=DATA_ROOT, split='train', download=True, 
        transform=train_transforms
    )
    train_dataset_extra = datasets.SVHN(
        root=DATA_ROOT, split='extra', download=True, 
        transform=train_transforms
    )
    valid_dataset = datasets.SVHN(
        root=DATA_ROOT, split='test', download=True, 
        transform=test_transforms
    )

    # 2. Combine Main and Extra Training Sets
    full_train_dataset = ConcatDataset([train_dataset_main, train_dataset_extra])

    # 3. Create DataLoaders
    train_loader = DataLoader(full_train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4) 
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # 4. Initialize Model, Loss, and Optimizer
    model = DigitClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 5. Training Loop
    best_accuracy = 0.0
    for epoch in range(1, NUM_EPOCHS + 1):
        # --- Training Phase ---
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # SVHN has a '10' label which corresponds to '0'. Remap it.
            labels[labels == 10] = 0

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(full_train_dataset)
        
        # --- Validation Phase ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                labels[labels == 10] = 0 # Remap '10' to '0'
                
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        epoch_accuracy = correct / total
        print(f'Epoch {epoch}/{NUM_EPOCHS} | Loss: {epoch_loss:.4f} | Validation Accuracy: {epoch_accuracy:.4f}')

        # 6. Save the Best Model
        if epoch_accuracy > best_accuracy:
            best_accuracy = epoch_accuracy
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Saved new best model with accuracy: {best_accuracy:.4f}")

    print("--- Training complete. ---")
    print(f"Final Model saved to: {MODEL_PATH}")
    print(f"Best validation accuracy achieved: {best_accuracy:.4f}")

if __name__ == '__main__':
    train_model()
