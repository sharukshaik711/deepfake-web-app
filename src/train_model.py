import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
train_dir = "../processed/train"
model_save_path = "../models/shahid_model.pth"
os.makedirs(os.path.dirname(model_save_path), exist_ok=True)

# Transforms
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Load dataset
print("Loading dataset...")
dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
print(f"Loaded {len(dataset)} images.")

# Balance dataset
def balance_dataset(dataset):
    print("Balancing dataset...")
    class_indices = {0: [], 1: []}
    for i, (_, label) in enumerate(dataset):
        class_indices[label].append(i)
    min_class_size = min(len(class_indices[0]), len(class_indices[1]))
    balanced_indices = class_indices[0][:min_class_size] + class_indices[1][:min_class_size]
    random.shuffle(balanced_indices)
    return Subset(dataset, balanced_indices)

balanced_dataset = balance_dataset(dataset)
train_loader = DataLoader(balanced_dataset, batch_size=32, shuffle=True)

# Load EfficientNet-B0 with updated method
weights = EfficientNet_B0_Weights.IMAGENET1K_V1
model = efficientnet_b0(weights=weights)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop
num_epochs = 30
patience = 5
best_val_loss = float('inf')
counter = 0

train_losses = []
train_accuracies = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        percentage = (correct / total) * 100
        pbar.set_postfix(loss=loss.item(), acc=f"{percentage:.2f}%")

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)

    print(f"\nEpoch [{epoch+1}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # Early stopping and model saving
    if epoch_loss < best_val_loss:
        best_val_loss = epoch_loss
        torch.save(model.state_dict(), model_save_path)
        counter = 0
        print(f"Model saved at epoch {epoch+1}.")
    else:
        counter += 1
        if counter >= patience:
            print("Early stopping triggered.")
            break

# Plot training metrics
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(train_accuracies, label="Train Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()

plt.tight_layout()
plt.savefig("training_metrics.png")
plt.show()

# Final accuracy bar
plt.figure()
plt.bar(["Train Acc."], [train_accuracies[-1]*100], color="skyblue")
plt.title("Final Accuracy")
plt.ylabel("Accuracy (%)")
plt.text(0, train_accuracies[-1]*100 + 1, f"{train_accuracies[-1]*100:.2f}%", ha='center')
plt.savefig("final_accuracy_bar.png")
plt.show()