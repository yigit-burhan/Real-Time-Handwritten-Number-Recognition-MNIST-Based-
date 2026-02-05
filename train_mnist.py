import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os

print("Training improved MNIST model...")

# Data augmentation
train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=5),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_data = datasets.MNIST(root='./data', train=True, download=True, transform=train_transform)
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=test_transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64)

print(f"Training samples: {len(train_data)}")
print(f"Test samples: {len(test_data)}")

# Model
class ImprovedCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.25),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 7 * 7, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = ImprovedCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5)

if os.path.exists("mnist_cnn_improved.pth"):
    try:
        model.load_state_dict(torch.load("mnist_cnn_improved.pth"))
        print("Loaded existing model, continuing training...")
    except:
        print("Could not load model, starting fresh...")
else:
    print("Starting fresh training...")

def test_accuracy_detailed(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    
    similar_pairs = {(1, 7): [0, 0], (3, 8): [0, 0], (4, 9): [0, 0], (5, 6): [0, 0], (2, 7): [0, 0]}

    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            for pred, label in zip(preds, labels):
                pred_val = pred.item()
                label_val = label.item()
                
                for (d1, d2), counts in similar_pairs.items():
                    if label_val == d1 or label_val == d2:
                        similar_pairs[(d1, d2)][1] += 1
                        if pred_val == label_val:
                            similar_pairs[(d1, d2)][0] += 1

    model.train()
    return correct / total, similar_pairs

print("\nTraining...")
num_epochs = 10
best_accuracy = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    acc, similar_pairs = test_accuracy_detailed(model, test_loader)
    scheduler.step(running_loss)
    
    print(f"\nEpoch {epoch+1}/{num_epochs} - Loss: {running_loss:.2f} | Accuracy: {acc*100:.2f}%")
    
    for (d1, d2), (correct, total) in similar_pairs.items():
        if total > 0:
            print(f"  {d1}/{d2}: {(correct/total)*100:.1f}%")
    
    if acc > best_accuracy:
        best_accuracy = acc
        torch.save(model.state_dict(), "mnist_cnn_improved.pth")
        print(f"  Saved new best model ({acc*100:.2f}%)")

print(f"\nTraining complete! Best accuracy: {best_accuracy*100:.2f}%")

model.eval()
images, labels = next(iter(test_loader))
outputs = model(images)
preds = outputs.argmax(1)
probs = torch.softmax(outputs, dim=1)

plt.figure(figsize=(12,5))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(images[i][0], cmap='gray')
    
    pred_val = preds[i].item()
    true_val = labels[i].item()
    confidence = probs[i][pred_val].item() * 100
    
    color = 'green' if pred_val == true_val else 'red'
    plt.title(f"P:{pred_val} T:{true_val}\n{confidence:.0f}%", color=color, fontsize=10)
    plt.axis('off')

plt.tight_layout()
plt.savefig('training_results.png', dpi=150)
print("Saved sample predictions to 'training_results.png'")
plt.show()

print(f"\nModel saved as: mnist_cnn_improved.pth")
print("Update realtime_recognition.py to use the improved model")