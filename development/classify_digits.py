import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

print("=" * 50)
print("🤖 DIGIT CLASSIFICATION - Phase 6, 7 & 8")
print("=" * 50)

# --------------------
# Load the trained model
# --------------------
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

print("\n📦 Loading trained model...")
model = CNN()
model.load_state_dict(torch.load("mnist_cnn.pth"))
model.eval()
print("✓ Model loaded successfully!")

# --------------------
# Preprocessing transform (MNIST-style)
# --------------------
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# --------------------
# Load all segmented digits
# --------------------
segmented_dir = 'segmented_digits'
print(f"\n📂 Loading segmented digits from '{segmented_dir}/'...")

if not os.path.exists(segmented_dir):
    print(f"❌ Error: Directory '{segmented_dir}' not found!")
    print("   Run segment_digits.py first!")
    exit()

# Get all digit files and sort them
digit_files = sorted([f for f in os.listdir(segmented_dir) if f.startswith('digit_')])

if len(digit_files) == 0:
    print(f"❌ Error: No digit files found in '{segmented_dir}'!")
    exit()

print(f"✓ Found {len(digit_files)} digits")

# --------------------
# Classify each digit
# --------------------
print("\n🔍 Classifying digits...")

predictions = []
confidences = []
processed_images = []

for i, filename in enumerate(digit_files):
    filepath = os.path.join(segmented_dir, filename)
    
    # Load image
    img_cv = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    img_pil = Image.fromarray(img_cv)
    
    # Apply transforms
    img_tensor = transform(img_pil)
    img_batch = img_tensor.unsqueeze(0)  # Add batch dimension
    
    # Predict
    with torch.no_grad():
        output = model(img_batch)
        probabilities = torch.softmax(output, dim=1)[0]
        prediction = output.argmax(1).item()
        confidence = probabilities[prediction].item()
    
    predictions.append(prediction)
    confidences.append(confidence)
    processed_images.append(img_tensor.squeeze().numpy())
    
    print(f"   Digit {i}: Predicted = {prediction} (confidence: {confidence*100:.1f}%)")

# --------------------
# Reconstruct the full number
# --------------------
print("\n🔢 Reconstructing full number...")

recognized_number = ''.join(map(str, predictions))

print("\n" + "=" * 50)
print(f"✅ FINAL RESULT: {recognized_number}")
print("=" * 50)

# Calculate average confidence
avg_confidence = np.mean(confidences) * 100
print(f"   Average confidence: {avg_confidence:.1f}%")

# --------------------
# Visualize results
# --------------------
print("\n📊 Displaying results...")

num_digits = len(predictions)
cols = min(5, num_digits)
rows = (num_digits + cols - 1) // cols

fig, axes = plt.subplots(rows, cols, figsize=(12, 3 * rows))
fig.suptitle(f'Classification Results: {recognized_number}', 
             fontsize=20, fontweight='bold')

# Handle single digit case
if num_digits == 1:
    axes = np.array([axes])

# Flatten axes if multiple rows
if rows > 1:
    axes = axes.flatten()

for i in range(num_digits):
    # Show the normalized 28x28 digit
    axes[i].imshow(processed_images[i], cmap='gray')
    axes[i].set_title(f'Digit {i}\nPredicted: {predictions[i]}\n'
                      f'Confidence: {confidences[i]*100:.1f}%',
                      fontsize=10)
    axes[i].axis('off')

# Hide unused subplots
for i in range(num_digits, len(axes)):
    axes[i].axis('off')

plt.tight_layout()
plt.savefig('classification_results.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization as 'classification_results.png'")
plt.show()

print("\n" + "=" * 50)
print("🎉 RECOGNITION COMPLETE!")
print("=" * 50)
print(f"\nRecognized Number: {recognized_number}")
print(f"Total Digits: {len(predictions)}")
print(f"Average Confidence: {avg_confidence:.1f}%")