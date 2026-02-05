import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import os

print("=" * 60)
print("🎯 HANDWRITTEN NUMBER RECOGNITION - COMPLETE PIPELINE")
print("=" * 60)

# --------------------
# CNN Model Definition
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

# --------------------
# Load trained model
# --------------------
print("\n[1/5] Loading trained model...")
model = CNN()
try:
    model.load_state_dict(torch.load("mnist_cnn.pth"))
    model.eval()
    print("      ✓ Model loaded successfully!")
except:
    print("      ❌ Error: Could not load mnist_cnn.pth")
    print("         Make sure you trained the model first!")
    exit()

# --------------------
# STEP 1: Webcam Capture
# --------------------
print("\n[2/5] Starting webcam capture...")
print("      Instructions:")
print("      • Write a number on WHITE paper with BLACK marker")
print("      • Hold it up to the webcam")
print("      • Press 'c' to CAPTURE")
print("      • Press 'q' to QUIT\n")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("      ❌ Error: Could not open webcam!")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

captured = False
captured_frame = None

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("      ❌ Error: Failed to grab frame")
        break
    
    display_frame = frame.copy()
    cv2.putText(display_frame, "Press 'C' to Capture | 'Q' to Quit", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if captured:
        cv2.putText(display_frame, "IMAGE CAPTURED!", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('Webcam - Press C to Capture', display_frame)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('c') or key == ord('C'):
        captured_frame = frame.copy()
        cv2.imwrite('temp_captured.png', captured_frame)
        print("      ✓ Image captured!")
        
        for _ in range(10):
            cv2.imshow('Webcam - Press C to Capture', display_frame)
            cv2.waitKey(100)
        
        break
    
    elif key == ord('q') or key == ord('Q'):
        print("      Exiting...")
        cap.release()
        cv2.destroyAllWindows()
        exit()

cap.release()
cv2.destroyAllWindows()

if captured_frame is None:
    print("      ❌ No image captured!")
    exit()

# --------------------
# STEP 2: Preprocessing
# --------------------
print("\n[3/5] Preprocessing image...")

img = captured_frame
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                cv2.THRESH_BINARY_INV, 11, 2)

kernel = np.ones((3, 3), np.uint8)
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

# Find ALL contours (don't crop to bounding box yet)
all_contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(all_contours) == 0:
    print("      ❌ No contours found!")
    exit()

# Filter contours to find potential digits
digit_candidates = []
for cnt in all_contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    
    # Calculate aspect ratio
    aspect_ratio = float(w) / h if h > 0 else 0
    
    # Calculate extent (ratio of contour area to bounding box area)
    rect_area = w * h
    extent = float(area) / rect_area if rect_area > 0 else 0
    
    # More lenient filters for thin digits like "1"
    # Basic size filters
    if (area > 30 and w > 3 and h > 8) or (h > 15 and w > 2):
        # Reject if too horizontal (like paper edges)
        if aspect_ratio > 5:  # Too wide compared to height
            continue
        
        # Reject if too vertical and thin (like paper edges)
        if aspect_ratio < 0.1 and w < 10:  # Too narrow
            continue
        
        # Reject if extent is too low (hollow or irregular shape)
        if extent < 0.15:  # Less than 15% filled
            continue
        
        # Reject if too large (probably background or paper)
        if w > gray.shape[1] * 0.3 or h > gray.shape[0] * 0.5:
            continue
        
        digit_candidates.append({
            'contour': cnt,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'area': area,
            'aspect_ratio': aspect_ratio,
            'extent': extent
        })

if len(digit_candidates) == 0:
    print("      ❌ No valid contours found!")
    exit()

# Sort by x-coordinate (left to right)
digit_candidates = sorted(digit_candidates, key=lambda d: d['x'])

# Get the overall bounding box for visualization
all_x = [d['x'] for d in digit_candidates]
all_y = [d['y'] for d in digit_candidates]
all_x2 = [d['x'] + d['w'] for d in digit_candidates]
all_y2 = [d['y'] + d['h'] for d in digit_candidates]

min_x, max_x = min(all_x), max(all_x2)
min_y, max_y = min(all_y), max(all_y2)

# Add padding to overall region
pad_x = int((max_x - min_x) * 0.05)
pad_y = int((max_y - min_y) * 0.05)
min_x = max(0, min_x - pad_x)
min_y = max(0, min_y - pad_y)
max_x = min(cleaned.shape[1], max_x + pad_x)
max_y = min(cleaned.shape[0], max_y + pad_y)

preprocessed = cleaned[min_y:max_y, min_x:max_x]

print(f"      ✓ Found {len(digit_candidates)} potential digits")
print(f"      ✓ Preprocessed (detected region: {max_x-min_x}x{max_y-min_y})")

# --------------------
# STEP 3: Segmentation (Use already detected contours)
# --------------------
print("\n[4/5] Segmenting digits...")

# Adjust coordinates to the cropped preprocessed image
segmented_digits = []
for d in digit_candidates:
    # Original coordinates
    x, y, w, h = d['x'], d['y'], d['w'], d['h']
    
    # Adjust to preprocessed image coordinates
    x_adj = x - min_x
    y_adj = y - min_y
    
    # Extract digit from cleaned image
    digit_img = cleaned[y:y+h, x:x+w]
    
    # Add padding to make it more square
    max_dim = max(w, h)
    pad_x = (max_dim - w) // 2
    pad_y = (max_dim - h) // 2
    extra_pad = int(max_dim * 0.2)
    
    padded_digit = cv2.copyMakeBorder(digit_img, 
                                       pad_y + extra_pad, pad_y + extra_pad,
                                       pad_x + extra_pad, pad_x + extra_pad,
                                       cv2.BORDER_CONSTANT, value=0)
    segmented_digits.append(padded_digit)

print(f"      ✓ Segmented {len(segmented_digits)} digits")

if len(segmented_digits) == 0:
    print("      ❌ No digits segmented!")
    exit()

# --------------------
# STEP 4: Classification
# --------------------
print("\n[5/5] Classifying digits...")

transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

predictions = []
confidences = []
processed_images = []
CONFIDENCE_THRESHOLD = 0.5  # Minimum 50% confidence

for i, digit_img in enumerate(segmented_digits):
    img_pil = Image.fromarray(digit_img)
    img_tensor = transform(img_pil)
    img_batch = img_tensor.unsqueeze(0)
    
    with torch.no_grad():
        output = model(img_batch)
        probabilities = torch.softmax(output, dim=1)[0]
        prediction = output.argmax(1).item()
        confidence = probabilities[prediction].item()
    
    # Only accept predictions above confidence threshold
    if confidence >= CONFIDENCE_THRESHOLD:
        predictions.append(prediction)
        confidences.append(confidence)
        processed_images.append(img_tensor.squeeze().numpy())
        print(f"      Digit {len(predictions)-1}: {prediction} (confidence: {confidence*100:.1f}%)")
    else:
        print(f"      ⚠️  Digit {i}: {prediction} (confidence: {confidence*100:.1f}%) - REJECTED (too low)")

if len(predictions) == 0:
    print("\n      ❌ No digits with sufficient confidence!")
    exit()

# --------------------
# FINAL RESULT
# --------------------
recognized_number = ''.join(map(str, predictions))
avg_confidence = np.mean(confidences) * 100

print("\n" + "=" * 60)
print(f"🎉 RECOGNIZED NUMBER: {recognized_number}")
print("=" * 60)
print(f"   Total Digits: {len(predictions)}")
print(f"   Average Confidence: {avg_confidence:.1f}%")
print("=" * 60)

# --------------------
# Visualization
# --------------------
print("\n📊 Displaying results...")

fig = plt.figure(figsize=(15, 8))
gs = fig.add_gridspec(3, max(len(segmented_digits), 3), hspace=0.4, wspace=0.3)

# Row 1: Original captured image
ax1 = fig.add_subplot(gs[0, :])
ax1.imshow(cv2.cvtColor(captured_frame, cv2.COLOR_BGR2RGB))
ax1.set_title('Original Captured Image', fontsize=14, fontweight='bold')
ax1.axis('off')

# Row 2: Preprocessed with bounding boxes
ax2 = fig.add_subplot(gs[1, :])
img_with_boxes = cv2.cvtColor(preprocessed, cv2.COLOR_GRAY2BGR)
for d in digit_candidates:
    # Adjust coordinates to preprocessed image
    x, y, w, h = d['x'], d['y'], d['w'], d['h']
    x_adj = x - min_x
    y_adj = y - min_y
    cv2.rectangle(img_with_boxes, (x_adj, y_adj), (x_adj+w, y_adj+h), (0, 255, 0), 2)
ax2.imshow(cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB))
ax2.set_title('Detected Digits', fontsize=14, fontweight='bold')
ax2.axis('off')

# Row 3: Individual classified digits
for i in range(len(predictions)):
    ax = fig.add_subplot(gs[2, i])
    ax.imshow(processed_images[i], cmap='gray')
    ax.set_title(f'{predictions[i]}\n{confidences[i]*100:.0f}%', 
                 fontsize=12, fontweight='bold')
    ax.axis('off')

# Add final result as title
fig.suptitle(f'FINAL RESULT: {recognized_number}', 
             fontsize=20, fontweight='bold', color='green')

plt.tight_layout()
plt.savefig('final_result.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization as 'final_result.png'")
plt.show()

# Cleanup temp file
if os.path.exists('temp_captured.png'):
    os.remove('temp_captured.png')

print("\n✅ RECOGNITION COMPLETE!")