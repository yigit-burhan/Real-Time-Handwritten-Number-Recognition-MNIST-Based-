import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

print("=" * 50)
print("✂️  DIGIT SEGMENTATION - Phase 5")
print("=" * 50)

# Load the preprocessed image
image_path = 'preprocessed.png'
print(f"\n📂 Loading: {image_path}")

img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"❌ Error: Could not load {image_path}")
    print("   Run preprocess.py first!")
    exit()

print("✓ Image loaded successfully!")
print(f"   Image shape: {img.shape}")

# --------------------
# METHOD: Contour Detection + Left-to-Right Sorting
# --------------------
print("\n[1/3] Finding contours...")

# Find all contours
contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(f"✓ Found {len(contours)} contours")

if len(contours) == 0:
    print("❌ No contours found! Check your preprocessing.")
    exit()

# --------------------
# Filter and sort contours
# --------------------
print("[2/3] Filtering and sorting contours...")

# Get bounding boxes for all contours
digit_contours = []
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    area = cv2.contourArea(cnt)
    
    # Filter out very small contours (noise)
    # Adjust these thresholds based on your images
    if area > 50 and w > 5 and h > 10:  # Minimum size filters
        digit_contours.append({
            'contour': cnt,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'area': area
        })

print(f"✓ Filtered to {len(digit_contours)} potential digits")

if len(digit_contours) == 0:
    print("❌ No valid digits found after filtering!")
    print("   Try adjusting the filter thresholds in the code.")
    exit()

# Sort left-to-right (by x coordinate)
digit_contours = sorted(digit_contours, key=lambda d: d['x'])

print("✓ Sorted left-to-right")

# --------------------
# Visualize detected regions
# --------------------
print("[3/3] Extracting and saving individual digits...")

# Create visualization
img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

# Create output directory for segmented digits
os.makedirs('segmented_digits', exist_ok=True)

# Extract each digit
segmented_digits = []
for i, d in enumerate(digit_contours):
    x, y, w, h = d['x'], d['y'], d['w'], d['h']
    
    # Draw bounding box on visualization
    cv2.rectangle(img_color, (x, y), (x+w, y+h), (0, 255, 0), 2)
    cv2.putText(img_color, str(i), (x, y-5), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Crop the digit
    digit_img = img[y:y+h, x:x+w]
    
    # Add padding (to make it more square)
    # MNIST digits are centered in 28x28
    max_dim = max(w, h)
    pad_x = (max_dim - w) // 2
    pad_y = (max_dim - h) // 2
    
    # Add extra padding (20% of max dimension)
    extra_pad = int(max_dim * 0.2)
    
    padded_digit = cv2.copyMakeBorder(
        digit_img, 
        pad_y + extra_pad, pad_y + extra_pad,
        pad_x + extra_pad, pad_x + extra_pad,
        cv2.BORDER_CONSTANT, 
        value=0
    )
    
    # Save the segmented digit
    digit_path = f'segmented_digits/digit_{i}.png'
    cv2.imwrite(digit_path, padded_digit)
    
    segmented_digits.append(padded_digit)
    print(f"   ✓ Digit {i}: Saved to {digit_path} (size: {padded_digit.shape})")

# --------------------
# Display results
# --------------------
print("\n📊 Displaying results...")

# Calculate grid size for display
num_digits = len(segmented_digits)
cols = min(5, num_digits)
rows = (num_digits + cols - 1) // cols

fig, axes = plt.subplots(rows + 1, cols, figsize=(15, 4 * (rows + 1)))
fig.suptitle('Digit Segmentation Results', fontsize=16, fontweight='bold')

# Ensure axes is always 2D array
if not isinstance(axes, np.ndarray):
    axes = np.array([[axes]])
elif len(axes.shape) == 1:
    axes = axes.reshape(-1, 1)

# Show full image with bounding boxes (top row, span all columns)
for col in range(cols):
    if col == 0:
        axes[0, col].imshow(cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB))
        axes[0, col].set_title('Detected Digits (Left-to-Right)')
        axes[0, col].axis('off')
    else:
        axes[0, col].axis('off')

# Show individual segmented digits
for i, digit_img in enumerate(segmented_digits):
    row = 1 + (i // cols)
    col = i % cols
    
    axes[row, col].imshow(digit_img, cmap='gray')
    axes[row, col].set_title(f'Digit {i}')
    axes[row, col].axis('off')

# Hide unused subplots
for i in range(num_digits, rows * cols):
    row = 1 + (i // cols)
    col = i % cols
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('segmentation_results.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization as 'segmentation_results.png'")
plt.show()

print("\n" + "=" * 50)
print(f"✅ SEGMENTATION COMPLETE!")
print(f"   Extracted {len(segmented_digits)} digits")
print(f"   Saved in: segmented_digits/")
print("=" * 50)
print("\nNext: We'll normalize and classify each digit")