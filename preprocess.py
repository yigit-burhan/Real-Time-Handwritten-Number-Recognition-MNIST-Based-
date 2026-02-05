import cv2
import numpy as np
import matplotlib.pyplot as plt

print("=" * 50)
print("🔧 IMAGE PREPROCESSING - Phase 4")
print("=" * 50)

# Load the captured image
image_path = 'captured_1.png'  # Change this to test different captures
print(f"\n📂 Loading: {image_path}")

img = cv2.imread(image_path)

if img is None:
    print(f"❌ Error: Could not load {image_path}")
    print("   Make sure you captured an image first!")
    exit()

print("✓ Image loaded successfully!")

# Store original for comparison
original = img.copy()

# --------------------
# STEP 1: Convert to Grayscale
# --------------------
print("\n[1/5] Converting to grayscale...")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# --------------------
# STEP 2: Apply Gaussian Blur (reduce noise)
# --------------------
print("[2/5] Applying Gaussian blur...")
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# --------------------
# STEP 3: Apply Binary Threshold
# --------------------
print("[3/5] Applying binary threshold...")
# Use adaptive threshold for better results with varying lighting
thresh = cv2.adaptiveThreshold(
    blurred, 255, 
    cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    cv2.THRESH_BINARY_INV,  # INV = inverts (white becomes black)
    11, 2
)

# Alternative: Simple threshold (uncomment to try)
# _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# --------------------
# STEP 4: Morphological operations (clean up)
# --------------------
print("[4/5] Cleaning up with morphology...")
kernel = np.ones((3, 3), np.uint8)
# Remove small noise
cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
# Close small gaps
cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

# --------------------
# STEP 5: Find and crop the number area
# --------------------
print("[5/5] Detecting number area...")

# Find contours
contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

if len(contours) == 0:
    print("⚠️  Warning: No contours found!")
    print("   Try adjusting lighting or threshold settings.")
else:
    print(f"✓ Found {len(contours)} contours")
    
    # Find the largest contour (assume it's the number)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    print(f"   Bounding box: x={x}, y={y}, w={w}, h={h}")
    
    # Draw bounding box on original for visualization
    img_with_box = original.copy()
    cv2.rectangle(img_with_box, (x, y), (x+w, y+h), (0, 255, 0), 2)
    
    # Crop the number region
    cropped = cleaned[y:y+h, x:x+w]
    
    # Add padding (10% on each side)
    pad_x = int(w * 0.1)
    pad_y = int(h * 0.1)
    padded = cv2.copyMakeBorder(cropped, pad_y, pad_y, pad_x, pad_x, 
                                cv2.BORDER_CONSTANT, value=0)
    
    print(f"   Cropped size: {cropped.shape}")
    print(f"   Padded size: {padded.shape}")

# --------------------
# Display all steps
# --------------------
print("\n📊 Displaying results...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle('Image Preprocessing Pipeline', fontsize=16, fontweight='bold')

# Row 1
axes[0, 0].imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title('1. Original Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(gray, cmap='gray')
axes[0, 1].set_title('2. Grayscale')
axes[0, 1].axis('off')

axes[0, 2].imshow(blurred, cmap='gray')
axes[0, 2].set_title('3. Blurred')
axes[0, 2].axis('off')

# Row 2
axes[1, 0].imshow(thresh, cmap='gray')
axes[1, 0].set_title('4. Threshold (Inverted)')
axes[1, 0].axis('off')

axes[1, 1].imshow(cleaned, cmap='gray')
axes[1, 1].set_title('5. Cleaned')
axes[1, 1].axis('off')

if len(contours) > 0:
    axes[1, 2].imshow(cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB))
    axes[1, 2].set_title('6. Detected Area')
    axes[1, 2].axis('off')
else:
    axes[1, 2].text(0.5, 0.5, 'No contours found', 
                    ha='center', va='center', fontsize=12)
    axes[1, 2].axis('off')

plt.tight_layout()
plt.savefig('preprocessing_steps.png', dpi=150, bbox_inches='tight')
print("✓ Saved visualization as 'preprocessing_steps.png'")
plt.show()

# Save the preprocessed result
if len(contours) > 0:
    cv2.imwrite('preprocessed.png', padded)
    print("✓ Saved preprocessed image as 'preprocessed.png'")

print("\n" + "=" * 50)
print("✅ PREPROCESSING COMPLETE!")
print("=" * 50)
print("\nNext: We'll segment this into individual digits")