import cv2
import numpy as np

print("=" * 50)
print("📸 WEBCAM CAPTURE - Phase 3")
print("=" * 50)
print("\nInstructions:")
print("  • Write a number on white paper with BLACK marker")
print("  • Hold it up to the webcam")
print("  • Press 'c' to CAPTURE")
print("  • Press 'q' to QUIT")
print("\n" + "=" * 50 + "\n")

# Initialize webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam!")
    print("   Check if your webcam is connected.")
    exit()

# Set resolution (optional - adjust if needed)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("✓ Webcam opened successfully!")
print("  Waiting for you to capture an image...\n")

captured = False
capture_count = 0

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    
    if not ret:
        print("❌ Error: Failed to grab frame")
        break
    
    # Create a copy for display with instructions
    display_frame = frame.copy()
    
    # Add text overlay with instructions
    cv2.putText(display_frame, "Press 'C' to Capture | 'Q' to Quit", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    if captured:
        cv2.putText(display_frame, "IMAGE CAPTURED!", 
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Show the frame
    cv2.imshow('Webcam - Handwritten Number Recognition', display_frame)
    
    # Wait for key press (1ms)
    key = cv2.waitKey(1) & 0xFF
    
    # Debug: print key code if any key is pressed
    if key != 255:  # 255 means no key pressed
        print(f"Key pressed: {key} (char: {chr(key) if key < 128 else 'N/A'})")
    
    # Capture image on 'c' key
    if key == ord('c') or key == ord('C'):
        capture_count += 1
        filename = f'captured_{capture_count}.png'
        cv2.imwrite(filename, frame)
        print(f"✓ Image captured and saved as '{filename}'")
        captured = True
        
        # Show captured message for 1 second
        for _ in range(10):
            cv2.imshow('Webcam - Handwritten Number Recognition', display_frame)
            cv2.waitKey(100)
        
        captured = False
    
    # Quit on 'q' key
    elif key == ord('q') or key == ord('Q'):
        print("\n👋 Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

print("\n" + "=" * 50)
print(f"📊 Total captures: {capture_count}")
print("=" * 50)
print("\n✓ Webcam closed successfully!")