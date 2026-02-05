# run.py
import os
import sys

def main():
    # Check if model exists
    if not os.path.exists("mnist_cnn.pth") and not os.path.exists("mnist_cnn_improved.pth"):
        print("Error: Trained model weights not found!")
        print("Please ensure mnist_cnn.pth is in the directory.")
        return

    print("Launching Real-Time Recognition...")
    try:
        import realtime_recognition
    except ImportError as e:
        print(f"Error: Missing dependencies. {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()