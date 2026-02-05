## Real-Time Handwritten Number Recognition (MNIST-Based)

This project trains a convolutional neural network (CNN) on the MNIST dataset and uses it for **real-time handwritten digit recognition** from a webcam feed. It has three main Python scripts:

- `train_mnist.py`: Trains and evaluates the CNN on MNIST and saves model weights.
- `realtime_recognition.py`: Opens the webcam, detects digits on a sheet of paper, and recognizes them in real time.
- `run.py`: Simple entry point that checks for a trained model and launches real-time recognition.

---

### 1. Project Structure Overview

- **Main folder (root)**
  - Keep these files in the project root:
    - `train_mnist.py`
    - `realtime_recognition.py`
    - `run.py`
    - `README.md`
    - `mnist_cnn_improved.pth` (trained weights)
    - `mnist_cnn.pth` (legacy/older weights, optional fallback)

- **Development folder**
  - `development/`
    Contains extra scripts, datasets, images, outputs, and other development artifacts (for example: `data/`, screenshots, result images, helper scripts, etc.).

- **Model training**
  - `train_mnist.py`  
    Trains an improved CNN on MNIST with data augmentation, tracks performance (especially on visually similar digit pairs), and saves the best model to `mnist_cnn_improved.pth`.

- **Real-time inference**
  - `realtime_recognition.py`  
    Loads the trained model, captures frames from the webcam, detects a white paper region, segments digit candidates, preprocesses them, and performs digit classification in real time with a user-friendly on-screen UI.

- **Launcher / CLI**
  - `run.py`  
    Verifies the presence of trained weights (`mnist_cnn.pth` or `mnist_cnn_improved.pth`) and imports `realtime_recognition.py` to start the app.

---

### 2. `train_mnist.py`: Training & Evaluation Script

**Key responsibilities**

- **Data loading & augmentation**
  - Uses `torchvision.datasets.MNIST` with:
    - Random rotations, affine transforms, scaling, and shear.
    - Random erasing and normalization to improve robustness.
  - Creates `DataLoader`s for train and test sets (`batch_size=64`).

- **Model definition (`ImprovedCNN`)**
  - **Convolutional block (`self.conv`)**:
    - Two stacked conv–batchnorm–ReLU layers → max pooling → dropout.
    - Repeats with increased channels (32 → 64).
  - **Fully connected block (`self.fc`)**:
    - Flattened features → dense layers with batch norm, ReLU, and dropout.
    - Final linear layer outputs logits for 10 classes (digits 0–9).

- **Training setup**
  - **Loss**: `CrossEntropyLoss`.
  - **Optimizer**: `Adam` with learning rate `0.001`.
  - **Scheduler**: `ReduceLROnPlateau` based on training loss.
  - If `mnist_cnn_improved.pth` exists, it tries to load and continue training; otherwise starts fresh.

- **Detailed evaluation (`test_accuracy_detailed`)**
  - Computes **overall test accuracy**.
  - Tracks performance on **similar digit pairs**:
    - `(1, 7), (3, 8), (4, 9), (5, 6), (2, 7)`
  - For each pair, it counts:
    - How many times digits from that pair appear.
    - How many of those instances are correctly classified.
  - Returns both the accuracy and a dictionary with per-pair stats.

- **Training loop**
  - Runs for `num_epochs = 10`.
  - Each epoch:
    - Trains over the entire training set.
    - Evaluates on the test set using `test_accuracy_detailed`.
    - Steps the LR scheduler with the epoch training loss.
    - Prints:
      - Epoch, total loss, overall accuracy.
      - Accuracy for each of the defined similar-digit pairs.
    - **Model checkpointing**:
      - If current accuracy is better than previous `best_accuracy`, saves the model to `mnist_cnn_improved.pth`.

- **Visualization of sample predictions**
  - After training:
    - Evaluates the model on a batch from `test_loader`.
    - Plots first 10 images with:
      - Predicted label (P), true label (T), and confidence.
      - Green title if correct, red if incorrect.
    - Saves to `training_results.png`.
  - Prints where the model is saved and a reminder to update real-time script with the improved model.

---

### 3. `realtime_recognition.py`: Real-Time Webcam Digit Recognition

**High-level flow**

1. **Load model**
2. **Configure preprocessing and real-time settings**
3. **Open webcam and start read–process–display loop**
4. **Detect paper and segment digits**
5. **Preprocess each digit, run CNN, and display results**
6. **Handle keyboard commands (quit, screenshot, reset, toggle lighting)**

#### 3.1. Model Definition & Loading

- **Model class (`CNN`)**
  - Structurally mirrors `ImprovedCNN` from `train_mnist.py`:
    - Two convolutional blocks with conv–batchnorm–ReLU, pooling, dropout.
    - Two fully connected layers with batchnorm, ReLU, dropout, then final output layer.

- **Model loading logic**
  - Tries to load **improved model weights**:
    - If `mnist_cnn_improved.pth` exists:
      - Loads into `CNN` and sets `model_loaded = True`.
  - If that fails, falls back to loading **old model**:
    - Uses inner `OldCNN` definition (simpler architecture).
    - Attempts to load from `mnist_cnn.pth`.
  - If neither is available:
    - Prints an error and exits, instructing the user to train the model first.

- **Model state**
  - Calls `model.eval()` for inference mode (no dropout updates, no gradients).

#### 3.2. Image Preprocessing & Lighting Enhancement

- **Adaptive preprocessing functions**
  - `enhance_brightness_contrast`: simple brightness/contrast adjustment.
  - `auto_adjust_brightness_contrast`:
    - Estimates mean brightness and chooses `(alpha, beta)` parameters adaptively.
  - `apply_clahe`: applies CLAHE (Contrast Limited Adaptive Histogram Equalization).
  - `remove_shadows`: dilates the image, median blurs, subtracts background, then normalizes.
  - `adaptive_preprocess(image)`:
    - Converts to grayscale if necessary.
    - Applies auto brightness, shadow removal, and CLAHE depending on flags:
      - `ENABLE_AUTO_BRIGHTNESS`
      - `ENABLE_CLAHE`
      - `ENABLE_SHADOW_REMOVAL`

- **Transform for CNN input (`transform`)**
  - `transforms.Compose`:
    - Resize to `28x28`.
    - Convert to tensor.
    - Normalize with mean and std `(0.5,)`.

#### 3.3. Digit Segmentation

- **Segmentation helpers**
  - `split_by_vertical_projection(digit_img)`:
    - Uses vertical projection profile and peak finding to split wide blobs into multiple digits.
  - `split_by_contour_analysis(digit_img)`:
    - Uses distance transform and contour detection to find separate components.

- **`advanced_segment_digit(digit_img, aspect_ratio)`**
  - For blobs with **high aspect ratio** (likely multiple digits in a row):
    - First attempts vertical projection splitting.
    - If that fails, tries contour-based splitting.
  - Returns either multiple segments or the original image as a single segment.

#### 3.4. Main Webcam Loop

- **Configuration parameters**
  - **Detection & tracking**
    - `CONFIDENCE_THRESHOLD`: min softmax probability to accept prediction.
    - `PROCESS_EVERY_N_FRAMES`: skip frames for performance.
    - `MIN_STABLE_FRAMES`: number of consecutive consistent frames before locking the result.
    - `HOLD_RESULT_SECONDS`: how long to display a locked result.
  - **Lighting control flags**
    - `ENABLE_AUTO_BRIGHTNESS`, `ENABLE_CLAHE`, `ENABLE_SHADOW_REMOVAL`.

- **Webcam and window setup**
  - Opens default camera (`cv2.VideoCapture(0)`).
  - Sets resolution to `640x480`.
  - Creates a resizable window `Number Recognition` (display size: `960x720`).
  - Prints usage instructions to console.

- **Per-frame processing**
  - Read frame, increment frame counter.
  - If there is a **locked result**:
    - Display “LOCKED” and countdown timer until it expires.
    - While locked and within `HOLD_RESULT_SECONDS`, skip detection.
  - Every `PROCESS_EVERY_N_FRAMES` when not locked:
    - Convert frame to HSV and threshold a white region (paper detection).
    - Clean mask using morphological open/close.
    - Find the largest paper-like contour; if area is big enough, treat as paper.
    - From the paper region:
      - Apply `adaptive_preprocess`.
      - Gaussian blur + adaptive threshold to get binary digits.
      - Morphological cleaning.
      - Find external contours as **digit candidates**.
      - Filter by area, width, height, aspect ratio, and extent to remove noise.
      - Sort candidates left to right.
      - For each candidate:
        - Optionally split into segments via `advanced_segment_digit`.
        - Draw bounding rectangles on display frame.
        - Center each segment in a square, pad with additional margin, and convert to PIL.
        - Convert to tensor via `transform`, run through the CNN.
        - Get class prediction and confidence via `softmax`.
        - If confidence ≥ `CONFIDENCE_THRESHOLD`, add to list and draw predicted digit above the box.
    - Concatenate digit predictions left→right to form a **number string**.
    - Maintain:
      - `last_recognized`
      - `last_confidence`
      - `stable_count`
    - If the current number matches the previous frame, increase `stable_count`. Once it reaches `MIN_STABLE_FRAMES`, **lock** the result (`locked_result`) and timestamp it.

- **Display overlay & UI**
  - Semi-transparent black panel at the top with:
    - Title: “Real-Time Number Recognition”.
    - Key hints: `Q=quit R=reset L=lighting[ON/OFF]`.
  - Shows:
    - Current or locked **recognized number** plus confidence.
    - A large box at the bottom with the number prominently displayed.
    - FPS estimate based on processing time and `PROCESS_EVERY_N_FRAMES`.
  - Screenshot toast area at the bottom when a screenshot is taken.

- **Keyboard controls**
  - `q` / `Q`: quit application.
  - `s` / `S`: save current display frame as `screenshot_<timestamp>.png` and show toast.
  - `r` / `R`: reset recognition state (unlock result, clear history).
  - `l` / `L`: toggle lighting enhancement features on/off (brightness, CLAHE, shadow removal).

---

### 4. `run.py`: Simple Launcher Script

**Purpose**

- Acts as a **user-friendly entry point** for the project so you can run real-time recognition with:

python run.py