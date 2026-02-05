import cv2
import numpy as np
import torch
import torch.nn as nn
import time
import os
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# Configuration
CONFIDENCE_THRESHOLD = 0.5
# Higher = faster (process fewer frames). You can tune this live in code.
PROCESS_EVERY_N_FRAMES = 4
MIN_STABLE_FRAMES = 3
HOLD_RESULT_SECONDS = 3.0
ENABLE_AUTO_BRIGHTNESS = True
ENABLE_CLAHE = True
ENABLE_SHADOW_REMOVAL = True

# Performance tuning
ROI_DOWNSCALE = 0.75  # process the paper region at lower resolution for speed (0.5-1.0)
def split_by_vertical_projection(digit_img):
    h, w = digit_img.shape
    vertical_proj = np.sum(digit_img, axis=0) / 255.0
    smoothed = gaussian_filter1d(vertical_proj, sigma=2)
    
    if np.max(smoothed) > 0:
        smoothed = smoothed / np.max(smoothed)
    
    inverted = 1.0 - smoothed
    peaks, _ = find_peaks(inverted, height=0.3, distance=w//4)
    
    if len(peaks) == 0:
        return [digit_img]
    
    split_points = [0] + list(peaks) + [w]
    segments = []
    
    for i in range(len(split_points) - 1):
        start, end = split_points[i], split_points[i + 1]
        if end - start > 3:
            segments.append(digit_img[:, start:end])
    
    return segments if len(segments) > 1 else [digit_img]


def split_by_contour_analysis(digit_img):
    dist_transform = cv2.distanceTransform(digit_img, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.3 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    
    contours, _ = cv2.findContours(sure_fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) <= 1:
        return [digit_img]
    
    segments = []
    bounding_boxes = [cv2.boundingRect(cnt) for cnt in contours 
                      if cv2.boundingRect(cnt)[2] > 3 and cv2.boundingRect(cnt)[3] > 5]
    bounding_boxes = sorted(bounding_boxes, key=lambda b: b[0])
    
    for x, y, w, h in bounding_boxes:
        x_start = max(0, x - 2)
        x_end = min(digit_img.shape[1], x + w + 2)
        y_start = max(0, y - 2)
        y_end = min(digit_img.shape[0], y + h + 2)
        segments.append(digit_img[y_start:y_end, x_start:x_end])
    
    return segments if len(segments) > 1 else [digit_img]


def advanced_segment_digit(digit_img, aspect_ratio):
    if aspect_ratio > 1.5:
        segments = split_by_vertical_projection(digit_img)
        if len(segments) > 1:
            return segments
        
        segments = split_by_contour_analysis(digit_img)
        if len(segments) > 1:
            return segments
    
    return [digit_img]
def enhance_brightness_contrast(image, alpha=1.0, beta=0):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def auto_adjust_brightness_contrast(image):
    mean_brightness = np.mean(image)
    
    if mean_brightness < 50:
        alpha, beta = 1.5, 50
    elif mean_brightness < 100:
        alpha, beta = 1.3, 30
    elif mean_brightness > 200:
        alpha, beta = 1.2, -30
    elif mean_brightness > 150:
        alpha, beta = 1.1, -10
    else:
        alpha, beta = 1.0, 0
    
    return enhance_brightness_contrast(image, alpha, beta)

def apply_clahe(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(image)

def remove_shadows(image):
    dilated = cv2.dilate(image, np.ones((7, 7), np.uint8))
    bg_img = cv2.medianBlur(dilated, 21)
    diff_img = 255 - cv2.absdiff(image, bg_img)
    return cv2.normalize(diff_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

def adaptive_preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    
    if ENABLE_AUTO_BRIGHTNESS:
        gray = auto_adjust_brightness_contrast(gray)
    if ENABLE_SHADOW_REMOVAL:
        gray = remove_shadows(gray)
    if ENABLE_CLAHE:
        gray = apply_clahe(gray)
    
    return gray

def _put_text_with_outline(img, text, org, font, scale, color, thickness, outline_color=(0, 0, 0), outline_thickness=None):
    if outline_thickness is None:
        outline_thickness = max(2, thickness + 2)
    cv2.putText(img, text, org, font, scale, outline_color, outline_thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)

def _digit_to_tensor(binary_digit_img):
    """
    Convert a binary digit image (0/255, white=foreground) to a normalized torch tensor [1, 28, 28].
    Matches training normalization: (x - 0.5) / 0.5 == 2x - 1.
    """
    # Resize to 28x28 (OpenCV is much faster than PIL+torchvision)
    resized = cv2.resize(binary_digit_img, (28, 28), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    x = (x * 2.0) - 1.0
    return torch.from_numpy(x).unsqueeze(0)  # [1,28,28]

def _draw_top_hud(
    frame,
    *,
    title,
    hint_line,
    number_text=None,
    number_color=(0, 255, 0),
    status_right=None,
    bar_alpha=0.55,
):
    """
    Draw a modern top HUD bar similar to the reference screenshot:
    - Large title (left)
    - Hint line (left, under title)
    - Recognized number line (left, bigger + colored)
    - Right-aligned status + FPS
    """
    h, w = frame.shape[:2]
    pad_x = 18
    pad_top = 10
    bar_h = 105
    x0, y0 = 0, 0
    x1, y1 = w, min(h, bar_h)

    overlay = frame.copy()
    # Warm brown-ish HUD like the reference image
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (35, 55, 95), -1)
    cv2.addWeighted(overlay, bar_alpha, frame, 1 - bar_alpha, 0, frame)
    # subtle bottom border
    cv2.line(frame, (0, y1 - 1), (w, y1 - 1), (0, 0, 0), 2)

    # Left: title + hints
    _put_text_with_outline(frame, title, (pad_x, pad_top + 34), cv2.FONT_HERSHEY_SIMPLEX, 1.10, (245, 245, 245), 3)
    cv2.putText(frame, hint_line, (pad_x, pad_top + 66), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (230, 230, 230), 2, cv2.LINE_AA)

    # Left: number line (bigger + colored)
    if number_text:
        _put_text_with_outline(frame, number_text, (pad_x, pad_top + 102), cv2.FONT_HERSHEY_SIMPLEX, 1.00, number_color, 3)

    # Right side: status
    right_x = w - pad_x
    y = pad_top + 30
    if status_right:
        (tw, th), _ = cv2.getTextSize(status_right, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 3)
        _put_text_with_outline(frame, status_right, (right_x - tw, y), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 255, 255), 3)
        y += 30

def _draw_footer_fps(frame, text):
    """Bottom-right footer text for FPS/timers (avoids overlapping the top HUD)."""
    h, w = frame.shape[:2]
    pad = 12
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    x = max(pad, w - pad - tw)
    y = max(th + pad, h - pad)
    _put_text_with_outline(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

# CNN Model
class CNN(nn.Module):
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

# Load model
print("Loading model...")
model = CNN()
model_loaded = False

# Device / inference optimizations
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.set_num_threads(max(1, os.cpu_count() or 1))

if os.path.exists("mnist_cnn_improved.pth"):
    try:
        model.load_state_dict(torch.load("mnist_cnn_improved.pth", map_location=DEVICE))
        print("Loaded improved model")
        model_loaded = True
    except Exception as e:
        print(f"Could not load improved model: {e}")

if not model_loaded and os.path.exists("mnist_cnn.pth"):
    try:
        class OldCNN(nn.Module):
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
        
        old_model = OldCNN()
        old_model.load_state_dict(torch.load("mnist_cnn.pth", map_location=DEVICE))
        model = old_model
        print("Loaded old model")
        model_loaded = True
    except Exception as e:
        print(f"Could not load old model: {e}")

if not model_loaded:
    print("Error: No model found. Run train_mnist_improved.py first!")
    exit()

model.to(DEVICE)
model.eval()

# --------------------
# Configuration (tune these for SPEED vs QUALITY)
# --------------------
CONFIDENCE_THRESHOLD = 0.5   # Minimum confidence to accept prediction
PROCESS_EVERY_N_FRAMES = 4   # Process every 4th frame -> MUCH FASTER, still responsive
MIN_STABLE_FRAMES = 3        # Frames needed before locking result
HOLD_RESULT_SECONDS = 3.0    # Hold recognized number for 3 seconds
    
# Lighting adaptation settings (turn ON for better visibility)
ENABLE_AUTO_BRIGHTNESS = True     # Auto adjust brightness/contrast (cheap, keep on)
ENABLE_CLAHE = True               # Contrast Limited Adaptive Histogram Equalization
ENABLE_SHADOW_REMOVAL = True      # Remove shadows from paper

# --------------------
# Start webcam
# --------------------
print("\n[2/2] Starting webcam...")
print("\n" + "=" * 60)
print("📸 INSTRUCTIONS:")
print("  • Hold a white paper with handwritten numbers")
print("  • Works in different lighting conditions!")
print("  • Press 'q' to QUIT")
print("  • Press 's' to SCREENSHOT")
print("  • Press 'r' to RESET")
print("  • Press 'l' to toggle LIGHTING ENHANCEMENT")
print("=" * 60 + "\n")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Error: Could not open webcam!")
    exit()

# Bump capture resolution for clearer digits (higher cost than 320x240)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Make the display window larger while keeping capture fast
WINDOW_NAME = 'Number Recognition'
cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, 960, 720)

print("✓ Webcam started - Show white paper with numbers!\n")

# State variables
frame_count = 0
last_recognized = ""
last_confidence = 0.0
stable_count = 0
processing_time = 0
locked_result = None        # Locked result (number, confidence, timestamp)
lock_time = 0               # When the result was locked
# UI timing
ema_frame_dt = 1.0 / 30.0   # Smoothed frame time for FPS / "next scan" estimation
# Screenshot toast state
last_screenshot_path = None
last_screenshot_time = 0.0
SCREENSHOT_TOAST_SECONDS = 1.2

while True:
    loop_start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        break
    
    display = frame.copy()
    frame_count += 1
    
    # Check locked result
    current_time = time.time()
    if locked_result is not None:
        time_since_lock = current_time - lock_time
        if time_since_lock < HOLD_RESULT_SECONDS:
            # Lock status is displayed in the top HUD
            remaining = HOLD_RESULT_SECONDS - time_since_lock
        else:
            locked_result = None
            lock_time = 0
            stable_count = 0
            last_recognized = ""
            print("Unlocked - ready for new number")
    
    if locked_result is None and frame_count % PROCESS_EVERY_N_FRAMES == 0:
        start_time = time.time()
        
        # Detect white paper
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 50, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        kernel = np.ones((5, 5), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        paper_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        paper_area = None
        if len(paper_contours) > 0:
            largest_paper = max(paper_contours, key=cv2.contourArea)
            if cv2.contourArea(largest_paper) > 5000:
                px, py, pw, ph = cv2.boundingRect(largest_paper)
                paper_area = (px, py, pw, ph)
                cv2.rectangle(display, (px, py), (px+pw, py+ph), (255, 0, 0), 2)
                cv2.putText(display, "Paper", (px, py - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        if paper_area is None:
            processing_time = time.time() - start_time
            stable_count = 0
            last_recognized = ""
        else:
            px, py, pw, ph = paper_area
            paper_region = frame[py:py+ph, px:px+pw]

            # Process a downscaled paper ROI for speed; later map boxes back to display coords.
            scale = float(ROI_DOWNSCALE)
            if scale < 1.0:
                small_w = max(1, int(paper_region.shape[1] * scale))
                small_h = max(1, int(paper_region.shape[0] * scale))
                paper_proc = cv2.resize(paper_region, (small_w, small_h), interpolation=cv2.INTER_AREA)
            else:
                paper_proc = paper_region
                scale = 1.0

            gray = adaptive_preprocess(paper_proc)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            thresh = cv2.adaptiveThreshold(
                blurred,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2,
            )

            kernel = np.ones((3, 3), np.uint8)
            cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
            cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)

            all_contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours
            digit_candidates = []
            for cnt in all_contours:
                x, y, w, h = cv2.boundingRect(cnt)
                area = cv2.contourArea(cnt)
                aspect_ratio = float(w) / h if h > 0 else 0
                extent = float(area) / (w * h) if w * h > 0 else 0
                
                if (area > 50 and w > 5 and h > 10) or (h > 20 and w > 3):
                    if aspect_ratio <= 8 and not (aspect_ratio < 0.1 and w < 10):
                        if extent >= 0.15 and w <= gray.shape[1] * 0.5 and h <= gray.shape[0] * 0.6:
                            digit_candidates.append({
                                'x': x, 'y': y, 'w': w, 'h': h,
                                'area': area, 'aspect_ratio': aspect_ratio
                            })
            
            digit_candidates = sorted(digit_candidates, key=lambda d: d['x'])
            
            all_segments = []
            for d in digit_candidates:
                x, y, w, h = d['x'], d['y'], d['w'], d['h']
                digit_img = cleaned[y:y+h, x:x+w]
                segments = advanced_segment_digit(digit_img, d['aspect_ratio'])
                
                for i, segment in enumerate(segments):
                    seg_h, seg_w = segment.shape
                    segment_x = x + (i * w // len(segments)) if len(segments) > 1 else x
                    all_segments.append({
                        'image': segment,
                        'x_display': int((segment_x / scale) + px),
                        'y_display': int((y / scale) + py),
                        'w': int(seg_w / scale), 'h': int(seg_h / scale)
                    })
            
            # Batch all segments into a single model call (much faster than per-digit inference)
            seg_tensors = []
            for seg in all_segments:
                cv2.rectangle(
                    display,
                    (seg['x_display'], seg['y_display']),
                    (seg['x_display'] + seg['w'], seg['y_display'] + seg['h']),
                    (0, 255, 0),
                    2,
                )

                digit_img = seg['image']
                max_dim = max(seg['w'], seg['h'])
                pad_x = (max_dim - seg['w']) // 2
                pad_y = (max_dim - seg['h']) // 2
                extra_pad = int(max_dim * 0.2)

                padded = cv2.copyMakeBorder(
                    digit_img,
                    pad_y + extra_pad,
                    pad_y + extra_pad,
                    pad_x + extra_pad,
                    pad_x + extra_pad,
                    cv2.BORDER_CONSTANT,
                    value=0,
                )

                seg_tensors.append(_digit_to_tensor(padded))

            predictions = []
            confidences = []

            if len(seg_tensors) > 0:
                batch = torch.stack(seg_tensors, dim=0).to(DEVICE)  # [N,1,28,28]
                with torch.inference_mode():
                    output = model(batch)
                    probs = torch.softmax(output, dim=1)
                    pred_idx = probs.argmax(dim=1)
                    pred_conf = probs.gather(1, pred_idx.unsqueeze(1)).squeeze(1)

                pred_idx = pred_idx.detach().cpu().numpy().tolist()
                pred_conf = pred_conf.detach().cpu().numpy().tolist()

                for seg, p, c in zip(all_segments, pred_idx, pred_conf):
                    if c >= CONFIDENCE_THRESHOLD:
                        predictions.append(int(p))
                        confidences.append(float(c))
                        _put_text_with_outline(
                            display,
                            str(int(p)),
                            (seg['x_display'], max(0, seg['y_display'] - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.95,
                            (0, 255, 0),
                            2,
                        )
            
            if len(predictions) > 0:
                current_number = ''.join(map(str, predictions))
                avg_conf = np.mean(confidences)
                
                if current_number == last_recognized:
                    stable_count += 1
                    if stable_count >= MIN_STABLE_FRAMES:
                        locked_result = {'number': current_number, 'confidence': avg_conf}
                        lock_time = time.time()
                        print(f"Locked: {current_number} ({avg_conf*100:.1f}%)")
                else:
                    stable_count = 1
                    last_recognized = current_number
                    last_confidence = avg_conf
            else:
                stable_count = 0
            
            processing_time = time.time() - start_time
    
    # Smooth FPS based on overall loop time (stable and matches what user sees)
    loop_dt = max(1e-6, time.time() - loop_start_time)
    ema_frame_dt = (0.90 * ema_frame_dt) + (0.10 * loop_dt)
    fps_ui = 1.0 / max(1e-6, ema_frame_dt)

    # Screenshot toast
    if last_screenshot_path and (time.time() - last_screenshot_time) < SCREENSHOT_TOAST_SECONDS:
        toast = f"Saved: {os.path.basename(last_screenshot_path)}"
        cv2.rectangle(display, (10, display.shape[0] - 45), (display.shape[1] - 10, display.shape[0] - 10), (0, 0, 0), -1)
        cv2.putText(display, toast, (20, display.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    
    # Display result
    display_number = None
    display_confidence = 0
    
    if locked_result:
        display_number = locked_result['number']
        display_confidence = locked_result['confidence']
    elif stable_count >= MIN_STABLE_FRAMES and last_recognized:
        display_number = last_recognized
        display_confidence = last_confidence
    
    if display_number:
        text_size = cv2.getTextSize(display_number, cv2.FONT_HERSHEY_SIMPLEX, 3, 4)[0]
        text_x = (display.shape[1] - text_size[0]) // 2
        text_y = display.shape[0] - 50
        
        color = (0, 255, 255) if locked_result else (0, 255, 0)
        cv2.rectangle(display, (text_x - 20, text_y - text_size[1] - 20),
                     (text_x + text_size[0] + 20, text_y + 20), color, -1)
        cv2.putText(display, display_number, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 4)

    # Top HUD (match the reference screenshot style)
    lighting_status = "ON" if (ENABLE_AUTO_BRIGHTNESS and ENABLE_CLAHE and ENABLE_SHADOW_REMOVAL) else "OFF"
    hint = f"White paper | Q:quit  R:reset  L:lighting[{lighting_status}]"

    number_line = None
    if display_number:
        number_line = f"Number: {display_number} ({display_confidence*100:.0f}%)"

    # Right-aligned status + timer
    status_right = None
    if locked_result is not None:
        status_right = "RESULT LOCKED"
        next_scan_text = f"Unlock in: {max(0.0, HOLD_RESULT_SECONDS - (time.time() - lock_time)):.1f}s"
    else:
        # Estimate time until the next processing pass
        frames_until = (PROCESS_EVERY_N_FRAMES - (frame_count % PROCESS_EVERY_N_FRAMES)) % PROCESS_EVERY_N_FRAMES
        next_scan_secs = frames_until * ema_frame_dt
        next_scan_text = f"Next scan in: {next_scan_secs:.1f}s"

    _draw_top_hud(
        display,
        title="Real-Time Number Recognition",
        hint_line=hint,
        number_text=number_line,
        number_color=(0, 255, 0),
        status_right=status_right,
    )

    # Footer FPS/timer (moved here to avoid overlapping the top HUD)
    _draw_footer_fps(display, f"FPS: {fps_ui:.1f}  |  {next_scan_text}")
    
    cv2.imshow(WINDOW_NAME, display)
    
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == ord('Q'):
        break
    elif key == ord('s') or key == ord('S'):
        last_screenshot_path = f'screenshot_{int(time.time())}.png'
        cv2.imwrite(last_screenshot_path, display)
        last_screenshot_time = time.time()
        print(f"Screenshot saved: {last_screenshot_path}")
    elif key == ord('r') or key == ord('R'):
        locked_result = None
        lock_time = 0
        stable_count = 0
        last_recognized = ""
        print("Reset")
    elif key == ord('l') or key == ord('L'):
        ENABLE_AUTO_BRIGHTNESS = not ENABLE_AUTO_BRIGHTNESS
        ENABLE_CLAHE = not ENABLE_CLAHE
        ENABLE_SHADOW_REMOVAL = not ENABLE_SHADOW_REMOVAL
        status = "ON" if ENABLE_AUTO_BRIGHTNESS else "OFF"
        print(f"Lighting: {status}")

cap.release()
cv2.destroyAllWindows()
print("Done")