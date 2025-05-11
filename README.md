# Deepfake Video Preprocessing

This project demonstrates various preprocessing techniques for enhancing deepfake video analysis. These techniques focus on isolating facial regions, examining image artifacts, and analyzing temporal inconsistencies.

---

## 🧪 Preprocessing Techniques

### 1. Frame Extraction

**What it does:**  
Extracts frames from a video at fixed intervals (e.g., every 30th frame).

**Why it's useful:**  
- Reduces video to analyzable static images.
- Captures variation across time (expressions, lighting, etc.).
- Enables targeted processing.

**Sample Output:**  
Images saved to `frames/<video_name>/frame => <count>.jpg`

---

### 2. Face Detection & Cropping

**What it does:**  
Detects faces in each frame using Haar cascades and crops them out.

**Why it's useful:**  
- Focuses processing on face regions where deepfakes usually apply changes.
- Standardizes input size and removes background clutter.

**Tuned Parameters:**  
- `scaleFactor=1.05`
- `minNeighbors=8`
- `minSize=(95, 95)`

**Sample Output:**  
Images saved to `faces/<video_name>/frame => <count>.jpg`

---

### 3. Temporal Feature Comparison

**What it does:**  
Compares two face crops from different frames to analyze how features change over time.

**Why it's useful:**  
- Reveals unnatural changes or inconsistencies that are common in deepfakes.
- Useful for detecting jitter or lack of temporal coherence.

**Technique Used:**  
Pixel-wise difference with `cv.absdiff()` after resizing faces to same dimensions.

---

### 4. Blurring Detection

**What it does:**  
Applies Gaussian blur to highlight over-smoothed areas.

**Why it's useful:**  
- Deepfakes often use blurring to hide stitching edges or blending artifacts.

---

### 5. Histogram Analysis

**What it does:**  
Plots RGB color histograms for a given image.

**Why it's useful:**  
- Reveals unnatural color distributions often introduced during generation.

---

### 6. Canny Edge Detection

**What it does:**  
Extracts edges using the Canny method.

**Why it's useful:**  
- Highlights unnatural or overly sharp boundaries introduced by face-swapping techniques.

---

### 7. Color Channel Splitting

**What it does:**  
Separates the Blue, Green, and Red channels for individual analysis.

**Why it's useful:**  
- May reveal inconsistencies hidden in individual color channels, often present in deepfakes.

---

## 🛠 Requirements

- Python 3.7+
- OpenCV (`opencv-python`)
- Matplotlib

---

## 📂 Project Structure

```
deepfake-video-preprocessing/
├── videos/              # Raw video files
├── frames/              # Extracted video frames
├── faces/               # Cropped face images
├── main.py              # All preprocessing code
├── haarcascade_frontalface_default.xml
└── README.md
```

---

## 👨‍💻 Author

Shantanu Tapole  
Deepfake Preprocessing Project (2025)