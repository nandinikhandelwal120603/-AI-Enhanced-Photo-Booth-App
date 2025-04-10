# 📸 Vintage Vibes – AI Photo Booth Web App

Create retro, classy, or artsy photos in-browser using your webcam and a sprinkle of OpenCV magic 🎩✨

## 🎯 Features

- Real-time camera feed in browser
- Classic filters: Black & White, Sepia, Vintage, Old Money
- Animated countdown & flash effect
- Horizontal/Vertical layout
- Downloadable final image strip

## 🛠️ Tech Stack

- Python + Flask
- OpenCV (image capture + filtering)
- HTML/CSS/JS for UI
- Jinja2 for rendering
- NumPy for processing

## 🚀 Run It Locally

```bash
pip install flask opencv-python numpy
python app.py
```
# Step 1: Clone the repo
git clone https://github.com/yourusername/vintage-photo-booth
cd vintage-photo-booth

# Step 2: Install dependencies
pip install flask opencv-python numpy

# Step 3: Run the app
python app.py

    A[User Input (index.html)] --> B[Capture UI (capture.html)]
    B --> C[POST /take_picture]
    C --> D[Apply Filter (OpenCV)]
    D --> E[Append Frame to List]
    E --> F[If done → /finalize]
    F --> G[Stitch Image (horizontal/vertical)]
    G --> H[Show /preview with final strip]
