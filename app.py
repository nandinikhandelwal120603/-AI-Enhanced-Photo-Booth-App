from flask import Flask, render_template, Response, request, redirect, url_for, session, jsonify, send_file
import cv2
import os
import numpy as np
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'super_secret_key'

camera = cv2.VideoCapture(0)
captured_images = []
def apply_vintage_filter(frame):
    # Convert to float32 for better manipulation
    frame = frame.astype(np.float32) / 255.0

    # 1. Exposure + Brightness + Contrast
    frame = frame * 0.85  # reduce exposure (approx -51)
    frame = frame + 0.05  # slight brightness boost

    # 2. Highlights & Black Point (simulate contrast shift)
    frame = np.clip(frame, 0.03, 0.97)  # simulates black point -100 and highlight -72

    # 3. Saturation + Vibrance (approx)
    hsv = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[...,1] *= 1.25  # Increase saturation
    hsv[...,1] = np.clip(hsv[...,1], 0, 255)
    frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # 4. Warmth tweak (shift towards red/yellow)
    b, g, r = cv2.split(frame)
    r += 0.07
    g += 0.03
    b -= 0.02
    frame = cv2.merge((b, g, r))
    frame = np.clip(frame, 0, 1)

    # 5. Sharpness boost (high-pass filter trick)
    blurred = cv2.GaussianBlur(frame, (0, 0), 3)
    sharp = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)

    # Final clip & convert back
    final = np.clip(sharp * 255, 0, 255).astype(np.uint8)
    return final
def apply_old_digital_film_filter(frame):
    # Convert to float32 for better manipulation
    frame = frame.astype(np.float32) / 255.0

    # 1. Brilliance (reduce overall dynamic range)
    frame = np.clip(frame * 0.8, 0, 1)  # Approx -100 brilliance

    # 2. Contrast Boost
    frame = cv2.addWeighted(frame, 1.22, np.zeros(frame.shape, frame.dtype), 0, 0)  # +22 contrast

    # 3. Highlights & Shadows
    frame = np.clip(frame, 0.05, 0.95)  # Reduce extreme highlights & shadows

    # 4. Black Point
    frame = np.clip(frame - 0.1, 0, 1)  # Simulating black point -100

    # 5. Vibrance Reduction
    hsv = cv2.cvtColor((frame * 255).astype(np.uint8), cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= 0.9  # Reduce vibrance (-10)
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    frame = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR).astype(np.float32) / 255.0

    # 6. Warmth & Tint Adjustment (Shift towards yellow and magenta)
    b, g, r = cv2.split(frame)
    r += 0.06  # Add warmth (+15)
    g += 0.03  
    b -= 0.02  

    r += 0.1  # Add tint (+27, slight magenta shift)
    b -= 0.1  
    frame = cv2.merge((b, g, r))
    frame = np.clip(frame, 0, 1)

    # 7. Definition Boost (Slight sharpening)
    blurred = cv2.GaussianBlur(frame, (0, 0), 2)
    sharp = cv2.addWeighted(frame, 1.1, blurred, -0.1, 0)  # Approx +10 definition

    # 8. Vignette Effect
    rows, cols = frame.shape[:2]
    X_resultant_kernel = cv2.getGaussianKernel(cols, cols / 2)
    Y_resultant_kernel = cv2.getGaussianKernel(rows, rows / 2)
    kernel = Y_resultant_kernel * X_resultant_kernel.T
    mask = kernel / kernel.max()
    
    vignette = np.dstack((mask, mask, mask))  # Apply to all channels
    frame = frame * (1 - 0.1 * vignette)  # Approx +10 vignette

    # Final clip & convert back
    final = np.clip(frame * 255, 0, 255).astype(np.uint8)
    return final

@app.route('/take_picture', methods=['POST'])
def take_picture():
    global captured_images

    ret, frame = camera.read()
    if not ret:
        return jsonify({'status': 'fail'})

    # Get filter once
    filter_type = session.get('filter', 'none')
    print(f"Applying filter: {filter_type}")  # Debugging line

    # Apply the correct filter
    if filter_type == 'bw':
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    elif filter_type == 'vintage':
        frame = apply_vintage_filter(frame)
    elif filter_type == 'old_film':
        frame = apply_old_digital_film_filter(frame)
    elif filter_type == 'sepia':
        sepia = np.array([[0.272, 0.534, 0.131],
                          [0.349, 0.686, 0.168],
                          [0.393, 0.769, 0.189]])
        frame = cv2.transform(frame, sepia)
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    # Save final image
    captured_images.append(frame)
    return jsonify({'status': 'success'})



@app.route('/finalize')
def finalize():
    global captured_images
    layout = session.get('layout', 'horizontal')

    if not captured_images:
        return redirect(url_for('capture'))

    # At least 3 images — repeat last one if not enough
    while len(captured_images) < 3:
        captured_images.append(captured_images[-1])

    # Optional: if > 6, only keep first 6 for layout aesthetics
    if len(captured_images) > 6:
        captured_images = captured_images[:6]

    # Resize all to same height
    resized_images = [cv2.resize(img, (400, 533)) for img in captured_images]

    # Create vertical strip (standard photo booth style)
    final_image = cv2.vconcat(resized_images)

    output_path = os.path.join("static", "final_output.jpg")
    cv2.imwrite(output_path, final_image)
    captured_images = []

    return redirect(url_for('preview'))


@app.route('/preview')
def preview():
    name = session.get('name', 'User')
    return render_template('preview.html', name=name)

@app.route('/video_feed')
def video_feed():
    def generate_frames():
        while True:
            success, frame = camera.read()
            if not success:
                break
            else:
                frame = cv2.resize(frame, (300, 400))
                ret, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')
    
@app.route('/debug')
def debug():
    return jsonify({
        'name': session.get('name'),
        'images_captured': len(captured_images),
        'layout': session.get('layout'),
        'filter': session.get('filter')
    })


@app.route('/capture')
def capture():
    name = session.get('name', 'User')
    return render_template('capture.html', name=name, total=session.get('num_images', 2))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session['name'] = request.form.get('name')
        session['num_images'] = int(request.form.get('num_images'))
        session['filter'] = request.form.get('filter')
        session['layout'] = request.form.get('layout')
        return redirect(url_for('capture'))
    return render_template('index.html')

@app.route('/download')
def download():
    return send_file("static/final_output.jpg", as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

