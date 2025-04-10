import cv2
import time
import os
import numpy as np

def capture_images(num_images=4, layout='horizontal', filter_type='none'):
    camera = cv2.VideoCapture(0)

    captured_images = []
    for i in range(num_images):
        print(f"Capturing image {i+1}/{num_images} in 5 seconds...")
        time.sleep(5)  # 5 second wait before each capture

        ret, frame = camera.read()
        if not ret:
            print("Failed to grab frame")
            continue

        # Resize
        frame = cv2.resize(frame, (300, 400))

        # Apply filter if selected
        if filter_type == 'bw':
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)  # keep 3 channels
        elif filter_type == 'sepia':
            sepia_filter = np.array([[0.272, 0.534, 0.131],
                                     [0.349, 0.686, 0.168],
                                     [0.393, 0.769, 0.189]])
            frame = cv2.transform(frame, sepia_filter)
            frame = np.clip(frame, 0, 255).astype(np.uint8)

        captured_images.append(frame)

    camera.release()

    # Combine images
    if layout == 'horizontal':
        final_image = cv2.hconcat(captured_images)
    else:
        final_image = cv2.vconcat(captured_images)

    # Save final image
    output_path = os.path.join("static", "final_output.jpg")
    cv2.imwrite(output_path, final_image)

    print(f"Saved final image to: {output_path}")
    return output_path
