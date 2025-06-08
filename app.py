from flask import Flask, request, render_template, send_file
import cv2
import numpy as np
import mediapipe as mp
from ultralytics import YOLO
import os
import uuid
import io
from PIL import Image

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'static/outputs'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Load YOLOv11 models
gender_model = YOLO('models/yolov11_gender.pt')  # Replace with your model path
glasses_model = YOLO('models/yolov11_glasses.pt')  # Replace with your model path

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return render_template('index.html', error="No file uploaded")
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error="No file selected")
        if file:
            # Save uploaded file
            filename = f"{uuid.uuid4().hex}.jpg"
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            file.save(file_path)

            # Load and process image
            image = cv2.imread(file_path)
            if image is None:
                os.remove(file_path)
                return render_template('index.html', error=f"Failed to load image from {file_path}")
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            height, width, _ = image_rgb.shape

            # Gender Detection
            gender_results = gender_model(image_rgb)
            gender_predictions = gender_results[0].boxes
            gender_status = "No Gender Detected"
            gender_box = None
            if len(gender_predictions) > 0:
                class_names = gender_results[0].names
                detected_class = class_names[int(gender_predictions.cls[0].item())]
                gender_status = detected_class
                gender_box = gender_predictions.xyxy[0].cpu().numpy()

            # Glasses Detection
            glasses_results = glasses_model(image_rgb)
            glasses_predictions = glasses_results[0].boxes
            glasses_status = "No Glasses"
            glasses_box = None
            if len(glasses_predictions) > 0:
                class_names = glasses_results[0].names
                detected_class = class_names[int(glasses_predictions.cls[0].item())]
                glasses_status = detected_class
                glasses_box = glasses_predictions.xyxy[0].cpu().numpy()

            # Shirt Color Detection
            results = pose.process(image_rgb)
            color_name = "Unknown"
            dominant_color = np.array([0, 0, 0])
            shirt_box = None

            if results.pose_landmarks:
                landmarks = results.pose_landmarks.landmark
                left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * width),
                                int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * height))
                right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * width),
                                int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * height))
                left_hip = (int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].x * width),
                            int(landmarks[mp_pose.PoseLandmark.LEFT_HIP].y * height))
                right_hip = (int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].x * width),
                            int(landmarks[mp_pose.PoseLandmark.RIGHT_HIP].y * height))

                # Crop Shirt Region
                x_min = min(left_shoulder[0], right_shoulder[0])
                x_max = max(left_shoulder[0], right_shoulder[0])
                y_min = min(left_shoulder[1], right_shoulder[1])
                y_max = max(left_hip[1], right_hip[1])

                # Add padding
                padding = 30
                x_min = max(0, x_min - padding)
                x_max = min(width, x_max + padding)
                y_min = max(0, y_min - padding)
                y_max = min(height, y_max + padding)
                shirt_box = [x_min, y_min, x_max, y_max]

                shirt_region = image_rgb[y_min:y_max, x_min:x_max]

                if shirt_region.size > 0:
                    # Convert to HSV
                    shirt_region_hsv = cv2.cvtColor(shirt_region, cv2.COLOR_RGB2HSV)

                    # Filter out skin tones
                    lower_skin = np.array([0, 10, 60], dtype=np.uint8)
                    upper_skin = np.array([20, 150, 255], dtype=np.uint8)
                    skin_mask = cv2.inRange(shirt_region_hsv, lower_skin, upper_skin)
                    mask = cv2.bitwise_not(skin_mask)

                    # Exclude dark pixels
                    min_value = 50
                    value_mask = shirt_region_hsv[:, :, 2] >= min_value
                    mask = cv2.bitwise_and(mask, value_mask.astype(np.uint8) * 255)

                    valid_pixels_hsv = shirt_region_hsv[mask > 0]

                    if len(valid_pixels_hsv) >= 100:
                        hue_values = valid_pixels_hsv[:, 0]
                        hist, bins = np.histogram(hue_values, bins=180, range=[0, 180])
                        dominant_hue = bins[np.argmax(hist)]

                        hue_mask = (valid_pixels_hsv[:, 0] >= dominant_hue - 5) & (valid_pixels_hsv[:, 0] <= dominant_hue + 5)
                        dominant_pixels = valid_pixels_hsv[hue_mask]
                        avg_saturation = np.mean(dominant_pixels[:, 1]) if len(dominant_pixels) > 0 else 0
                        avg_value = np.mean(dominant_pixels[:, 2]) if len(dominant_pixels) > 0 else 0

                        hsv_color = np.array([[[dominant_hue, int(avg_saturation), int(avg_value)]]], dtype=np.uint8)
                        rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)[0][0]
                        dominant_color = rgb_color.astype(int)

                        def classify_color(hue, saturation, value):
                            if value > 180 and saturation < 20:
                                return "white"
                            elif value < 50:
                                return "black"
                            elif saturation < 30 and value > 50:
                                return "gray"
                            elif 90 <= hue <= 150 and saturation > 40:
                                return "blue"
                            elif (0 <= hue <= 10 or 170 <= hue <= 179) and saturation > 40:
                                return "red"
                            elif 35 <= hue <= 85:
                                return "green"
                            elif 20 <= hue <= 34:
                                return "yellow"
                            elif 141 <= hue <= 169:
                                return "purple"
                            elif 0 <= hue <= 20 and saturation > 50 and value < 200:
                                return "pink"
                            elif 10 <= hue <= 20:
                                return "orange"
                            elif saturation > 50 and value < 100:
                                return "brown"
                            else:
                                return "unknown"

                        color_name = classify_color(dominant_hue, avg_saturation, avg_value)

            # Create annotated image
            annotated_image = image_rgb.copy()

            # Draw gender detection box
            if gender_box is not None:
                x1, y1, x2, y2 = map(int, gender_box)
                label = f"Gender: {gender_status} ({gender_predictions.conf[0].item():.2f})"
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green
                cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Draw glasses detection box
            if glasses_box is not None:
                x1, y1, x2, y2 = map(int, glasses_box)
                label = f"Glasses: {glasses_status} ({glasses_predictions.conf[0].item():.2f})"
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue
                cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            # Draw shirt region box and landmarks
            if results.pose_landmarks and shirt_box is not None:
                x1, y1, x2, y2 = map(int, shirt_box)
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red
                cv2.circle(annotated_image, left_shoulder, 5, (255, 0, 0), -1)
                cv2.circle(annotated_image, right_shoulder, 5, (255, 0, 0), -1)
                cv2.circle(annotated_image, left_hip, 5, (0, 255, 0), -1)
                cv2.circle(annotated_image, right_hip, 5, (0, 255, 0), -1)

                # Draw color patch
                patch_size = 50
                annotated_image[-patch_size:, -patch_size:, :] = dominant_color
                cv2.rectangle(annotated_image, (width - patch_size, height - patch_size), (width, height), (255, 255, 255), 2)
                cv2.putText(annotated_image, f"Shirt: {color_name}", (width - patch_size, height - patch_size - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            # Save output image
            output_filename = f"output_{uuid.uuid4().hex}.jpg"
            output_path = os.path.join(OUTPUT_FOLDER, output_filename)
            cv2.imwrite(output_path, cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

            # Clean up uploaded file
            if os.path.exists(file_path):
                os.remove(file_path)

            # Render results
            return render_template('index.html', 
                                output_image=output_filename,
                                gender=gender_status,
                                glasses=glasses_status,
                                shirt_color=color_name)

    return render_template('index.html')

@app.route('/outputs/<filename>')
def serve_output(filename):
    return send_file(os.path.join(OUTPUT_FOLDER, filename))

if __name__ == '__main__':
    app.run(debug=True)