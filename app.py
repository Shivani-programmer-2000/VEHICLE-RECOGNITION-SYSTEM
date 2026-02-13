from flask import Flask, request, render_template, jsonify, send_from_directory
import cv2
import numpy as np
import pytesseract
import os
from ultralytics import YOLO

# Load the trained YOLOv8 model
model = YOLO(r"C:\Users\Windows\vrs-web-app\runs\detect\train\weights\best.pt")

# Configure Tesseract path
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Initialize Flask app
app = Flask(__name__, static_folder='static')
app.config['UPLOAD_FOLDER'] = 'static/uploads'  # Directory to store processed images
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image_np = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(image_np, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': 'Invalid image format'}), 400

    # Run YOLO inference
    results = model(image)
    detections = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Bounding box
            conf = round(float(box.conf[0]), 2)  # Confidence score
            cls = int(box.cls[0])  # Class ID
            label = model.names[cls]  # Class label

            # Extract ROI for OCR
            roi = image[y1:y2, x1:x2]
            if roi.size == 0:
                ocr_text = "N/A"
            else:
                # Preprocess for better OCR
                gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
                ocr_text = pytesseract.image_to_string(gray, config='--psm 6').strip()

            # Draw bounding box and label
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, f"{label} {conf}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            detections.append({
                "class": label,
                "confidence": conf,
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "ocr_text": ocr_text if ocr_text else "N/A"
            })

    # Save image with detections
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], 'detected_image.jpg')
    cv2.imwrite(output_path, image)

    return jsonify({'detections': detections, 'image_url': f'/static/uploads/detected_image.jpg'})

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory(app.static_folder, filename)

if __name__ == '__main__':
    app.run(debug=True)
