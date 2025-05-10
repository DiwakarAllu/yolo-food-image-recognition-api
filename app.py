from flask import Flask, redirect, request, jsonify
from ultralytics import YOLO
from PIL import Image
from gradio_client import Client, handle_file
import tempfile
from flask_cors import CORS
import os
import cv2
import tempfile
import numpy as np
from ultralytics import YOLO
from werkzeug.utils import secure_filename
import base64
from db_data import class_names, nutrition_data


app = Flask(__name__)
client = Client("cutiepi3/bhojan-ai")
CORS(app)
model = YOLO("best.pt")


@app.route("/predict", methods=["POST"])
def hugging_predict():
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    image = request.files["image"]


    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        image.save(temp.name)
        temp_path = temp.name

    try:
       
        result = client.predict(
            image=handle_file(temp_path),
            api_name="/yolo_predict"
        )

       
        os.remove(temp_path)

        image_path = result[0]  
        detections = result[1]

        
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        encoded_img = base64.b64encode(img_bytes).decode("utf-8")

        if os.path.exists(image_path):
            os.remove(image_path)

        return jsonify({
            "detections": detections,
            "image": encoded_img
        })

    except Exception as e:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({"error": str(e)}), 500

@app.route('/yolo_predict', methods=['POST'])
def yolo_predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    filename = secure_filename(file.filename)
    temp_dir = tempfile.mkdtemp()
    file_path = os.path.join(temp_dir, filename)
    file.save(file_path)

    try:
        results = model(file_path)[0]
        image = cv2.imread(file_path)

        detections = []

        for box in results.boxes:
            cls_id = int(box.cls)
            confidence = float(box.conf)
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            label = f"{class_names[cls_id]}: {confidence:.2f}"

            
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2)

            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            area = (x2 - x1) * (y2 - y1)

            detections.append({
                "class": class_names[cls_id],
                "confidence": round(confidence, 2),
                "bbox": [x1, y1, x2, y2],
                "center": {"x": center_x, "y": center_y},
                "nutritional_info": nutrition_data.get(str(cls_id), {}),
                "area": area
            })

        _, buffer = cv2.imencode('.jpg', image)
        encoded_image = base64.b64encode(buffer.tobytes()).decode('utf-8')

        return jsonify({
            "image": encoded_image,
            "detections": detections
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/web", methods=["GET"])
def webview():
    return redirect("https://cutiepi3-bhojan-ai.hf.space/?__theme=system", code=302)  

@app.route("/", methods=["GET"])
def home():
    return "Bhojan AI Food Detection API is up!😊", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
