from flask import Flask, request, render_template, jsonify
import os
from PIL import Image
import numpy as np
import tensorflow as tf
import logging

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'path_to_your_trained_model.h5'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load the trained model
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise

# Define the label mappings
label_to_id = {'nv': 0, 'mel': 1, 'bkl': 2, 'bcc': 3, 'akiec': 4, 'vasc': 5, 'df': 6}
id_to_label = {v: k for k, v in label_to_id.items()}

# TensorFlow memory growth configuration for GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if physical_devices:
    try:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)
    except RuntimeError as e:
        logging.error(f"Error setting GPU memory growth: {e}")

def preprocess_image(image):
    """Preprocess the image to the format required by the model."""
    try:
        image = Image.open(image).convert('RGB')
        image = image.resize((224, 224))  # Assuming the model requires 224x224 input size
        image = np.array(image)
        image = image / 255.0  # Normalize to [0, 1] range
        image = np.expand_dims(image, axis=0)  # Add batch dimension
        logging.info("Image preprocessed successfully.")
        return image
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        logging.error("No file part in request.")
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        logging.error("No file selected.")
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Open the image file
        image = Image.open(filepath)
        preprocessed_image = preprocess_image(image)

        if preprocessed_image is None:
            logging.error("Image preprocessing failed.")
            return jsonify({'error': 'Image preprocessing failed'}), 400

        try:
            # Predict using the model
            prediction = model.predict(preprocessed_image)
            predicted_class = np.argmax(prediction, axis=1)[0]

            # Map the predicted class to the corresponding label
            result = id_to_label[predicted_class]

            logging.info(f"Prediction successful: {result}")
            return jsonify({'prediction': result})
        except Exception as e:
            logging.error(f"Prediction error: {e}")
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
