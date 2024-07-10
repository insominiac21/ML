from flask import Flask, request, send_file, render_template, jsonify
import os
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
MODEL_PATH = 'path_to_your_trained_model.h5'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image):
    """Preprocess the image to the format required by the model."""
    image = image.resize((224, 224))  # Assuming the model requires 224x224 input size
    image = np.array(image)
    image = image / 255.0  # Normalize to [0, 1] range
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Open the image file
        image = Image.open(filepath)
        preprocessed_image = preprocess_image(image)

        # Predict using the model
        prediction = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # For simplicity, let's assume your model's output is an integer corresponding to the class
        # You might need to map this to an actual class name
        classes = ['Class 0', 'Class 1', 'Class 2']  # Example class names
        result = classes[predicted_class]

        return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
