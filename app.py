import os
import numpy as np
import tensorflow as tf
from flask import Flask, request, jsonify
from PIL import Image

app = Flask(__name__)

# Load the trained model
model = tf.keras.models.load_model('path_to_your_trained_model.h5')

# Define a function to preprocess images
def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    return image.astype(np.float32)

# Define a route for health check
@app.route('/')
def health_check():
    return 'API is running!'

# Define a route for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        image = Image.open(file.stream)
        preprocessed_image = preprocess_image(image)
        image_batch = np.expand_dims(preprocessed_image, axis=0)
        predictions = model.predict(image_batch)
        predicted_label = np.argmax(predictions, axis=1)[0]
        id_to_label = {0: 'label_0', 1: 'label_1', 2: 'label_2', 3: 'label_3', 4: 'label_4', 5: 'label_5', 6: 'label_6'}  # Replace with your actual label mapping
        return jsonify({'predicted_label': id_to_label[predicted_label]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
