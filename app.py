import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

model = load_model()

# Preprocess the image
def preprocess_image(image):
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.LANCZOS)
    image_array = np.asarray(image)
    normalized_image_array = image_array.astype(np.float32) / 255.0
    reshaped_image = normalized_image_array.reshape(1, 224, 224, 3)
    return reshaped_image

# Create the Streamlit app
st.title("Skin Disease Prediction App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    predicted_class = np.argmax(prediction)

    # Replace with your actual class labels
    class_labels = ['bkl', 'nv', 'df', 'mel', 'vasc', 'bcc', 'akiec'] 

    st.write(f"Prediction: {class_labels[predicted_class]}")
