import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Use st.cache_data for caching the model
@st.cache_data
def load_model():
    model = tf.keras.models.load_model('model.h5')
    return model

model = load_model()

# Preprocess the image with aspect ratio preservation and error handling
def preprocess_image(image):
    # Target size for the model
    target_size = (224, 224)
    # Preserve aspect ratio and pad the rest; fill color is set to black (0)
    image = ImageOps.pad(image, target_size, method=Image.LANCZOS, color=0)
    # Convert image to array
    image_array = np.asarray(image)
    # Normalize the image
    normalized_image_array = image_array.astype(np.float32) / 255.0
    # Reshape for the model input
    reshaped_image = normalized_image_array.reshape(1, *target_size, 3)
    return reshaped_image

# Streamlit app interface
st.title("Skin Disease Prediction App")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write("Classifying...")
        preprocessed_image = preprocess_image(image)
        prediction = model.predict(preprocessed_image)
        predicted_class = np.argmax(prediction)

        # Class labels
        class_labels = ['Benign keratosis-like lesions', 'Melanocytic nevi', 'Dermatofibroma', 'Melanoma', 'Vascular lesions', 'Basal cell carcinoma', 'Actinic keratoses'] 

        st.write(f"Prediction: {class_labels[predicted_class]}")
    except Exception as e:
        
        st.error(
            """
            An error occurred while processing the image:
            - Please ensure the image is not too small or too large and closest to being a square.
            - Ideal image sizes range from small (e.g., 100x100 pixels) to moderately large (e.g., 1000x1000 pixels).
            - Extremely large images or those with unusual aspect ratios might cause processing issues.
            
            **Error details:** {}
            """.format(e)
        )



