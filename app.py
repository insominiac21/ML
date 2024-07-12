import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps
import numpy as np

# Use st.cache for caching the model
# Load the model
@st.cache_resource
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
        class_labels = [
            'Benign keratosis-like lesions', 
            'Melanocytic nevi', 
            'Dermatofibroma', 
            'Melanoma', 
            'Vascular lesions', 
            'Basal cell carcinoma (BCC)',  
            'Actinic keratoses (AKs)'
        ]

        st.write(f"Prediction: {class_labels[predicted_class]}")

        # Details for each class
        details = {
            'Benign keratosis-like lesions': {
                'Diagnosis': """Medical Examination: A dermatologist or healthcare provider should visually inspect the lesion and may perform additional tests like a biopsy if necessary.
Differential Diagnosis: Differentiating between various benign conditions (like seborrheic keratosis, actinic keratosis, verruca vulgaris, etc.) is crucial for appropriate treatment.""",
                'Remedies and Treatments': """1. Observation: In some cases, especially if the lesion is small and stable, observation without treatment might be recommended.
2. Topical Treatments:
   - Topical Retinoids: Prescription retinoid creams can help in some cases, such as for actinic keratosis.
   - Cryotherapy: Freezing the lesion with liquid nitrogen can be effective for certain types of growths.
   - Topical Chemotherapy: For more persistent lesions, topical chemotherapy creams may be prescribed.
3. Surgical Procedures:
   - Curettage: Scraping off the lesion with a curette.
   - Electrosurgery: Using electricity to burn off or destroy the lesion.
   - Laser Therapy: Laser treatment to remove the lesion.
4. Home Remedies: While not substitutes for medical treatment, certain home remedies might alleviate symptoms temporarily, such as moisturizing the affected area or using over-the-counter hydrocortisone cream (under medical guidance)."""
            },
            'Melanocytic nevi': {
                'Diagnosis': """1. Visual Examination: A healthcare provider typically diagnoses melanocytic nevi through visual inspection. They look for features such as asymmetry, irregular borders, uneven coloration, and changes in size or shape.
2. Dermoscopy: In cases where the diagnosis is unclear, dermoscopy—a non-invasive technique using a handheld device with magnification and light—can aid in examining the mole's structure and pigment patterns.
3. Biopsy: If a mole shows suspicious features or changes, a biopsy may be recommended. A small sample of tissue is removed and examined under a microscope to confirm the diagnosis and rule out melanoma.""",
                'Remedies and Management': """1. Observation: Most melanocytic nevi are benign and require no treatment other than periodic observation. Your healthcare provider may recommend regular skin checks to monitor any changes.
2. Surgical Removal: This may be recommended if the mole shows concerning features (e.g., asymmetry, irregular borders, changes in color or size). Surgical removal can be done through excision (cutting out the mole and stitching the skin) or shave biopsy (shaving off the mole with a scalpel).
3. Cosmetic Removal: If the mole is bothersome cosmetically (e.g., it's in a prominent location or frequently irritated), it can be removed for aesthetic reasons.
4. Sun Protection: Since sun exposure can contribute to the development of new moles and potentially melanoma, practicing sun safety (using sunscreen, wearing protective clothing, avoiding peak sun hours) is essential."""
            },
            'Dermatofibroma': {
                'Diagnosis': """1. Clinical Examination: A dermatologist usually diagnoses dermatofibromas based on their typical appearance and texture. They may use a dermatoscope (a handheld instrument with light and magnification) to examine the lesion closely.
2. Biopsy: If the diagnosis is uncertain or if there are atypical features, a biopsy may be performed. A small sample of tissue is taken and examined under a microscope to confirm the diagnosis and rule out other conditions.""",
                'Remedies and Management': """1. Observation: Dermatofibromas are generally harmless and may not require treatment unless they cause symptoms or cosmetic concerns. Your healthcare provider may recommend periodic monitoring to check for changes.
2. Symptomatic Relief: If the dermatofibroma is causing itching or irritation, topical treatments such as corticosteroid creams or moisturizers may provide relief.
3. Surgical Excision: If the dermatofibroma is bothersome cosmetically or if there is uncertainty about the diagnosis, surgical removal (excision) may be recommended. This involves cutting out the lesion and closing the skin with stitches.
4. Cryotherapy: Freezing the dermatofibroma with liquid nitrogen may be an option for removal, particularly for smaller lesions."""
            },
            'Melanoma': {
                'Diagnosis': """1. Clinical Examination: A dermatologist or healthcare provider will conduct a thorough examination of your skin, including any suspicious moles or lesions.
2. Dermoscopy: Dermoscopy is a non-invasive technique that allows dermatologists to examine skin lesions with a magnifying tool equipped with light, aiding in the detection of suspicious features.
3. Biopsy: If a lesion is suspected to be melanoma based on its appearance or changes, a biopsy is performed. This involves removing a sample of tissue from the lesion and examining it under a microscope to confirm the diagnosis.""",
                'Remedies and Management': """1. Surgical Excision: The primary treatment for melanoma is surgical removal. Depending on the size and depth of the melanoma, the surgical procedure may involve removing the lesion along with a margin of normal skin (wide local excision).
2. Sentinel Lymph Node Biopsy: In cases where melanoma is thicker or has spread deeper into the skin, a sentinel lymph node biopsy may be recommended to determine if the cancer has spread to nearby lymph nodes.
3. Adjuvant Therapy: In certain cases of melanoma, especially those with a higher risk of recurrence or spread, adjuvant therapies such as targeted therapy or immunotherapy may be recommended after surgery to reduce the risk of recurrence.
4. Radiation Therapy: Radiation therapy may be used in some cases to treat melanoma that cannot be completely removed with surgery or to target areas where melanoma has spread."""
            },
            'Vascular lesions': {
                'Diagnosis': """1. Clinical Examination: A dermatologist or vascular specialist will visually inspect the lesion and may use a dermatoscope for a closer examination.
2. Imaging Studies: For deeper lesions or those affecting internal organs, imaging techniques such as ultrasound, MRI, or CT scans may be used to assess the extent and characteristics of the vascular lesion.
3. Biopsy: In some cases, a biopsy may be performed to confirm the diagnosis, especially if the lesion appears atypical or is suspected to be a vascular tumor.""",
                'Remedies and Management': """1. Observation: Many vascular lesions are harmless and may not require treatment unless they cause symptoms or cosmetic concerns. Regular monitoring may be recommended.
2. Topical Treatments: Some superficial vascular lesions, such as small hemangiomas or telangiectasias, may respond to topical treatments like laser therapy or topical medications (e.g., timolol gel for infantile hemangiomas).
3. Laser Therapy: Laser treatments, such as pulsed dye laser (PDL) or Nd:YAG laser, are often effective for treating vascular lesions like port-wine stains, spider veins, and certain hemangiomas.
4. Sclerotherapy: This involves injecting a sclerosing agent into the blood vessel to shrink and eventually eliminate small to medium-sized vascular malformations or spider veins.
5. Surgical Excision: Surgical removal may be considered for larger or deeper vascular lesions, particularly if they are causing symptoms or cosmetic concerns.
6. Embolization: For complex vascular malformations or lesions that bleed excessively, embolization may be performed to block off the abnormal blood vessels."""
            },
            'Basal cell carcinoma (BCC)': {
                'Diagnosis': """1. Clinical Examination: A dermatologist or healthcare provider will examine suspicious lesions on the skin. BCC often appears as a pinkish or pearly bump with a rolled edge, or as a flat, scaly, reddish patch.
2. Biopsy: If a lesion is suspected to be basal cell carcinoma based on its appearance or changes, a biopsy is performed. A small sample of tissue is removed from the lesion and examined under a microscope by a pathologist to confirm the diagnosis.""",
                'Remedies and Management': """1. Surgical Excision: The primary treatment for basal cell carcinoma is surgical removal. The procedure involves cutting out the cancerous tissue along with a surrounding margin of healthy skin. This is typically done under local anesthesia.
2. Mohs Surgery: Mohs micrographic surgery is a specialized technique used for BCC and other skin cancers, especially on the face or areas where tissue conservation is crucial. It involves removing thin layers of tissue and examining them under a microscope immediately, layer by layer, until no cancer cells remain.
3. Electrodessication and Curettage (ED&C): For small, superficial BCCs, ED&C may be used. The lesion is scraped with a curette (a sharp, spoon-shaped instrument) to remove cancerous tissue, followed by electrodessication to destroy any remaining cancer cells.
4. Topical Treatments: In certain cases of superficial BCCs or for patients who are not suitable candidates for surgery, topical medications like imiquimod or 5-fluorouracil (5-FU) may be prescribed to apply directly to the lesion over several weeks.
5. Radiation Therapy: Radiation therapy may be considered for elderly patients or those with medical conditions that make surgery difficult. It can also be used for areas where surgical removal is challenging, such as around the eyes or nose."""
            },
            'Actinic keratoses (AKs)': {
                'Diagnosis': """1. Clinical Examination: A dermatologist or healthcare provider will visually inspect the skin for signs of actinic keratoses. They may use a dermatoscope for a closer examination to differentiate AKs from other skin conditions.
2. Biopsy: If a lesion appears suspicious or if there is uncertainty about the diagnosis, a biopsy may be performed. A small sample of tissue is removed and examined under a microscope to confirm the presence of AK and rule out skin cancer.""",
                'Remedies and Management': """1. Topical Treatments:
   - Topical Retinoids: Prescription retinoid creams (e.g., tretinoin) help to normalize cell turnover and reduce the number of AK lesions.
   - Topical Chemotherapy: Creams containing 5-fluorouracil (5-FU) or imiquimod stimulate the immune system to attack AK cells.
   - Diclofenac Gel: This topical treatment has anti-inflammatory properties and is used specifically for AKs.
2. Cryotherapy: Liquid nitrogen is applied to freeze and destroy AKs. This is a common treatment for isolated lesions.
3. Photodynamic Therapy (PDT): This involves applying a photosensitizing agent to the skin and then exposing it to a light source that activates the agent, selectively destroying AK cells.
4. Chemical Peels: For widespread AKs, chemical peels may be used to remove the top layers of skin, promoting new skin growth.
5. Surgical Removal: In cases where AKs have progressed to become thick or develop into squamous cell carcinoma, surgical removal may be necessary."""
            }
        }

        # Display details for the predicted class
        if class_labels[predicted_class] in details:
            detail = details[class_labels[predicted_class]]
            for key, value in detail.items():
                with st.expander(key):
                    st.write(value)

    except Exception as e:
        
        st.error(
            """
            An error occurred while processing the image:
            - Please ensure the image is not too small or too large and be closest to being a square.
            - Ideal image sizes range from small (e.g., 100x100 pixels) to moderately large (e.g., 1000x1000 pixels).
            - Extremely large images or those with unusual aspect ratios might cause processing issues.
            
            **Error details:** {}
            """.format(e)
        )


