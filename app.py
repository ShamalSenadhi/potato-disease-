import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Potato Leaf Disease Classifier",
    page_icon="🥔",
    layout="centered"
)

# ---------------------------------------------------------
# Constants
# ---------------------------------------------------------
IMAGE_SIZE = 256
CLASS_NAMES = ['Bacteria', 'Fungi', 'Healthy', 'Nematode', 'Pest', 'Phytopthora', 'Virus']
MODEL_PATH = "model_v1.h5"   # change to your saved model filename (.h5 also works)

DISEASE_INFO = {
    "Bacteria": "Caused by bacteria entering through wounds or natural openings. Common symptoms: leaf spots, blights, cankers, wilts.",
    "Fungi": "Caused by fungal pathogens that thrive in moist conditions. Symptoms include leaf spots, mildews, blights, and rots.",
    "Healthy": "No visible signs of disease detected. The leaf appears healthy.",
    "Nematode": "Caused by parasitic roundworms affecting roots. Symptoms include stunted growth and yellowing leaves.",
    "Pest": "Caused by insect or mite infestations. Symptoms include chewing/sucking damage, webbing, and honeydew.",
    "Phytopthora": "Caused by oomycetes (water molds), often called Late Blight. Causes water-soaked lesions that turn dark brown/black.",
    "Virus": "Caused by plant viruses, often spread by insect vectors. Symptoms include mosaic patterns, leaf distortion, and stunting."
}

# ---------------------------------------------------------
# Load model (cached so it only loads once)
# ---------------------------------------------------------
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# ---------------------------------------------------------
# Prediction function
# ---------------------------------------------------------
def predict(image: Image.Image):
    image = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.utils.img_to_array(image)
    img_array = tf.expand_dims(img_array, 0)  # create batch dimension

    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    all_scores = {CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2) for i in range(len(CLASS_NAMES))}
    return predicted_class, confidence, all_scores

# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("🥔 Potato Leaf Disease Classifier")
st.write(
    "Upload a photo of a potato leaf and this app will predict whether it is "
    "**Healthy** or affected by **Bacteria, Fungi, Nematode, Pest, Phytophthora, or Virus**."
)

uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

col1, col2 = st.columns(2)

if uploaded_file is not None:
    image = Image.open(uploaded_file)

    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analysing leaf..."):
        predicted_class, confidence, all_scores = predict(image)

    with col2:
        st.subheader("Prediction")
        if predicted_class == "Healthy":
            st.success(f"**{predicted_class}** ({confidence}% confidence)")
        else:
            st.error(f"**{predicted_class}** ({confidence}% confidence)")

        st.write(DISEASE_INFO[predicted_class])

    st.subheader("Confidence breakdown")
    st.bar_chart(all_scores)

else:
    st.info("👆 Upload an image to get a prediction.")

st.markdown("---")
st.caption("Model trained on potato leaf disease dataset (7 classes) using TensorFlow/Keras CNN.")
