import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from datetime import datetime
import pandas as pd
import os
import traceback

# ===========================================================
# Page configuration
# ===========================================================
st.set_page_config(
    page_title="Potato Leaf Disease Detector",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ===========================================================
# Constants
# ===========================================================
IMAGE_SIZE = 256
CLASS_NAMES = ['Bacteria', 'Fungi', 'Healthy', 'Nematode', 'Pest', 'Phytopthora', 'Virus']

# Map of available models -> local file path
# Update these paths/filenames to match whatever you actually uploaded to the repo
MODEL_PATHS = {
    "Custom CNN": "cnn_model_v1.keras",
    "VGG16 (Transfer Learning)": "vgg16_model_v1.keras",
    "MobileNetV2 (Transfer Learning)": "mobilenet_v2_model_v1.keras",
}

# Recorded test-set results from model evaluation (update if you re-train)
MODEL_RESULTS = {
    "Custom CNN": 54.41,
    "VGG16 (Transfer Learning)": 69.28,
    "MobileNetV2 (Transfer Learning)": 81.37,
}

COMPARISON_CSV_PATH = "model_comparison_results.csv"

# ===========================================================
# Disease knowledge base
# ===========================================================
DISEASE_INFO = {
    "Healthy": {
        "icon": "✅", "severity": "None", "severity_color": "green",
        "description": "No visible signs of disease. This leaf appears healthy.",
        "symptoms": ["Uniform green colour", "No spots, lesions, or discolouration", "Normal leaf shape and texture"],
        "causes": [],
        "treatment": ["No treatment needed.", "Continue routine monitoring to catch any issues early."],
        "prevention": [
            "Maintain a consistent watering schedule (avoid overwatering).",
            "Ensure good air circulation between plants.",
            "Rotate crops each season to reduce pathogen build-up in soil.",
            "Continue regular field scouting."
        ]
    },
    "Bacteria": {
        "icon": "🦠", "severity": "High", "severity_color": "red",
        "description": "Bacterial diseases are caused by bacteria entering through wounds, natural openings, or insect vectors, multiplying and spreading through plant tissue.",
        "symptoms": ["Leaf spots", "Blights", "Cankers", "Wilting", "Galls"],
        "causes": ["Bacterial pathogens surviving in soil, plant debris, or seeds", "Spread via water splash, tools, or insects"],
        "treatment": [
            "Remove and destroy infected leaves/plants immediately to limit spread.",
            "Apply copper-based bactericides as a protective spray (follow local agricultural guidelines).",
            "Avoid overhead irrigation — water at the base of plants instead.",
            "Disinfect tools between uses (e.g. diluted bleach or alcohol solution)."
        ],
        "prevention": [
            "Use certified disease-free seed potatoes.",
            "Practice crop rotation (avoid planting potatoes in the same soil for 2–3 years).",
            "Avoid working in fields when leaves are wet.",
            "Improve field drainage and avoid waterlogging."
        ]
    },
    "Fungi": {
        "icon": "🍄", "severity": "Medium-High", "severity_color": "orange",
        "description": "Fungal diseases are among the most common plant diseases, thriving in moist and humid conditions. Fungi penetrate tissue through wounds or natural openings.",
        "symptoms": ["Leaf spots", "Rusts", "Powdery or downy mildew", "Blights", "Rot"],
        "causes": ["Fungal spores spread by wind, water, insects, or human activity", "Favoured by warm, humid conditions"],
        "treatment": [
            "Apply an appropriate fungicide (e.g. chlorothalonil or mancozeb-based products) as soon as symptoms appear.",
            "Remove and destroy heavily infected foliage.",
            "Improve air circulation by spacing plants and pruning excess foliage.",
            "Avoid overhead watering, especially in the evening."
        ],
        "prevention": [
            "Plant resistant potato varieties where available.",
            "Practice crop rotation.",
            "Apply preventive fungicide sprays during high-risk (humid) periods.",
            "Remove plant debris after harvest to reduce overwintering spores."
        ]
    },
    "Nematode": {
        "icon": "🪱", "severity": "Medium", "severity_color": "orange",
        "description": "Nematode diseases are caused by microscopic parasitic roundworms that infest roots, reducing water and nutrient uptake.",
        "symptoms": ["Stunted growth", "Yellowing (chlorosis)", "Wilting despite adequate moisture", "Root galls or lesions"],
        "causes": ["Plant-parasitic nematodes living in soil", "Spread via contaminated soil, water, equipment, or planting material"],
        "treatment": [
            "Apply nematicides if infestation is severe (consult local agricultural extension for approved products).",
            "Solarize soil (cover with clear plastic in hot weather) to reduce nematode populations before replanting.",
            "Remove and destroy severely infected plants."
        ],
        "prevention": [
            "Rotate with non-host crops (e.g. cereals) for at least 2 seasons.",
            "Use nematode-resistant potato varieties.",
            "Maintain healthy soil with organic matter to support natural nematode predators.",
            "Clean equipment between fields to avoid spreading infested soil."
        ]
    },
    "Pest": {
        "icon": "🐛", "severity": "Medium", "severity_color": "orange",
        "description": "Pest damage is caused by insect or mite infestations rather than microbial pathogens — these organisms feed on plant tissue or transmit disease.",
        "symptoms": ["Holes or shredded leaves", "Yellowing/stippling from sucking insects", "Webbing (spider mites)", "Honeydew and sooty mould"],
        "causes": ["Insects (aphids, whiteflies, beetles, caterpillars)", "Mites (e.g. spider mites)"],
        "treatment": [
            "Apply an appropriate insecticide or miticide targeted to the pest observed.",
            "Introduce or encourage natural predators (ladybirds, lacewings) for biological control.",
            "Use insecticidal soap or neem oil for milder infestations.",
            "Manually remove heavily infested leaves if the infestation is localized."
        ],
        "prevention": [
            "Regularly scout crops for early signs of pest activity.",
            "Use row covers or physical barriers where practical.",
            "Avoid excess nitrogen fertilization, which attracts sap-sucking pests.",
            "Maintain field hygiene — remove weeds that can host pests."
        ]
    },
    "Phytopthora": {
        "icon": "💧", "severity": "Critical", "severity_color": "red",
        "description": "Phytophthora disease (Late Blight) is caused by oomycetes that spread rapidly under cool, wet, and humid conditions — it can destroy a crop within days if untreated.",
        "symptoms": ["Water-soaked lesions on leaves/stems", "Rapidly expanding dark brown/black patches", "Fuzzy white growth on leaf undersides", "Tuber rot"],
        "causes": ["Phytophthora infestans spores spread by wind, rain splash, and contaminated soil/water", "Favoured by 18–25°C and high humidity"],
        "treatment": [
            "⚠️ Act immediately — this disease can spread and destroy a crop within days.",
            "Apply a systemic fungicide labelled for Late Blight (e.g. metalaxyl or fluazinam-based products).",
            "Remove and destroy infected plants/tubers — do not compost them.",
            "Avoid overhead irrigation and reduce field humidity where possible."
        ],
        "prevention": [
            "Plant certified, disease-free seed potatoes.",
            "Use resistant varieties where available.",
            "Apply preventive fungicide sprays before wet, humid weather.",
            "Destroy volunteer potato plants and cull piles that can harbour the pathogen.",
            "Ensure good field drainage and airflow."
        ]
    },
    "Virus": {
        "icon": "🧬", "severity": "High", "severity_color": "red",
        "description": "Viral diseases are caused by plant viruses — obligate parasites that replicate only inside host cells and are typically spread by insect vectors.",
        "symptoms": ["Mosaic or mottling patterns", "Leaf curling/distortion", "Stunting", "Yellowing", "Ring spots"],
        "causes": ["Insect vectors (aphids, whiteflies, thrips)", "Contaminated tools, seed, or grafting material"],
        "treatment": [
            "⚠️ There is no cure for viral infections once established.",
            "Remove and destroy infected plants immediately to prevent spread.",
            "Control insect vectors (aphids, whiteflies) with appropriate insecticides.",
            "Avoid handling healthy plants after touching infected ones."
        ],
        "prevention": [
            "Use certified virus-free seed potatoes.",
            "Control aphid and insect vector populations proactively.",
            "Remove weeds that may host viruses or vectors.",
            "Disinfect tools between plants."
        ]
    },
}

# ===========================================================
# Model loading with validation — never crashes the whole app.
# Cached per model path so each is only loaded once per session.
# ===========================================================
@st.cache_resource(show_spinner=False)
def _load_model_cached(model_path):
    return tf.keras.models.load_model(model_path)

def safe_load_model(name, path):
    """
    Attempt to load a model. Returns (model, error_message).
    error_message is None on success.
    """
    if not os.path.exists(path):
        return None, f"File not found: `{path}`. Make sure it's uploaded to the app folder."
    try:
        model = _load_model_cached(path)
        return model, None
    except Exception as e:
        return None, f"Failed to load `{path}`: {e}"

def validate_all_models():
    """
    Check every configured model without necessarily loading it into memory
    long-term (loading is still cached, so this is a one-time cost).
    Returns dict: name -> {"status": "ok"/"error", "detail": str or None}
    """
    status = {}
    for name, path in MODEL_PATHS.items():
        model, error = safe_load_model(name, path)
        if error:
            status[name] = {"status": "error", "detail": error}
        else:
            status[name] = {"status": "ok", "detail": None}
    return status

# ===========================================================
# Session state
# ===========================================================
if "history" not in st.session_state:
    st.session_state.history = []
if "model_status" not in st.session_state:
    st.session_state.model_status = None

# ===========================================================
# Prediction function
# ===========================================================
def predict(model, image: Image.Image):
    img = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array, verbose=0)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    all_scores = {CLASS_NAMES[i]: round(float(predictions[0][i]) * 100, 2) for i in range(len(CLASS_NAMES))}
    return predicted_class, confidence, all_scores

# ===========================================================
# Sidebar navigation
# ===========================================================
st.sidebar.title("🥔 Potato Disease Detector")
page = st.sidebar.radio(
    "Navigate",
    ["🔍 Detect Disease", "✅ Model Validation", "⚖️ Model Comparison", "📚 Disease Library", "📊 Session History", "ℹ️ About"]
)

st.sidebar.markdown("---")
st.sidebar.caption(f"Classes: {', '.join(CLASS_NAMES)}")

# ===========================================================
# PAGE 1: Detect Disease
# ===========================================================
if page == "🔍 Detect Disease":
    st.title("🔍 Detect Potato Leaf Disease")
    st.write("Upload a photo of a potato leaf. Choose a model to run the prediction, "
             "or check the **Model Validation** page first if you're unsure which models are working.")

    working_models = {name: path for name, path in MODEL_PATHS.items() if os.path.exists(path)}

    if not working_models:
        st.error("No model files were found in the app folder. Check the **Model Validation** page for details.")
    else:
        selected_model_name = st.selectbox("Choose model", list(working_models.keys()))
        uploaded_file = st.file_uploader("Upload a leaf image", type=["jpg", "jpeg", "png"])

        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            col1, col2 = st.columns([1, 1.3])
            with col1:
                st.image(image, caption="Uploaded Image", use_container_width=True)

            with st.spinner(f"Loading {selected_model_name}..."):
                model, error = safe_load_model(selected_model_name, working_models[selected_model_name])

            if error:
                st.error(f"⚠️ Could not use this model: {error}")
                st.info("Try a different model, or check the **Model Validation** page for details on what went wrong.")
            else:
                with st.spinner("Analysing leaf..."):
                    predicted_class, confidence, all_scores = predict(model, image)

                info = DISEASE_INFO[predicted_class]

                st.session_state.history.append({
                    "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Model": selected_model_name,
                    "Prediction": predicted_class,
                    "Confidence (%)": confidence,
                    "Severity": info["severity"]
                })

                with col2:
                    st.subheader(f"{info['icon']} Prediction: {predicted_class}")
                    st.caption(f"Model used: {selected_model_name}")

                    if predicted_class == "Healthy":
                        st.success(f"**Confidence: {confidence}%**")
                    else:
                        st.error(f"**Confidence: {confidence}%**  |  Severity: **{info['severity']}**")

                    st.write(info["description"])

                    st.markdown("**Confidence breakdown across all classes**")
                    df_scores = pd.DataFrame({"Class": list(all_scores.keys()), "Confidence (%)": list(all_scores.values())})
                    st.bar_chart(df_scores.set_index("Class"))

                st.markdown("---")
                tab1, tab2, tab3 = st.tabs(["🩺 Symptoms & Causes", "💊 Treatment", "🛡️ Prevention"])
                with tab1:
                    if info["symptoms"]:
                        st.markdown("**Common symptoms:**")
                        for s in info["symptoms"]:
                            st.markdown(f"- {s}")
                    if info["causes"]:
                        st.markdown("**Causes:**")
                        for c in info["causes"]:
                            st.markdown(f"- {c}")
                    if not info["symptoms"] and not info["causes"]:
                        st.write("No disease symptoms detected.")
                with tab2:
                    st.markdown("**Recommended actions:**")
                    for t in info["treatment"]:
                        st.markdown(f"- {t}")
                with tab3:
                    st.markdown("**How to prevent this in future:**")
                    for p in info["prevention"]:
                        st.markdown(f"- {p}")

                if predicted_class != "Healthy":
                    st.warning("⚠️ This prediction is for guidance only. For confirmed diagnosis and treatment, "
                               "consult a local agricultural extension officer or plant pathologist.")
        else:
            st.info("👆 Upload an image to get started.")
            with st.expander("💡 Tips for best results"):
                st.markdown("""
                - Use a clear, well-lit photo of a single leaf.
                - Avoid heavy shadows or blur.
                - Fill most of the frame with the leaf.
                - Photograph both sides of the leaf if symptoms are unclear on one side.
                """)

# ===========================================================
# PAGE 2: Model Validation
# ===========================================================
elif page == "✅ Model Validation":
    st.title("✅ Model Validation")
    st.write("Checks whether each configured model file exists and loads correctly. "
             "Run this after deploying or updating model files to confirm everything works.")

    if st.button("🔄 Run validation check"):
        with st.spinner("Checking all models..."):
            st.session_state.model_status = validate_all_models()

    if st.session_state.model_status is None:
        st.info("Click **Run validation check** to test all configured models.")
    else:
        for name, path in MODEL_PATHS.items():
            result = st.session_state.model_status.get(name, {"status": "unknown", "detail": None})
            col1, col2 = st.columns([1, 4])
            with col1:
                if result["status"] == "ok":
                    st.success("✅ OK")
                else:
                    st.error("❌ Failed")
            with col2:
                st.markdown(f"**{name}**  \n`{path}`")
                if result["detail"]:
                    with st.expander("Show error details"):
                        st.code(result["detail"])

        st.markdown("---")
        n_ok = sum(1 for r in st.session_state.model_status.values() if r["status"] == "ok")
        n_total = len(st.session_state.model_status)
        if n_ok == n_total:
            st.success(f"All {n_total} models loaded successfully.")
        elif n_ok == 0:
            st.error("No models loaded successfully. Check file paths and formats.")
        else:
            st.warning(f"{n_ok} of {n_total} models loaded successfully. "
                       "Failed models are excluded from the Detect Disease page automatically.")

        with st.expander("💡 Common causes of load failures"):
            st.markdown("""
            - **File not found** — the `.keras` file wasn't pushed to the repo, or the filename doesn't match `MODEL_PATHS` in the code.
            - **Format mismatch** — a model saved as `.h5` with an incompatible Keras version. Re-save using `model.save("name.keras")` (native Keras 3 format) rather than `.h5`.
            - **Corrupted upload** — the file was only partially uploaded (check its size on GitHub matches what you expect).
            - **Custom objects/layers** — if the model uses a custom Lambda layer (e.g. `preprocess_input`) that isn't recognised at load time, you may need to pass `custom_objects` to `load_model()`.
            """)

# ===========================================================
# PAGE 3: Model Comparison
# ===========================================================
elif page == "⚖️ Model Comparison":
    st.title("⚖️ Model Comparison")
    st.write("Performance comparison across the three candidate architectures evaluated for this project.")

    if os.path.exists(COMPARISON_CSV_PATH):
        metrics_df = pd.read_csv(COMPARISON_CSV_PATH, index_col=0)
        st.dataframe(metrics_df, use_container_width=True)
    else:
        results_df = pd.DataFrame({
            "Model": list(MODEL_RESULTS.keys()),
            "Test Accuracy (%)": list(MODEL_RESULTS.values())
        }).set_index("Model")
        st.bar_chart(results_df)
        st.dataframe(results_df, use_container_width=True)
        st.caption(f"Loaded from recorded evaluation results. Add `{COMPARISON_CSV_PATH}` to the app "
                   "folder to override with a freshly exported comparison table.")

    st.markdown("---")
    st.markdown("""
    ### Architecture summary

    | Model | Approach | Strength | Trade-off |
    |---|---|---|---|
    | **Custom CNN** | Trained from scratch | Lightweight, fully interpretable, no external dependencies | Needs more data to generalise well; higher overfitting risk |
    | **VGG16** | Transfer learning (ImageNet) | Strong pretrained features | Heaviest model — slow to train and deploy |
    | **MobileNetV2** | Transfer learning (ImageNet) | Best accuracy-to-size trade-off; fast inference | Slightly less expressive than deeper networks like VGG16 |

    **Recommended for deployment:** MobileNetV2 achieved the highest test accuracy (81.37%) while
    remaining lightweight enough for fast, responsive predictions.
    """)

# ===========================================================
# PAGE 4: Disease Library
# ===========================================================
elif page == "📚 Disease Library":
    st.title("📚 Potato Disease Library")
    st.write("Reference guide for all disease classes this model can detect.")

    for disease, info in DISEASE_INFO.items():
        with st.expander(f"{info['icon']} {disease}  —  Severity: {info['severity']}"):
            st.write(info["description"])
            col1, col2 = st.columns(2)
            with col1:
                if info["symptoms"]:
                    st.markdown("**Symptoms**")
                    for s in info["symptoms"]:
                        st.markdown(f"- {s}")
                if info["causes"]:
                    st.markdown("**Causes**")
                    for c in info["causes"]:
                        st.markdown(f"- {c}")
            with col2:
                st.markdown("**Treatment**")
                for t in info["treatment"]:
                    st.markdown(f"- {t}")
                st.markdown("**Prevention**")
                for p in info["prevention"]:
                    st.markdown(f"- {p}")

# ===========================================================
# PAGE 5: Session History
# ===========================================================
elif page == "📊 Session History":
    st.title("📊 Session Prediction History")

    if len(st.session_state.history) == 0:
        st.info("No predictions yet in this session. Go to **Detect Disease** to analyse a leaf image.")
    else:
        df_history = pd.DataFrame(st.session_state.history)
        st.dataframe(df_history, use_container_width=True)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total scans", len(df_history))
        col2.metric("Healthy detections", int((df_history["Prediction"] == "Healthy").sum()))
        col3.metric("Disease detections", int((df_history["Prediction"] != "Healthy").sum()))
        col4.metric("Models used", df_history["Model"].nunique() if "Model" in df_history.columns else 1)

        st.markdown("**Detections by class**")
        st.bar_chart(df_history["Prediction"].value_counts())

        if "Model" in df_history.columns:
            st.markdown("**Scans by model**")
            st.bar_chart(df_history["Model"].value_counts())

        csv = df_history.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download history as CSV", csv, "prediction_history.csv", "text/csv")

        if st.button("🗑️ Clear history"):
            st.session_state.history = []
            st.rerun()

# ===========================================================
# PAGE 6: About
# ===========================================================
elif page == "ℹ️ About":
    st.title("ℹ️ About this project")
    st.markdown("""
    This dashboard uses three Convolutional Neural Network architectures — a **Custom CNN**,
    **VGG16** (transfer learning), and **MobileNetV2** (transfer learning) — to classify potato
    leaf images into **7 categories**: Bacteria, Fungi, Healthy, Nematode, Pest, Phytophthora, and Virus.

    **How it works**
    1. Check the **Model Validation** page to confirm which models are working.
    2. Upload a photo of a potato leaf on the **Detect Disease** page.
    3. Choose a model to run the prediction.
    4. The image is resized to 256×256 pixels and passed through the selected model.
    5. The model outputs a predicted class along with a confidence score.
    6. The dashboard provides relevant symptoms, causes, treatment, and prevention guidance
       based on the prediction.

    **Why validate models separately?**
    Model files can fail to load for reasons like format mismatches or incomplete uploads.
    Rather than letting one broken file crash the entire app, this dashboard checks each model
    independently and only offers the ones that load successfully.

    **Disclaimer**
    This tool is intended to assist with preliminary screening only and should not replace
    professional agricultural diagnosis. Always consult a qualified agronomist or plant
    pathologist for confirmed diagnosis and treatment planning.

    ---
    Built with Streamlit, TensorFlow, and Keras.
    """)
