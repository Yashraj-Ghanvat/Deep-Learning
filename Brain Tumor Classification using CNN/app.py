# app.py
import streamlit as st
import numpy as np
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Brain Tumor Classification",
    page_icon="🧠",
    layout="wide"
)

# ---------------------------
# Model Paths
# ---------------------------
MODEL_PATHS = {
    "Model (brain_tumor_model_new.h5)": r"C:\Users\Admin\Desktop\Yashraj\Deep Learning\Brain Tumor Classification using CNN\brain_tumor_model_new.h5",
    "Model (brain_tumor_model.keras)": r"C:\Users\Admin\Desktop\Yashraj\Deep Learning\Brain Tumor Classification using CNN\brain_tumor_model.keras",
    "Model (model.h5)": r"C:\Users\Admin\Desktop\Yashraj\Deep Learning\Brain Tumor Classification using CNN\model.h5"
}

# ---------------------------
# Cache model loading
# ---------------------------
@st.cache_resource
def load_cnn_model(model_path):
    try:
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Failed to load model at {model_path}\n\nError: {e}")
        return None

# ---------------------------
# Prediction function
# ---------------------------
def predict_tumor(model, uploaded_file, class_names):
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)[0]
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    return predicted_class, confidence, predictions

# ---------------------------
# Sidebar Navigation
# ---------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Upload & Classify", "Model Info", "About Us"])

# ---------------------------
# Home Page
# ---------------------------
if page == "Home":
    st.title("🧠 Brain Tumor Classification using VGG16")
    st.markdown("""
    This application allows you to upload MRI scans and classify tumor type 
    using deep learning models trained with **VGG16**.  

    ⚠️ *Disclaimer: This app is for **educational/demo purposes only** 
    and should not be used for medical diagnosis.*
    """)

# ---------------------------
# Upload & Classify Page
# ---------------------------
elif page == "Upload & Classify":
    st.header("📤 Upload MRI Scan")

    # Choose model
    model_choice = st.selectbox("Choose a Model", list(MODEL_PATHS.keys()))
    model_path = MODEL_PATHS[model_choice]

    if os.path.exists(model_path):
        model = load_cnn_model(model_path)
    else:
        st.error(f"⚠️ Model file not found: {model_path}")
        model = None

    uploaded_file = st.file_uploader("Upload an MRI Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        st.subheader("🖼 Uploaded Image Preview")
        st.image(uploaded_file, caption="Uploaded MRI", use_column_width=True)

        if model is not None and st.button("🔍 Run Classification"):
            # Define classes (update if needed)
            CLASS_NAMES = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]

            predicted_class, confidence, predictions = predict_tumor(model, uploaded_file, CLASS_NAMES)

            # Show results
            st.success(f"### ✅ Prediction: {predicted_class}")
            st.info(f"Confidence: {confidence:.2%}")

            # Probability distribution
            st.subheader("📊 Prediction Probabilities")
            prob_dict = {CLASS_NAMES[i]: float(predictions[i]) for i in range(len(CLASS_NAMES))}
            st.bar_chart(prob_dict)

# ---------------------------
# Model Info Page
# ---------------------------
elif page == "Model Info":
    st.header("📚 Model Information")
    st.markdown("""
    - **Base Architecture**: VGG16 (pre-trained on ImageNet, fine-tuned for brain tumor classification).  
    - **Classes**:  
        1. No Tumor  
        2. Glioma  
        3. Meningioma  
        4. Pituitary  
    - **Input Size**: 224 × 224 × 3  
    - **Framework**: TensorFlow / Keras  
    """)

# ---------------------------
# About Us Page
# ---------------------------
elif page == "About Us":
    st.header("👨‍💻 About Us")
    st.markdown("""
    This app was developed as part of a **Deep Learning Project on Medical Imaging**.  

    **Team**:  
    - Yashraj Ghanvat – Deep Learning Engineer  

    📧 Contact: your.email@example.com  
    """)
