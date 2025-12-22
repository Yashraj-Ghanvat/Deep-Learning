import streamlit as st
import tensorflow as tf
from tensorflow import keras
import numpy as np
from PIL import Image
import cv2

# Page configuration
st.set_page_config(
    page_title="👓 Glasses Detection AI",
    page_icon="👓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for attractive styling
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    .upload-text {
        font-size: 1.2rem;
        color: #4a5568;
        text-align: center;
        padding: 20px;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        background: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px 0;
    }
    .prediction-text {
        font-size: 2rem;
        font-weight: bold;
        text-align: center;
        padding: 15px;
    }
    .with-glasses {
        color: #48bb78;
    }
    .without-glasses {
        color: #f56565;
    }
    .confidence-text {
        font-size: 1.3rem;
        text-align: center;
        color: #4a5568;
    }
    </style>
    """, unsafe_allow_html=True)

# Load model with caching
@st.cache_resource
def load_model():
    try:
        model_path = r"C:\Users\Admin\Desktop\Yashraj\Deep Learning\CNN\Glass Dection in image classification in CNN\best_model.keras"
        model = keras.models.load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Preprocess image
def preprocess_image(image, target_size=(150, 150)):
    """
    Preprocess the image for model prediction
    Model expects input shape (150, 150, 3)
    """
    img_array = np.array(image)
    # Convert to RGB if needed
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    elif img_array.shape[2] == 4:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
    # Resize image
    img_resized = cv2.resize(img_array, target_size)
    # Normalize pixel values
    img_normalized = img_resized / 255.0
    # Add batch dimension
    img_batch = np.expand_dims(img_normalized, axis=0)
    return img_batch

# Make prediction
def predict_glasses(model, image):
    """
    Make prediction on the image
    """
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image, verbose=0)
    
    # Assuming binary classification: 0 = No Glasses, 1 = With Glasses
    # Adjust this based on your model's output
    confidence = float(prediction[0][0])
    
    if confidence > 0.5:
        label = "👓 Wearing Glasses"
        css_class = "with-glasses"
    else:
        label = "😊 Not Wearing Glasses"
        css_class = "without-glasses"
        confidence = 1 - confidence
    
    return label, confidence, css_class

# Main app
def main():
    # Header
    st.markdown("<h1 style='text-align: center; color: #2d3748;'>👓 Glasses Detection AI</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #4a5568; font-size: 1.1rem;'>Upload an image to detect if the person is wearing glasses</p>", unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.image("https://cdn-icons-png.flaticon.com/512/2991/2991148.png", width=150)
        st.markdown("### 📊 About This App")
        st.info(
            "This application uses a Convolutional Neural Network (CNN) "
            "to detect whether a person in an image is wearing glasses or not."
        )
        st.markdown("### 🎯 How to Use")
        st.markdown("""
        1. Upload an image (JPG, JPEG, or PNG)
        2. Wait for the AI to analyze
        3. View the prediction results
        """)
        st.markdown("### 📈 Model Info")
        st.success("Model: CNN-based classifier\nStatus: Ready ✅")
    
    # Load model
    model = load_model()
    
    if model is None:
        st.error("⚠️ Failed to load the model. Please check the model path.")
        return
    
    # File uploader
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        uploaded_file = st.file_uploader(
            "Choose an image...",
            type=["jpg", "jpeg", "png"],
            help="Upload a clear image of a person's face"
        )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Create two columns for image and results
        col_img, col_result = st.columns(2)
        
        with col_img:
            st.markdown("### 📸 Uploaded Image")
            st.image(image, use_container_width=True)
        
        with col_result:
            st.markdown("### 🎯 Prediction Results")
            
            # Show loading spinner while predicting
            with st.spinner("🔍 Analyzing image..."):
                try:
                    label, confidence, css_class = predict_glasses(model, image)
                    
                    # Display results
                    st.markdown(f"""
                        <div class='result-box'>
                            <p class='prediction-text {css_class}'>{label}</p>
                            <p class='confidence-text'>Confidence: {confidence*100:.2f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence bar
                    st.progress(confidence)
                    
                    # Additional info
                    if confidence > 0.9:
                        st.success("✅ High confidence prediction!")
                    elif confidence > 0.7:
                        st.info("ℹ️ Good confidence prediction")
                    else:
                        st.warning("⚠️ Low confidence - try a clearer image")
                    
                except Exception as e:
                    st.error(f"Error during prediction: {str(e)}")
        
        # Action buttons
        st.markdown("<br>", unsafe_allow_html=True)
        col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
        
        with col_btn2:
            if st.button("🔄 Try Another Image", use_container_width=True):
                st.rerun()
    
    else:
        # Show sample instructions when no image is uploaded
        st.markdown("<br><br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.info("👆 Please upload an image to get started!")
            st.markdown("""
                <div style='text-align: center; padding: 20px;'>
                    <p style='font-size: 1.1rem; color: #4a5568;'>
                        Supported formats: JPG, JPEG, PNG<br>
                        Best results with clear, front-facing photos
                    </p>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()