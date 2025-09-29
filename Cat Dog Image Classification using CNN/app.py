import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="AI Pet Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Enhanced CSS styling
st.markdown("""
<style>
    /* Main container styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    
    .app-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .app-subtitle {
        font-size: 1.2rem;
        margin-top: 0.5rem;
        opacity: 0.9;
    }
    
    /* Card styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.08);
        border: 1px solid #e1e5e9;
        margin-bottom: 1rem;
    }
    
    .upload-card {
        border: 2px dashed #cbd5e0;
        text-align: center;
        padding: 2rem;
        background: #f8fafc;
    }
    
    /* Result styling */
    .result-card {
        text-align: center;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
    }
    
    .cat-result {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #2d3748;
        border: 2px solid #ff6b6b;
    }
    
    .dog-result {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        color: #2d3748;
        border: 2px solid #4299e1;
    }
    
    .prediction-text {
        font-size: 1.8rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .confidence-text {
        font-size: 1.2rem;
        opacity: 0.8;
    }
    
    /* Metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 8px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
    }
    
    /* Info boxes */
    .info-box {
        background: #edf2f7;
        border-left: 4px solid #4299e1;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .success-box {
        background: #f0fff4;
        border-left: 4px solid #48bb78;
        color: #2f855a;
    }
    
    .warning-box {
        background: #fffaf0;
        border-left: 4px solid #ed8936;
        color: #c05621;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: transform 0.2s;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    /* Remove streamlit branding */
    .stDeployButton {
        display: none;
    }
    
    #MainMenu {
        visibility: hidden;
    }
    
    footer {
        visibility: hidden;
    }
    
    .stException {
        display: none;
    }
    
    /* Image styling */
    .uploaded-image {
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    """Load the trained CNN model"""
    try:
        model_path = r"C:\Users\Admin\Desktop\Yashraj\Deep Learning\Cat Dog Image Classification using CNN\cat_dog_model.pkl"
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        return model, None
    except FileNotFoundError:
        return None, "Model file not found. Please check the file path."
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

def preprocess_image(image):
    """Preprocess image for model prediction"""
    try:
        # Convert to RGB
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to model input size
        image_resized = image.resize((128, 128))
        
        # Convert to array and normalize
        img_array = np.array(image_resized, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        st.error(f"Error preprocessing image: {str(e)}")
        return None

def predict_image(model, processed_image):
    """Generate prediction from processed image"""
    try:
        with st.spinner("Analyzing image..."):
            prediction = model.predict(processed_image, verbose=0)
            raw_score = float(prediction[0][0])
            
            # Interpret prediction (sigmoid output)
            if raw_score > 0.5:
                label = "Dog"
                confidence = raw_score
            else:
                label = "Cat"
                confidence = 1 - raw_score
                
            return label, confidence, raw_score
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return None, None, None

def create_confidence_visualization(raw_prediction):
    """Create interactive confidence chart using Plotly"""
    cat_confidence = (1 - raw_prediction) * 100
    dog_confidence = raw_prediction * 100
    
    fig = go.Figure()
    
    # Add bars
    fig.add_trace(go.Bar(
        y=['Cat', 'Dog'],
        x=[cat_confidence, dog_confidence],
        orientation='h',
        marker_color=['#ff6b6b', '#4299e1'],
        text=[f'{cat_confidence:.1f}%', f'{dog_confidence:.1f}%'],
        textposition='auto',
        textfont=dict(color='white', size=14, family="Arial Black"),
    ))
    
    fig.update_layout(
        title="Prediction Confidence",
        xaxis_title="Confidence (%)",
        height=200,
        margin=dict(l=50, r=50, t=50, b=50),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(size=12),
        showlegend=False
    )
    
    fig.update_xaxis(range=[0, 100], showgrid=True, gridcolor='lightgray')
    fig.update_yaxis(showgrid=False)
    
    return fig

def main():
    # Header
    st.markdown("""
    <div class="app-header">
        <h1 class="app-title">AI Pet Classifier</h1>
        <p class="app-subtitle">Advanced CNN-powered Cat vs Dog Detection</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Load model
    model, error = load_model()
    
    if model is None:
        st.error(f"**Model Loading Failed:** {error}")
        st.markdown("""
        <div class="info-box warning-box">
            <h4>Troubleshooting Steps:</h4>
            <ul>
                <li>Ensure the model file exists at the specified path</li>
                <li>Check file permissions</li>
                <li>Verify the model was saved correctly after training</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        return
    
    # Main content area
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📁 Image Upload")
        
        uploaded_file = st.file_uploader(
            "",
            type=['jpg', 'jpeg', 'png', 'webp'],
            help="Supported formats: JPG, JPEG, PNG, WEBP",
            label_visibility="collapsed"
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file)
            
            # Display image
            st.image(
                image, 
                caption="Uploaded Image", 
                use_column_width=True,
                output_format="auto"
            )
            
            # Image information
            st.markdown(f"""
            <div class="info-box">
                <strong>Image Info:</strong><br>
                📏 Dimensions: {image.size[0]} × {image.size[1]} pixels<br>
                🎨 Color Mode: {image.mode}<br>
                📄 Format: {uploaded_file.type}<br>
                💾 Size: {uploaded_file.size / 1024:.1f} KB
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("🎯 Prediction Results")
        
        if uploaded_file:
            # Process and predict
            processed_image = preprocess_image(image)
            
            if processed_image is not None:
                label, confidence, raw_prediction = predict_image(model, processed_image)
                
                if label:
                    # Result display
                    result_class = "cat-result" if label == "Cat" else "dog-result"
                    animal_icon = "🐱" if label == "Cat" else "🐶"
                    
                    st.markdown(f"""
                    <div class="result-card {result_class}">
                        <div class="prediction-text">
                            {animal_icon} This is a {label}! {animal_icon}
                        </div>
                        <div class="confidence-text">
                            Confidence: {confidence:.1%}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Confidence visualization
                    fig = create_confidence_visualization(raw_prediction)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Confidence interpretation
                    if confidence > 0.9:
                        st.markdown("""
                        <div class="info-box success-box">
                            <strong>🎯 Excellent Prediction!</strong><br>
                            Very high confidence - the model is quite certain about this classification.
                        </div>
                        """, unsafe_allow_html=True)
                    elif confidence > 0.7:
                        st.markdown("""
                        <div class="info-box">
                            <strong>✅ Good Confidence</strong><br>
                            Reliable prediction with good confidence level.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div class="info-box warning-box">
                            <strong>⚠️ Low Confidence</strong><br>
                            The image might be unclear, ambiguous, or contain both animals.
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="upload-card">
                <h3>👆 Upload an image to get started</h3>
                <p>Select a clear photo of a cat or dog for best results</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Additional information section
    st.markdown("---")
    
    # Model information and tips in expandable sections
    col1, col2 = st.columns(2)
    
    with col1:
        with st.expander("🤖 Model Information", expanded=False):
            st.markdown("""
            **Architecture:** Convolutional Neural Network (CNN)
            
            **Specifications:**
            - Input Resolution: 128×128 pixels
            - Color Channels: RGB (3 channels)
            - Output: Binary classification
            - Activation: Sigmoid (final layer)
            
            **Training Details:**
            - Dataset: Cats vs Dogs
            - Validation Accuracy: ~80%
            - Optimizer: Adam
            - Loss Function: Binary Crossentropy
            - Data Augmentation: Applied
            """)
    
    with col2:
        with st.expander("💡 Tips for Best Results", expanded=False):
            st.markdown("""
            **Image Quality:**
            - Use well-lit, clear images
            - Ensure the animal fills most of the frame
            - Avoid heavily filtered or edited photos
            
            **What Works Best:**
            - Single animal per image
            - Front or side view of the animal
            - High contrast with background
            - Minimal distracting elements
            
            **Avoid:**
            - Blurry or low-resolution images
            - Multiple animals in one image
            - Heavy shadows or poor lighting
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #718096; padding: 1rem;">
        <strong>AI Pet Classifier</strong> | Powered by TensorFlow & Streamlit<br>
        <small>Educational project - Results may vary based on image quality</small>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()