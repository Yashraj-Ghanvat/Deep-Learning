import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# CIFAR-10 classes
class_names = [
    "Airplane", "Automobile", "Bird", "Cat", "Deer",
    "Dog", "Frog", "Horse", "Ship", "Truck"
]

# Load model once and cache
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(
        r"C:\Users\Admin\Desktop\Yashraj\Deep Learning\CIFER10 classification using CNN\cifar10_model.h5"
    )
    return model

model = load_model()

st.title("🚀 CIFAR-10 Image Classification using CNN")

# Upload image
uploaded_file = st.file_uploader("Upload an image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Preprocess
    img_resized = image.resize((32, 32))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    st.subheader(f"Prediction: {class_names[predicted_class]}")
    st.write(f"Confidence: {confidence:.2f}")

    # Show probabilities as bar chart
    st.subheader("Class Probabilities")
    fig, ax = plt.subplots()
    ax.bar(class_names, prediction[0])
    ax.set_ylabel("Probability")
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45)
    st.pyplot(fig)
