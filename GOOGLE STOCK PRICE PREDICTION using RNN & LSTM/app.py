import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os

# ==============================
# 1. App Configuration
# ==============================
st.set_page_config(
    page_title="Google Stock Price Prediction",
    page_icon="📈",
    layout="centered"
)

st.title("📈 Google Stock Price Prediction using RNN & LSTM")
st.markdown("This app uses a trained LSTM model to predict Google stock prices.")

# ==============================
# 2. Load the trained model
# ==============================
MODEL_PATH = r"C:\Users\Admin\Desktop\Yashraj\Deep Learning\RNN & LSTM\GOOGLE STOCK PRICE PREDICTION using RNN & LSTM\best_model.h5"

@st.cache_resource
def load_trained_model(path):
    try:
        model = load_model(path)
        return model, None
    except Exception as e:
        return None, str(e)

model, error = load_trained_model(MODEL_PATH)

if error:
    st.error(f"❌ Error loading model: {error}")
    st.stop()
else:
    st.success("✅ Model loaded successfully!")

# ==============================
# 3. User Input Section
# ==============================
st.subheader("Enter Last 60 Days Stock Prices")
st.markdown("Provide the **last 60 days closing prices** to predict the next day's price.")

# Option 1: Upload CSV
uploaded_file = st.file_uploader("Upload CSV file with 'Close' column", type=["csv"])

# Option 2: Manual input (fallback)
manual_input = st.text_area("Or paste 60 comma-separated stock prices:", "")

# ==============================
# 4. Data Preparation
# ==============================
input_data = None

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if 'Close' in df.columns:
            input_data = df['Close'].values[-60:]
        else:
            st.error("CSV must contain a 'Close' column.")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")

elif manual_input:
    try:
        input_data = np.array([float(x) for x in manual_input.split(",")])
        if len(input_data) != 60:
            st.error("Please provide exactly 60 values.")
            input_data = None
    except Exception as e:
        st.error(f"Invalid input format: {e}")

# ==============================
# 5. Prediction
# ==============================
if input_data is not None and st.button("Predict Next Day Price"):
    try:
        # Reshape for LSTM [samples, timesteps, features]
        scaled_input = np.array(input_data).reshape(1, 60, 1)
        prediction = model.predict(scaled_input)
        st.success(f"📊 Predicted Next Day Stock Price: **${prediction[0][0]:.2f}**")
    except Exception as e:
        st.error(f"❌ Prediction error: {e}")

# ==============================
# 6. Footer
# ==============================
st.markdown("---")
st.caption("Developed with ❤️ using Streamlit, RNN, and LSTM")
