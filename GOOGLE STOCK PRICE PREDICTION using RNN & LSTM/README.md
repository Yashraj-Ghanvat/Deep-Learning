📈 Google Stock Price Prediction (RNN & LSTM)
This repository contains a Streamlit web application that predicts the next day's Google stock price using a Deep Learning model built with Recurrent Neural Networks (RNN) and Long Short-Term Memory (LSTM) layers.

🚀 Overview
Predicting stock prices is a complex time-series task. This application utilizes an LSTM architecture, which is specifically designed to remember long-term dependencies in sequential data, making it ideal for financial forecasting.

Key Features:
Dual Input Methods: Upload a .csv file containing historical data or manually paste stock prices.

Real-time Prediction: Uses a pre-trained .h5 model to generate predictions instantly.

Interactive UI: Clean and responsive interface built with Streamlit.

🛠️ Tech Stack
Python 3.x

Streamlit: For the web interface.

TensorFlow/Keras: For loading and running the LSTM model.

NumPy & Pandas: For data manipulation and preprocessing.

📋 Prerequisites
Before running the app, ensure you have the following installed:

Python 3.8 or higher

The trained model file (best_model.h5) located in your specified directory.

⚙️ Installation & Setup
Clone the repository:

Bash

git clone https://github.com/yourusername/google-stock-prediction.git
cd google-stock-prediction
Install dependencies:

Bash

pip install streamlit numpy pandas tensorflow
Configure Model Path: Open app.py and update the MODEL_PATH variable to point to the location of your best_model.h5 file:

Python

MODEL_PATH = r"C:\path\to\your\best_model.h5"
Run the application:

Bash

streamlit run app.py
🖥️ How to Use
Load the App: Once the Streamlit server starts, open the URL (usually http://localhost:8501).

Input Data: * Option A: Upload a CSV file that includes a column named Close. The app will automatically take the last 60 entries.

Option B: Manually paste 60 comma-separated stock prices into the text area.

Predict: Click the "Predict Next Day Price" button.

Result: The model will process the 3D tensor shape (1, 60, 1) and display the predicted price for the following trading day.

⚠️ Note on Predictions
This tool is for educational purposes only. Stock market prediction is inherently volatile, and model-based forecasts should not be used as financial advice.

Developed with ❤️ using Streamlit, RNN, and LSTM
