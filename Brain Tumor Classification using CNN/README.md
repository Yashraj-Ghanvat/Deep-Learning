🧠 Brain Tumor Classification Using CNN (VGG16)

A Streamlit-based web application for classifying brain tumor types from MRI images using deep learning (VGG16 architecture).
This project is intended for educational and demonstration purposes only.

📌 Project Overview

Brain tumor detection is a critical application of deep learning in medical imaging.
This application allows users to upload MRI images and classify them into predefined tumor categories using pre-trained CNN models.

🔍 Supported Classes

No Tumor

Glioma

Meningioma

Pituitary

🚀 Features

Upload MRI images (.jpg, .jpeg, .png)

Choose between multiple trained CNN models

Real-time tumor classification

Confidence score for predictions

Probability distribution visualization

User-friendly Streamlit interface

Modular and scalable code structure

🛠️ Tech Stack

Programming Language: Python

Framework: Streamlit

Deep Learning: TensorFlow / Keras

Model Architecture: VGG16 (transfer learning)

Image Processing: NumPy, Keras preprocessing

📂 Project Structure
├── app.py                     # Main Streamlit application
├── brain_tumor_model_new.h5   # Trained CNN model
├── brain_tumor_model.keras    # Alternative saved model
├── model.h5                   # Backup trained model
├── README.md                  # Project documentation

⚙️ Installation & Setup
1️⃣ Clone the Repository
git clone https://github.com/your-username/brain-tumor-classification.git
cd brain-tumor-classification

2️⃣ Install Dependencies
pip install streamlit tensorflow numpy

3️⃣ Update Model Paths (Important)

In app.py, update the model paths to match your local or deployment directory:

MODEL_PATHS = {
    "Model (brain_tumor_model_new.h5)": "path/to/brain_tumor_model_new.h5",
    "Model (brain_tumor_model.keras)": "path/to/brain_tumor_model.keras",
    "Model (model.h5)": "path/to/model.h5"
}

▶️ How to Run the Application
streamlit run app.py


Once started, open the browser link shown in the terminal (usually http://localhost:8501).

🧪 How It Works

User uploads an MRI image

Image is resized to 224 × 224 and normalized

Selected CNN model processes the image

Model predicts probabilities for each tumor class

Final prediction and confidence score are displayed

📊 Output Example

Predicted Class: Glioma

Confidence: 92.45%

Visualization: Bar chart showing probability distribution

⚠️ Disclaimer

This application is not intended for medical diagnosis.
It is strictly for academic, learning, and demonstration purposes.

👨‍💻 Author

Yashraj Ghanvat
Deep Learning & Software Engineering Enthusiast
