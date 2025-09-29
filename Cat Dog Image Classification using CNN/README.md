# 🐱🐶 Cat vs Dog Image Classification using CNN

This project is a **Cat vs Dog Image Classification Web App** built with **Streamlit** and a **Convolutional Neural Network (CNN)** trained on image data.
You can upload an image, and the model will predict whether it’s a **Cat** or a **Dog**.

---

## 🚀 Features

* Upload an image (`.jpg`, `.jpeg`, `.png`)
* Real-time prediction using trained CNN model
* Simple and interactive Streamlit web interface

---

## 📂 Project Structure

```
├── app.py                # Streamlit app file
├── cifar10_model.h5      # Trained CNN model
├── requirements.txt      # Dependencies
└── README.md             # Project documentation
```

---

## ⚙️ Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/your-username/cat-dog-classification.git
   cd cat-dog-classification
   ```

2. Create a virtual environment (optional but recommended):

   ```bash
   python -m venv venv
   source venv/bin/activate     # For Linux/Mac
   venv\Scripts\activate        # For Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

---

## ▶️ Usage

1. Run the Streamlit app:

   ```bash
   streamlit run app.py
   ```

2. Upload an image of a cat 🐱 or a dog 🐶.

3. Get the prediction instantly!

---

## 🛠 Requirements

Add this to your `requirements.txt`:

```
streamlit
tensorflow
numpy
pillow
```

---

## 📊 Model Details

* Architecture: **Convolutional Neural Network (CNN)**
* Framework: **TensorFlow / Keras**
* Output: **Binary Classification (Cat / Dog)**

---

## 📌 Example

* Input: Cat image
* Output: **Prediction → Cat**

---

## ✨ Future Improvements

* Extend to multi-class classification
* Deploy on cloud platforms (Heroku, Streamlit Cloud, etc.)
* Improve model accuracy with more training data

---

## 👨‍💻 Author

Developed by **Yashraj Ghanvat**
