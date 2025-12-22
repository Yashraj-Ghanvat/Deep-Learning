Glass Detection in Images using CNN
This project implements a Deep Learning solution to classify images based on whether a person is wearing glasses or not. It leverages Convolutional Neural Networks (CNN) and Transfer Learning to achieve high classification accuracy.

📌 Project Overview
Detecting glasses in facial images is a common preprocessing step for facial recognition systems and augmented reality applications. This notebook explores the complete pipeline from data exploration and preprocessing to model training and evaluation using TensorFlow and Keras.

📊 Dataset
The dataset is structured into training, validation, and testing sets, categorized into two classes:

Glasses (Yes): Images of individuals wearing eyewear.

No Glasses (No): Images of individuals without eyewear.

The project includes data visualization scripts to inspect sample images and ensure balanced data representation.

🛠️ Tech Stack
Language: Python 3

Deep Learning: TensorFlow, Keras

Data Manipulation: NumPy, Pandas

Computer Vision: OpenCV (cv2)

Visualization: Matplotlib, Seaborn

Model Architectures used/referenced:

Sequential CNN

EfficientNetB4

Xception

ResNet50

🚀 Key Features
Image Augmentation: Uses ImageDataGenerator for robust training against variations in lighting and orientation.

Transfer Learning: Implements state-of-the-art architectures (EfficientNet, ResNet) to leverage pre-trained weights.

Model Callbacks: Includes EarlyStopping and ModelCheckpoint to prevent overfitting and save the best performing model.

Performance Metrics: Evaluates models using Accuracy, Confusion Matrices, and training/validation loss curves.

💻 Installation & Usage
Prerequisites
Ensure you have Python 3.x installed along with the following libraries:

Bash

pip install tensorflow numpy pandas matplotlib seaborn opencv-python scikit-learn
Running the Project
Clone the repository:

Bash

git clone https://github.com/yourusername/glass-detection-cnn.git
Open the Jupyter Notebook:

Bash

jupyter notebook Glass_Detection_in_image.ipynb
Update the train_dir and val_dir paths in the notebook to point to your local dataset directory.

📈 Results
The project provides detailed plots for:

Training vs. Validation Accuracy.

Training vs. Validation Loss.

Confusion Matrix for classification performance.

🤝 Contributing
Contributions are welcome! If you have suggestions for improving model performance or adding new features, please open an issue or submit a pull request.
