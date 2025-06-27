# GrainPalette  - Rice Type Classification Using Deep Learning

GrainPalette is a deep learning-based project that classifies different types of rice grains using image recognition. Powered by transfer learning and TensorFlow, this model accurately predicts rice varieties based on visual features and offers a clean user interface through Streamlit.

##  Project Highlights

-  Classifies 5 rice types: **Arborio, Basmati, Ipsala, Jasmine, Karacadag**
-  Built using a pre-trained CNN model (transfer learning)
-  Interactive web app using Streamlit for real-time predictions
-  Supports uploading rice grain images and gives prediction with confidence score

---

##  Tech Stack

- **Python**
- **TensorFlow / Keras**
- **NumPy / Pillow**
- **Streamlit** for the web app interface

---

## 📁 Project Structure
GrainPalette/
│
├── Rice_Image_Dataset/ # Dataset folder with 5 rice classes
│ ├── Arborio/
│ ├── Basmati/
│ ├── Ipsala/
│ ├── Jasmine/
│ └── Karacadag/
│
├── Rice_Citation_Request.txt # Dataset license/citation info
├── app.py # Streamlit web app for user prediction
├── predict.py # Script to test the model on a local image
├── rice_type_model.py # Script to train the rice classification model
├── testmodel.py # Optional/test script
├── rice_type_model.h5 # Saved trained model file
└── README.md # Project documentation

## Install Dependencies

pip install tensorflow streamlit pillow numpy

## If you want to train from scratch:

python rice_type_model.py

##  Run the Streamlit App

streamlit run app.py

## Run a Prediction Script
Edit image path in predict.py and run:

python predict.py

## Acknowledgements

This project was developed as part of the SmartInternz Internship Program.
Special thanks to mentors and SmartBridge for guidance and support.

