




import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import requests

# -------------------------------
# Download model from Google Drive
# -------------------------------
MODEL_PATH = "plant_disease_cnn_model12.keras"
URL = "https://drive.google.com/uc?export=download&id=1aBcdEfGhIjklMnOpq"  # Replace with your file ID

if not os.path.exists(MODEL_PATH):
    st.info("Downloading model, please wait...")
    response = requests.get(URL)
    with open(MODEL_PATH, "wb") as f:
        f.write(response.content)
    st.success("Model downloaded!")

# -------------------------------
# Load the trained model
# -------------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------------
# Define class names
# -------------------------------
class_names = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___Cedar_apple_rust",
    "Apple___healthy",
    "Blueberry___healthy",
    "Cherry___healthy",
    "Cherry___Powdery_mildew",
    "Corn___Cercospora_leaf_spot_Gray_leaf_spot",
    "Corn___Common_rust",
    "Corn___healthy",
    "Corn___Northern_Leaf_Blight",
    "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)",
    "Grape___healthy",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
    "Orange___Haunglongbing_(Citrus_greening)",
    "Peach___Bacterial_spot",
    "Peach___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
    "Potato___Early_blight",
    "Potato___healthy",
    "Potato___Late_blight",
    "Raspberry___healthy",
    "Rice-Bacterial leaf blight",
    "Rice-Brown spot",
    "Rice-Leaf smut",
    "Soybean___healthy",
    "Squash___Powdery_mildew",
    "Strawberry___healthy",
    "Strawberry___Leaf_scorch",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___healthy",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites_Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus"
]

# -------------------------------
# Streamlit App
# -------------------------------
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image and the app will predict the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Leaf', use_column_width=True)
    
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.write(f"**Prediction:** {predicted_class}")
    st.write(f"**Confidence:** {confidence*100:.2f}%")
