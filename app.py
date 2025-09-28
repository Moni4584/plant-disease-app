from tensorflow import keras

# If the file is in the same folder
model = keras.models.load_model("plant_disease_cnn_model12.keras")

# Or full path
# model = keras.models.load_model("C:/Users/Monica/Downloads/plant_disease_cnn_model12.keras")

import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np

# Load the model
model = keras.models.load_model("plant_disease_cnn_model12.keras")

# Map class indices to actual plant disease names
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
def predict_image(img):
    img = img.resize((224, 224))  # Adjust to your model input size
    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction)
    return class_names[class_idx], float(np.max(prediction))

# Streamlit UI
st.title("🌿 Plant Disease Detection App")
st.write("Upload a leaf image and the app will predict the disease.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)
    
    # Prediction
    disease, confidence = predict_image(img)
    st.success(f"Predicted Disease: {disease}")
    st.info(f"Confidence: {confidence*100:.2f}%")
