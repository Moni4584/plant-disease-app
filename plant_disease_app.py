import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------
# Load the trained model
# -----------------------
@st.cache_resource
def load_cnn_model():
    model = tf.keras.models.load_model("plant_disease_cnn_model12.keras")
    return model

model = load_cnn_model()

# -----------------------
# Define class names
# -----------------------
# ⚠️ Replace with your actual dataset class names in correct order
# Order should match your dataset classes
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


# -----------------------
# Streamlit App Layout
# -----------------------
st.title("🍃 Plant Disease Detection App")
st.write("Upload a plant leaf image and the model will predict the disease.")

# File uploader
uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Show uploaded image
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Leaf Image", use_column_width=True)

    # -----------------------
    # Preprocess the image
    # -----------------------
    img = img.resize((224, 224))  # Change (224,224) if your model used another input size
    img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -----------------------
    # Make prediction
    # -----------------------
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # -----------------------
    # Display result
    # -----------------------
    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
