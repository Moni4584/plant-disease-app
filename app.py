import streamlit as st
from tensorflow import keras
from PIL import Image
import numpy as np

# Load the model
model = keras.models.load_model("plant_disease_cnn_model12.keras")

# Map class indices to actual plant disease names
class_names = ["Healthy", "Early Blight", "Late Blight", "Leaf Mold", "Powdery Mildew"]

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
