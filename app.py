import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf

# ----------------------------
# 1️⃣ App Title
# ----------------------------
st.title("🌿 Plant Disease Detection")
st.write("Upload a leaf image and the model will predict the disease.")

# ----------------------------
# 2️⃣ Class Names
# ----------------------------
# Order should match your dataset classes
class_labels = [
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


# ----------------------------
# 3️⃣ Load TFLite Model
# ----------------------------
interpreter = tf.lite.Interpreter(model_path="plant_disease_model_quant.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------------------
# 4️⃣ Preprocess Image
# ----------------------------
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img, dtype=np.float32)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ----------------------------
# 5️⃣ Predict Function
# ----------------------------
def predict(img):
    img_array = preprocess_image(img)
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])
    class_index = np.argmax(output)
    confidence = np.max(output) * 100
    return class_names[class_index], confidence

# ----------------------------
# 6️⃣ File Uploader
# ----------------------------
uploaded_file = st.file_uploader("Choose a leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")
    disease, confidence = predict(image)
    st.success(f"Predicted Disease: **{disease}**")
    st.info(f"Confidence: **{confidence:.2f}%**")
