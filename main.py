import streamlit as st
import tensorflow as tf

st.set_page_config(page_title="Keras to TFLite Converter", layout="centered")

st.title("🌱 Plant Disease Model Converter")
st.write("Upload a Keras `.keras` model and get a quantized TFLite model!")

# Upload Keras model
uploaded_model = st.file_uploader("Choose Keras model file (.keras)", type=["keras"])

if uploaded_model is not None:
    st.success(f"Uploaded: {uploaded_model.name}")

    # Load Keras model
    try:
        model = tf.keras.models.load_model(uploaded_model)
        st.success("✅ Keras model loaded successfully!")
    except Exception as e:
        st.error(f"Failed to load model: {e}")

    # Convert to TFLite with quantization
    if st.button("Convert to Quantized TFLite"):
        try:
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            tflite_model = converter.convert()

            # Save to BytesIO for download
            import io
            tflite_file = io.BytesIO(tflite_model)

            st.download_button(
                label="Download Quantized TFLite Model",
                data=tflite_file,
                file_name="plant_disease_model_quant.tflite",
                mime="application/octet-stream"
            )

            st.success("✅ Conversion successful!")
        except Exception as e:
            st.error(f"Conversion failed: {e}")
