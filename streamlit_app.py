import streamlit as st
import numpy as np
import tensorflow as tf # type: ignore
from tensorflow.keras.preprocessing import image # type: ignore
import pickle
from PIL import Image

# Load the trained CNN model
model = tf.keras.models.load_model('cnn_thyroid_model.h5')

# Load the label encoder
with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Function to preprocess the image
def preprocess_image(img):
    img = img.resize((128, 128))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array

# Title of the app
st.title('Thyroid Nodule Classification')

# Instructions
st.write("""
    Upload a thyroid ultrasound image and get the classification result (Benign or Malignant).
""")

# Image upload widget
uploaded_image = st.file_uploader("Upload a thyroid ultrasound image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the image
    img = Image.open(uploaded_image)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess the image
    processed_image = preprocess_image(img)

    # Predict the class
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions, axis=1)
    class_label = label_encoder.inverse_transform(predicted_class)[0]

    # Display the prediction result
    if class_label == 'benign':
        st.success(f"The thyroid nodule is **Benign** (Confidence: {predictions[0][predicted_class]*100:.2f}%)")
    elif class_label == 'malignant':
        confidence = float(predictions[0][predicted_class]) * 100
        st.error(f"The thyroid nodule is *Malignant* (Confidence: {confidence:.2f}%)")
    else:
        st.warning(f"Unknown class: {class_label}")

