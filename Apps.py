import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from PIL import Image
import os

# Path to the folder containing your model
MODEL_DIR = MODEL_DIR = r"C:\Users\LENOVO\OneDrive\Documents\ANN_LAB_Project"
  # Update this with the actual path
MODEL_PATH = os.path.join(MODEL_DIR, "mulberry_leaf_disease_model_enhanced5.h5")

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)

# Class labels
all_labels = ['Healthy_Leaves', 'Rust_leaves', 'Spot_leaves', 'deformed_leaves', 'Yellow_leaves']

# Streamlit interface
st.title("Mulberry Leaf Disease Detection")

st.write("""
Upload an image of a Mulberry leaf, and the model will predict whether it's healthy or has a disease.
""")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Load the image
    img = Image.open(uploaded_file)
    
    # Preprocess the image for the model
    img = img.resize((256, 256))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img.astype('float32') / 255.0
    
    # Make prediction
    prediction = model.predict(img)
    predicted_label = all_labels[np.argmax(prediction)]

    # Display the result
    st.write(f"**Prediction:** The leaf is likely to have `{predicted_label}`.")

    # Handle case where uploaded image is not a leaf
    st.write("If the uploaded image is not a Mulberry leaf, the prediction might not be accurate.")
