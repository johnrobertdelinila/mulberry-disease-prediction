import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from PIL import Image
import os

# Configure Streamlit page
st.set_page_config(
    page_title="Mulberry Disease Detection",
    page_icon="🌿",
    layout="wide"
)

# Path to the model (relative to the app)
MODEL_PATH = "Model/mulberry_leaf_disease_model_enhanced.h5"

# Load the trained model with error handling
@st.cache_resource
def load_model():
    """Load the trained model with caching"""
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model file not found at: {MODEL_PATH}")
            st.info("Please make sure you have trained the model first using train_model.py")
            return None
        
        model = tf.keras.models.load_model(MODEL_PATH)
        st.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

# Load the model
model = load_model()

# Class labels
all_labels = ['Healthy_Leaves', 'Rust_leaves', 'Spot_leaves', 'deformed_leaves', 'Yellow_leaves']

# Streamlit interface
st.title("🌿 Mulberry Leaf Disease Detection")
st.markdown("---")

# Sidebar with information
with st.sidebar:
    st.header("📋 About")
    st.write("""
    This app uses a deep learning model to classify mulberry leaf diseases:
    - **Healthy Leaves**
    - **Rust Leaves** 
    - **Spot Leaves**
    - **Deformed Leaves**
    - **Yellow Leaves**
    """)
    
    st.header("🎯 Instructions")
    st.write("""
    1. Upload a clear image of a mulberry leaf
    2. The model will analyze the image
    3. Get instant disease classification
    """)

# Main content
st.subheader("📤 Upload Image")
uploaded_file = st.file_uploader(
    "Choose an image of a mulberry leaf...", 
    type=["jpg", "png", "jpeg"],
    help="Supported formats: JPG, PNG, JPEG"
)

if uploaded_file is not None:
    # Create two columns for image and results
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("🖼️ Uploaded Image")
        st.image(uploaded_file, caption='Your uploaded image', use_column_width=True)
    
    with col2:
        st.subheader("🔍 Analysis Results")
        
        if model is not None:
            try:
                # Load and preprocess the image
                img = Image.open(uploaded_file)
                img = img.resize((256, 256))
                img_array = img_to_array(img)
                img_array = np.expand_dims(img_array, axis=0)
                img_array = img_array.astype('float32') / 255.0
                
                # Make prediction
                with st.spinner("🔮 Analyzing image..."):
                    prediction = model.predict(img_array, verbose=0)
                
                # Get results
                predicted_class_idx = np.argmax(prediction[0])
                confidence = prediction[0][predicted_class_idx] * 100
                predicted_label = all_labels[predicted_class_idx]
                
                # Display prediction with confidence
                st.success(f"**Prediction: {predicted_label}**")
                st.metric("Confidence", f"{confidence:.1f}%")
                
                # Show all probabilities
                st.subheader("📊 All Probabilities")
                prob_df = {
                    'Disease Type': all_labels,
                    'Probability (%)': [f"{p*100:.1f}" for p in prediction[0]]
                }
                st.bar_chart({label: float(prob) for label, prob in zip(all_labels, prediction[0])})
                
                # Interpretation
                if confidence > 80:
                    st.success("✅ High confidence prediction")
                elif confidence > 60:
                    st.warning("⚠️ Medium confidence prediction")
                else:
                    st.error("❌ Low confidence prediction - image may be unclear")
                
            except Exception as e:
                st.error(f"❌ Error processing image: {e}")
                st.info("Please try uploading a different image")
        else:
            st.error("❌ Model not loaded. Please train the model first using train_model.py")

# Footer
st.markdown("---")
st.markdown("**Note:** This model works best with clear, well-lit images of mulberry leaves. Results may vary with image quality.")
