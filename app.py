import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np

# Load the trained model
model = load_model('plant_disease_detector.h5')

# Function to preprocess image
def preprocess_image(image):
    img = image.resize((128, 128))  # Resize image to match the model input size
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    img = img / 255.0  # Normalize image
    return img

# Prediction function
def predict_disease(image):
    processed_img = preprocess_image(image)
    prediction = model.predict(processed_img)

    # Get the probability from the model's output
    probability = prediction[0][0]

    # Check if the probability indicates a disease (assuming 0 is healthy, 1 is diseased)
    if probability > 0.5:
        return 'Healthy'
    else:
        return 'Not Healthy'


# Streamlit App Interface
st.title("Plant Disease Detection")
st.write("Upload an image of a plant leaf to detect if it is healthy or diseased.")

# File uploader for image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Predict button
    if st.button("Predict"):
        result = predict_disease(image)
        st.write(f"Prediction: {result}")
