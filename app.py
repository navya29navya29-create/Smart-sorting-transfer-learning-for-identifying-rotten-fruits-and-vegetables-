import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

model = load_model('model/fruit_classifier.h5')

def predict(img_path):
    img = Image.open(img_path).resize((224, 224))
    img_arr = image.img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    prediction = model.predict(img_arr)
    return "Rotten" if prediction < 0.5 else "Fresh"

st.title("Smart Sorting - Rotten or Fresh?")
uploaded = st.file_uploader("Upload an image", type=["jpg", "png"])

if uploaded:
    result = predict(uploaded)
    st.image(uploaded, caption='Uploaded Image', use_column_width=True)
    st.success(f"This fruit/vegetable is **{result}**")
