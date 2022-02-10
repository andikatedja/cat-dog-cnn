import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import tensorflow as tf
import numpy as np
from PIL import Image

st.write("""
# Cat and Dog Prediction  
Using mobilenet_v2 model with global_average_pooling  

---
""")

img = st.file_uploader(label='Upload an image (cat or dog)', type=['png', 'jpg'])
if img:
    with st.spinner("Please wait..."):
        model = load_model('mobilenet_v2.h5')

        im = Image.open(img)
        im = im.resize((224, 224))
        im = np.array(im)
        im = np.array([im])
        im = preprocess_input(im)

        results = {
            0:'Cat',
            1:'Dog'}

        pred = np.argmax(model.predict(im), axis=-1)
        st.success(f'It is a **{results[pred[0]]}**!')
        st.image(img)

