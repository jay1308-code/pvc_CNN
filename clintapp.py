import streamlit as st
from tensorflow import keras
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained model
model = keras.models.load_model("PVC.h5")

st.set_page_config(page_title="PVC-Classification",layout="wide")

st.title("PVC Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Preprocess the image
    img = image.load_img(uploaded_file, target_size=(150, 150))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.

    # Make a prediction
    prediction = model.predict(img)
    

    # Display the result
    if prediction[0][0] > 0.5:
        st.image(uploaded_file, caption='PVC', use_column_width=True)
    else:
        st.image(uploaded_file, caption='NOT PVC', use_column_width=True)
       
