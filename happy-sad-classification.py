import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

model = load_model("imageclassifier.h5")

st.set_page_config(page_title="Happy - Sad")
st.title("Exploring Your Emotions: Happy or Sad?")
st.caption("Upload your image and let me guess...!")

upload_image = st.file_uploader("Choose an image", ['jpeg', 'jpg', 'bmp', 'png'])

if upload_image is not None:
    image = Image.open(upload_image) 
    image_array = np.array(image)  
    st.success("Image uploaded successfully.")
    st.image(image, caption="Uploaded Image") 
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)  
    resized_image = tf.image.resize(image_tensor, (256, 256))  
    img = np.expand_dims(resized_image / 255, 0)
    output = model(img)
    if output > 0.5:
        st.info(f'Predicted class is Sad')
    else:
        st.info(f'Predicted class is Happy')

else:
    st.write("Please upload an image to proceed.")


sample_images = [
    ("Picture1", "images490.jpg"),
    ("Picture2", "images73.jpg"),
    ("Picture3", "images624.jpg"),
    ("Picture4", "images91.jpg")
]

st.subheader("No image? Try these sample images:")


sample_image_paths = [image[1] for image in sample_images]
sample_labels = [image[0] for image in sample_images]


cols = st.columns(4) 

selected_image = st.radio("Select an image to predict:", options=sample_image_paths, format_func=lambda x: sample_labels[sample_image_paths.index(x)])


selected_image_idx = sample_image_paths.index(selected_image)
image = Image.open(selected_image)

st.image(image.resize((100, 100)), caption=f"Selected Image: {sample_labels[selected_image_idx]}")


if st.button("Submit"):
    image_array = np.array(image)  
    image_tensor = tf.convert_to_tensor(image_array, dtype=tf.float32)  
    resized_image = tf.image.resize(image_tensor, (256, 256))  
    img = np.expand_dims(resized_image / 255, 0)
    output = model(img)
    if output > 0.5:
        st.info(f'Predicted class is Sad')
    else:
        st.info(f'Predicted class is Happy')
