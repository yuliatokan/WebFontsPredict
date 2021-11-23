# -*- coding: utf-8 -*-

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

def import_and_predict(image_data, model):
    
        size = (20,20)    
        image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
        image = image.convert('RGB')
        image = np.asarray(image)
        image = (image.astype(np.float32) / 255.0)

        img_reshape = np.reshape(image, (-1,20,20,1))

        prediction = model.predict(img_reshape[:1])
        
        fonts=[]
        with open('listfile.txt', 'r') as filehandle:  
          fonts = [current_font.rstrip() for current_font in filehandle.readlines()]
        return fonts[np.argmax(prediction)]

model = tf.keras.models.load_model('my_model.hdf5')

st.write("""
         # Fonts Prediction
         """
         )

st.write("This is a simple image classification web app to fonts predict")

file = st.file_uploader("Please upload an image file .jpg or .png", type=["jpg", "png"])
#
if file is None:
    st.text("You haven't uploaded an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    
    st.text("Probaby, it is ")
    st.write(prediction)