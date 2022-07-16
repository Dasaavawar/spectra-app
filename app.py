import streamlit as st
import pandas as pd
import numpy as np
import cv2 as cv
import csv as csv
import sys as sys

from datetime import datetime

st.title("Spectrum CSV and plot")
st.subheader("Upload an image to display its spectrum")
image = st.file_uploader("Choose a image file", type=["jpg", "jpeg", "png"])

#Convert df pd to csv
def convert_df_to_csv(df):
  return df.to_csv(index=False)

if image is not None:
    # Convert the file to an opencv image:
    fbytes = np.asarray(bytearray(image.read()), dtype=np.uint8)
    cvimage = cv.imdecode(fbytes, 1)

    # Display image and values:
    st.subheader("Image uploaded:")
    st.image(cvimage, channels="BGR")
    
    h, w, _ = cvimage.shape
    st.subheader("Image size:")
    st.text("height;")
    st.text(h)
    st.text("width;")
    st.text(w)
    
    #Calculate its spectrum
    spec = np.empty((0))
    for x in range(0,w):
        [r,g,b]=cvimage[255,x]
        intensity = (int(r)+int(g)+int(b))/3
        spec = np.append(spec, intensity)
    
    #Set print options and print in Webapp
    np.set_printoptions(3)
    np.set_printoptions(threshold=sys.maxsize)
    
    specd = pd.DataFrame(spec)
    specdf = specd.round(decimals=3)

    st.subheader("Image spectrum plotted:")
    st.line_chart(specdf)

    st.subheader("Image spectrum data:")
    st.text(spec)

    st.subheader("Export spectrum data as a CSV file:")
    st.download_button(
        label="Download data as CSV",
        data=convert_df_to_csv(specdf),
        file_name=image.name+".csv",
        mime='text/csv',
        )
