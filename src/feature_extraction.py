
# filename - src/feature_extraction.py

import cv2
import numpy as np
# from skimage.feature import hog
import streamlit as st


def extract(image, method):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    if method == "SIFT":
        sift = cv2.SIFT_create()
        keypoints, descriptors = sift.detectAndCompute(gray, None)
        return cv2.drawKeypoints(image, keypoints, None)
    
    elif method == "ORB":
        orb = cv2.ORB_create()
        keypoints, descriptors = orb.detectAndCompute(gray, None)
        return cv2.drawKeypoints(image, keypoints, None)
    
    elif method == "HOG":
        pass
        # features, _ = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        # return features
    
    return image

# Streamlit UI Integration
# st.title("Feature Extraction")
# uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])
# method = st.selectbox("Select Feature Extraction Method", ["SIFT", "ORB", "HOG"])

# if uploaded_file is not None:
#     file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
#     image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
#     extracted_image = extract(image, method)
#     st.image(extracted_image, channels="GRAY" if method == "HOG" else "BGR")
