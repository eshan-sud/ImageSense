
# filename - src/script.py

# venv\scripts\activate
# pip install scikit-image scikit-learn matplotlib tensorflow
# pip freeze > src/requirements.txt
# streamlit run src/script.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Import custom modules
import image_processing
import face_recognition
import feature_extraction

def user_interface():
    st.set_page_config(page_title="ImageSense", layout="centered")
    st.title("ImageSense")
    st.caption("Image Processing & Pattern Analysis")

    # Sidebar
    st.sidebar.header("Navigation")
    option = st.sidebar.radio("Select Feature:", [
        "Image Pre-processing",
        "Edge Detection",
        "Feature Extraction",
        "Face Recognition"
    ])

    # Upload Image
    st.sidebar.info("Upload an image to get started.")
    image = upload_image()
    if image is not None:
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.sidebar.success("Image uploaded successfully!")
    else:
        st.warning("Please upload an image.")

    # Feature Selection
    if option == "Image Preprocessing":
        st.subheader("Image Preprocessing")
        col1, col2 = st.columns(2)
        with col1:
            grayscale = st.checkbox("Convert to Grayscale")
            blur = st.slider("Blur Level", 0, 10, 0)
        with col2:
            resize = st.checkbox("Resize Image")
            if resize:
                width = st.number_input("Width", 10, 1000, image.shape[1])
                height = st.number_input("Height", 10, 1000, image.shape[0])
        if st.button("Apply Changes"):
            processed_image = image_processing.preprocess(image, grayscale, blur, resize, (width, height))
            st.image(processed_image, caption="Processed Image", use_container_width=True)

    elif option == "Edge Detection":
        st.subheader("Edge Detection")
        method = st.selectbox("Select Edge Detection Method:", ["Canny", "Sobel", "Laplacian"])
        if st.button("Detect Edges"):
            edged_image = image_processing.edge_detection(image, method)
            st.image(edged_image, caption=f"{method} Edge Detection", use_container_width=True)

    elif option == "Feature Extraction":
        st.subheader("Feature Extraction")
        feature_type = st.selectbox("Select Feature Type:", ["SIFT", "HOG", "ORB"])
        if st.button("Extract Features"):
            feature_image = feature_extraction.extract(image, feature_type)
            st.image(feature_image, caption=f"{feature_type} Features", use_container_width=True)

    elif option == "Face Recognition":
        st.subheader("Face Recognition")
        detect_faces = st.button("Detect Faces")
        if detect_faces:
            face_image = face_recognition.detect_faces(image)
            st.image(face_image, caption="Detected Faces", use_container_width=True)
    
    elif option == "Segmentation":
        st.subheader("Image Segmentation")
        segmentation_method = st.selectbox("Choose a Segmentation Method", ["K-Means Clustering", "Watershed", "GrabCut"])
        if st.button("Segment Image"):
            segmented_image = image_processing.segment(image, segmentation_method)
            st.image(segmented_image, caption=f"Segmented using {segmentation_method}", use_column_width=True)

    elif option == "Classification":
        st.subheader("Image Classification")
        classifier_type = st.selectbox("Choose Classifier", ["KNN", "SVM", "CNN"])
        if st.button("Classify Image"):
            classified_label = image_processing.classify(image, classifier_type)
            st.success(f"Classified as: {classified_label}")

    st.sidebar.text("Developed with ❤️ by Eshan Sud")

def upload_image():
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg", "svg", "webp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        return np.array(image)
    return None

if __name__ == "__main__":
    user_interface()