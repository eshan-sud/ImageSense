
# filename - src/script.py

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import time
from PIL import Image
import io

# Imports the custom modules
import image_processing
import face_recognition
import feature_extraction

# Displays a temporary SUCCESS message
def show_temp_success(success):
    success_msg = st.success(success)
    time.sleep(3)
    success_msg.empty()

# Displays a temporary ERROR message
def show_temp_error(error):
    error_msg = st.error(error)
    time.sleep(3)
    error_msg.empty()

# Displays a temporary WARNING message
def show_temp_warning(warning):
    warning_msg = st.warning(warning)
    time.sleep(3)
    warning_msg.empty()

# Image is checked
def image_exists(img):
    return img is not None

# Image is saved on device
def save_image(img, format):
    pil_img = Image.fromarray(img)
    img_byte_arr = io.BytesIO()
    pil_img.save(img_byte_arr, format=format)
    img_byte_arr.seek(0)
    return img_byte_arr

# User Interface is created
def user_interface():
    # Page Configiration
    st.set_page_config(
        page_title="ImageSense",
        page_icon="🧊",
        layout="centered",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://google.com", # CHANGE THIS
            "Report a bug": "mailto:eshansud22@gmail.com",
            "About": "This is an application to perform image processing"
        }
    )
    st.title("ImageSense")
    st.caption("Image Processing & Pattern Analysis")

    # Sidebar
    st.sidebar.header("Navigation")
    option = st.sidebar.radio("Select Feature:", [
        "Image Preprocessing",
        "Edge Detection",
        "Feature Extraction",
        "Face Recognition"
    ])

    # Upload Image
    flag = None
    image, image_name = upload_image()
    if image_exists(image):
        new_name = st.text_input("Rename the uploaded image", value=image_name).lower()
        if new_name != image_name:
            show_temp_success(f"Image renamed to: {new_name}")
        st.image(image, caption=f"Uploaded Image : {new_name}")
        if flag:
            show_temp_success("Image uploaded successfully!")
            flag = False

        # Feature Selection
        if option == "Image Preprocessing":
            st.header("Image Preprocessing")
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
            st.header("Edge Detection")
            method = st.selectbox("Select Edge Detection Method:", ["Canny", "Sobel", "Laplacian"])
            if st.button("Detect Edges"):
                edged_image = image_processing.edge_detection(image, method)
                st.image(edged_image, caption=f"{method} Edge Detection", use_container_width=True)

        elif option == "Feature Extraction":
            st.header("Feature Extraction")
            feature_type = st.selectbox("Select Feature Type:", ["SIFT", "HOG", "ORB"])
            if st.button("Extract Features"):
                feature_image = feature_extraction.extract(image, feature_type)
                st.image(feature_image, caption=f"{feature_type} Features", use_container_width=True)

        elif option == "Face Recognition":
            st.header("Face Recognition")
            detect_faces = st.button("Detect Faces")
            if detect_faces:
                face_image = face_recognition.detect_faces(image)
                st.image(face_image, caption="Detected Faces", use_container_width=True)

        elif option == "Segmentation":
            st.header("Image Segmentation")
            segmentation_method = st.selectbox("Choose a Segmentation Method", ["K-Means Clustering", "Watershed", "GrabCut"])
            if st.button("Segment Image"):
                segmented_image = image_processing.segment(image, segmentation_method)
                st.image(segmented_image, caption=f"Segmented using {segmentation_method}", use_column_width=True)

        elif option == "Classification":
            st.header("Image Classification")
            classifier_type = st.selectbox("Choose Classifier", ["KNN", "SVM", "CNN"])
            if st.button("Classify Image"):
                classified_label = image_processing.classify(image, classifier_type)
                show_temp_success(f"Classified as: {classified_label}")

        # Save Image
        st.header("Save image")
        format = st.radio("Select Format:", [
            "JPEG",
            "PNG",
            "WEBP",
            "BMP",
            "SVG"
        ])
        img_byte_arr = save_image(image, format=format)
        st.download_button(
            label="Download Image",
            data=img_byte_arr,
            file_name=f"{new_name}.{format.lower()}",
            mime=f"image/{format.lower()}"
        )
    else:
        flag = True
        st.sidebar.warning("Upload an image to get started.")
    st.sidebar.text("Developed with ❤️ by Eshan Sud")

# Iamge is first uploaded & then processing is performed
def upload_image():
    uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg", "svg", "webp"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_name = uploaded_file.name
        return np.array(image), image_name
    return None, None

# Program starts from here
if __name__ == "__main__":
    user_interface()