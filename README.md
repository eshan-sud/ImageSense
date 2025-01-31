# ImageSense

A powerful platform for image processing & recognition.

## Software

- Python 3.13
- Streamlit
- OpenCV2

## Features & Details:

1. User Interface

   [x] Basic UI

   [x] Upload the image (JPG, PNG, SVG, WEBp)

2. Image Preprocessing

   [x] Display the uploaded image

   [] Convert to grayscale & histogram equalization

   [] Resize, crop, and rotate images

   [] Apply smoothing & sharpening filters (Gaussian, Median, Bilateral)

3. Edge Detection & Segmentation

   [] Edge detection using Sobel, Prewitt, Laplacian, Canny

   [] Image thresholding (Global, Adaptive, Otsu's method)

   [] Segmentation using K-Means, Watershed, GrabCut

   [] Morphological operations (Erosion, Dilation, Opening, Closing)

4. Feature Extraction

   [] Extract & visualize key features using SIFT, SURF, ORB, HOG

   [] PCA-based dimensionality reduction visualization

   [] Compute texture features (GLCM, LBP)

   [] Histogram-based feature analysis

5. Pattern Recognition & Classification

   [] Train & test classifiers on extracted features

   [] Support for KNN, SVM, Decision Trees, CNN (Pretrained Models)

   [] Upload custom datasets for classification

   [] Evaluate models with accuracy, precision, recall

6. Face Detection & Recognition

   [] Detect faces using Haar Cascades, DNN (ResNet, MobileNet)

   [] Recognize faces using LBPH, Eigenfaces, Fisherfaces

   [] Live face recognition via webcam feed

7. Interactive Visualizations

   [] Show histograms, feature maps, contour plots

   [] Compare different image processing techniques side-by-side

   [] Display real-time classifier performance

8. API Integration & Deployment

   [] Allow image uploads via API for batch processing

   [] Deploy seamlessly on Streamlit Cloud

   [] Share app via public URL

## Setup (for devs)

```
    python setup.py
```
