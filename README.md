# ImageSense

## Software

Python 3.13
Streamlit
OpenCV2

## Features & Details:

1. Image Preprocessing

   [] Upload and display images (JPG, PNG, etc.)
   [] Convert to grayscale & histogram equalization
   [] Resize, crop, and rotate images
   [] Apply smoothing & sharpening filters (Gaussian, Median, Bilateral)

2. Edge Detection & Segmentation

   [] Edge detection using Sobel, Prewitt, Laplacian, Canny
   [] Image thresholding (Global, Adaptive, Otsu's method)
   [] Segmentation using K-Means, Watershed, GrabCut
   [] Morphological operations (Erosion, Dilation, Opening, Closing)

3. Feature Extraction

   [] Extract & visualize key features using SIFT, SURF, ORB, HOG
   [] PCA-based dimensionality reduction visualization
   [] Compute texture features (GLCM, LBP)
   [] Histogram-based feature analysis

4. Pattern Recognition & Classification

   [] Train & test classifiers on extracted features
   [] Support for KNN, SVM, Decision Trees, CNN (Pretrained Models)
   [] Upload custom datasets for classification
   [] Evaluate models with accuracy, precision, recall

5. Face Detection & Recognition

   [] Detect faces using Haar Cascades, DNN (ResNet, MobileNet)
   [] Recognize faces using LBPH, Eigenfaces, Fisherfaces
   [] Live face recognition via webcam feed

6. Interactive Visualizations

   [] Show histograms, feature maps, contour plots
   [] Compare different image processing techniques side-by-side
   [] Display real-time classifier performance

7. API Integration & Deployment

   [] Allow image uploads via API for batch processing
   [] Deploy seamlessly on Streamlit Cloud
   [] Share app via public URL

## Setup (for devs)

```
    python setup.py
```
