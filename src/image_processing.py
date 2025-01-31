
# filename - src/image_processing.py

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.cluster import KMeans
from tensorflow.keras.models import load_model

def graycale(img): pass
def increase_contrast(img): pass
def decrease_contrast(img): pass
def resize(img, new_length, new_breadth): pass
def crop(img, new_length, new_breadth): pass
def add_frame(img): pass
def rotate(img, direction):pass
def smoothen(img): pass

# Load pre-trained model for classification (if applicable)
# model = load_model("path_to_model.h5")

def preprocess(image, grayscale, blur, resize, new_size):
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if blur > 0:
        image = cv2.GaussianBlur(image, (blur * 2 + 1, blur * 2 + 1), 0)
    if resize:
        image = cv2.resize(image, new_size)
    return image

def edge_detection(image, method):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if method == "Canny":
        return cv2.Canny(gray, 100, 200)
    elif method == "Sobel":
        return cv2.Sobel(gray, cv2.CV_64F, 1, 1, ksize=5)
    elif method == "Laplacian":
        return cv2.Laplacian(gray, cv2.CV_64F)
    return image

def segment(image, method):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if method == "K-Means Clustering":
        pixels = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=3).fit(pixels)
        segmented = kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape)
        return segmented.astype(np.uint8)
    elif method == "Watershed":
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
        _, sure_fg = cv2.threshold(dist_transform, 0.7 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        unknown = cv2.subtract(binary, sure_fg)
        markers = cv2.connectedComponents(sure_fg)[1] + 1
        markers[unknown == 255] = 0
        markers = cv2.watershed(image, markers)
        image[markers == -1] = [255, 0, 0]
        return image
    elif method == "GrabCut":
        mask = np.zeros(image.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (50, 50, image.shape[1] - 50, image.shape[0] - 50)
        cv2.grabCut(image, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        return image * mask2[:, :, np.newaxis]
    return image

def classify(image, classifier):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (64, 64)) / 255.0
    resized = np.expand_dims(resized, axis=(0, -1))
    if classifier == "CNN":
        # predicted_class = np.argmax(model.predict(resized), axis=1)
        predicted_class = "Class X"  # Placeholder
    elif classifier == "KNN":
        predicted_class = "Class Y"  # Placeholder for KNN implementation
    elif classifier == "SVM":
        predicted_class = "Class Z"  # Placeholder for SVM implementation
    return predicted_class
