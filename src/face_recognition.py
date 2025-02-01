
# filename - src/face_recognition.py

import cv2
import numpy as np
import face_recognition

def detect_faces(image):
    face_locations = face_recognition.face_locations(image)
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    return image

def encode_faces(image):
    face_encodings = face_recognition.face_encodings(image)
    return face_encodings

def match_faces(known_encodings, test_encoding, tolerance=0.6):
    results = face_recognition.compare_faces(known_encodings, test_encoding, tolerance)
    return results