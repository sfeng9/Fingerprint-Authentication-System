import cv2
import numpy as np

def preprocess(image):
    
    # 1. Noise Reduction using Median Blur
    blurred = cv2.medianBlur(image, 3)
    
    # 2. Contrast Enhancement (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blurred)
    
    # 3. Normalization
    normalized = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX)
    
    return normalized