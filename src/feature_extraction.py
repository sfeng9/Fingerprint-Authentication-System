import cv2

def extract_features(image):
    """Extracts ORB features from the image."""
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=500)
    
    # find the keypoints and descriptors
    kp, des = orb.detectAndCompute(image, None)
    
    return kp, des