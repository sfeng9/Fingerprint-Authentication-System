import cv2


def extract_features(image):
    """Extracts SIFT descriptors (more stable than ORB on ridge patterns)."""
    sift = cv2.SIFT_create(nfeatures=300)
    kp, des = sift.detectAndCompute(image, None)
    return kp, des
