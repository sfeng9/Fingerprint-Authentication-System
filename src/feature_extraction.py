from typing import Optional

import numpy as np
import cv2


def _root_sift_descriptors(des: Optional[np.ndarray]) -> Optional[np.ndarray]:
    """RootSIFT: improves matching vs. raw SIFT on textured imagery."""
    if des is None or len(des) == 0:
        return des
    d = des.astype(np.float64)
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    d = np.sqrt(np.abs(d))
    d /= np.linalg.norm(d, axis=1, keepdims=True) + 1e-12
    return d.astype(np.float32)


def _normalized_histogram(gray: np.ndarray, bins: int = 128) -> np.ndarray:
    hist = cv2.calcHist([gray], [0], None, [bins], [0, 256])
    cv2.normalize(hist, hist)
    h = hist.flatten().astype(np.float64)
    n = np.linalg.norm(h) + 1e-9
    return h / n


def extract_features(image):
    """
    RootSIFT descriptors plus a global intensity histogram (L2-normalized).
    More keypoints + RootSIFT typically lifts 1:N hit rate on ridge patterns.
    """
    sift = cv2.SIFT_create(nfeatures=500)
    kp, des = sift.detectAndCompute(image, None)
    if des is not None:
        des = _root_sift_descriptors(des)
    hist = _normalized_histogram(image, bins=128)
    feats = {"sift": des, "hist": hist}
    return kp, feats
