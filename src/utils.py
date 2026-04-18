#Imports
import os
import cv2
import re

def load_image(path):
    """Loads images"""
    return cv2.imread(path, cv2.IMREAD_GRAYSCALE)

def parse_filename(filename):
    """Parses YYY_R0_KKK.bmp to extract person ID"""
    match = re.match(r"(\d+)_R0_(\d+)", filename)
    if match:
        #Return person_id and image_index
        return match.group(1), match.group(2)
    return None, None

def get_all_images(folder_path):
    """Returns a list of all the images"""
    images = []
    for f in os.listdir(folder_path):
        if f.endswith(".bmp"):
            person_id, _ = parse_filename(f)
            images.append({
                'id': person_id,
                'path': os.path.join(folder_path, f),
                'filename': f
            })
    return images