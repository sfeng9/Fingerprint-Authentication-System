import cv2

def match_fingerprints(des1, des2):
    """Returns a score based on number of matching descriptors."""
    if des1 is None or des2 is None:
        return 0
        
    # Create BFMatcher object
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # Match descriptors
    matches = bf.match(des1, des2)
    
    # Sort them in the order of their distance
    matches = sorted(matches, key = lambda x:x.distance)
    
    # A simple score: number of matches with distance below a threshold
    good_matches = [m for m in matches if m.distance < 30]
    return len(good_matches)

def identify_user(probe_des, database, threshold=15):
    """
    Compares probe against all enrolled users.
    Returns the best matching ID or 'Unknown'
    """
    best_score = -1
    best_id = "Unknown"
    
    for person_id, enrolled_descriptors in database.items():
        # Score against each enrolled image for that person
        for enrolled_des in enrolled_descriptors:
            score = match_fingerprints(probe_des, enrolled_des)
            if score > best_score:
                best_score = score
                best_id = person_id
                
    if best_score < threshold:
        return "Unknown", best_score
        
    return best_id, best_score