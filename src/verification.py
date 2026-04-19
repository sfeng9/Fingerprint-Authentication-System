import cv2


def match_fingerprints(des1, des2, bf=None):
    """Score: count of strong L2 matches after cross-check (SIFT descriptors)."""
    if des1 is None or des2 is None:
        return 0

    if bf is None:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(des1, des2)
    max_dist = 220.0
    return sum(1 for m in matches if m.distance < max_dist)


def identify_user(probe_des, database, threshold=34, margin=5):
    """
    1:N identification using best per-person score and a gap to the runner-up.
    """
    if not database:
        return "Unknown", 0

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    per_person = {}

    for person_id, enrolled_descriptors in database.items():
        best_for_person = 0
        for enrolled_des in enrolled_descriptors:
            score = match_fingerprints(probe_des, enrolled_des, bf)
            if score > best_for_person:
                best_for_person = score
        per_person[person_id] = best_for_person

    ranked = sorted(per_person.items(), key=lambda x: -x[1])
    best_id, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0

    if best_score < threshold:
        return "Unknown", best_score
    if best_score - second_score < margin:
        return "Unknown", best_score

    return best_id, best_score
