import cv2
import numpy as np

# Slightly looser distance to recover genuine matches after RootSIFT rescaling
_SIFT_MAX_DIST = 255.0
_HIST_BASE = 0.965
_HIST_WEIGHT = 150.0


def _histogram_cosine(h1: np.ndarray, h2: np.ndarray) -> float:
    if h1 is None or h2 is None:
        return 0.0
    return float(np.clip(np.dot(h1, h2), -1.0, 1.0))


def match_fingerprints(feat1, feat2, bf=None) -> float:
    """
    Hybrid score: RootSIFT strong-match count + histogram cosine bonus.
    """
    if feat1 is None or feat2 is None:
        return 0.0

    d1 = feat1.get("sift") if isinstance(feat1, dict) else feat1
    d2 = feat2.get("sift") if isinstance(feat2, dict) else feat2
    h1 = feat1.get("hist") if isinstance(feat1, dict) else None
    h2 = feat2.get("hist") if isinstance(feat2, dict) else None

    hist_sim = _histogram_cosine(h1, h2)
    hist_bonus = _HIST_WEIGHT * max(0.0, hist_sim - _HIST_BASE)

    if d1 is None or d2 is None or len(d1) < 2 or len(d2) < 2:
        return hist_bonus

    if bf is None:
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    matches = bf.match(d1, d2)
    good = sum(1 for m in matches if m.distance < _SIFT_MAX_DIST)
    return float(good) + hist_bonus


def identify_user(
    probe_feat,
    database,
    threshold=28.5,
    margin=1.6,
):
    """
    1:N identification. Lower threshold/margin vs. earlier builds to limit
    false rejects (Unknown) while RootSIFT + multi-template keeps impostor
    peaks more separable.
    """
    if not database:
        return "Unknown", 0.0

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    per_person = {}

    for person_id, enrolled_templates in database.items():
        best_score = 0.0
        for templ in enrolled_templates:
            if templ is None:
                continue
            s = match_fingerprints(probe_feat, templ, bf)
            if s > best_score:
                best_score = s
        per_person[person_id] = best_score

    ranked = sorted(per_person.items(), key=lambda x: -x[1])
    best_id, best_score = ranked[0]
    second_score = ranked[1][1] if len(ranked) > 1 else 0.0

    if best_score < threshold:
        return "Unknown", best_score
    if best_score - second_score < margin:
        return "Unknown", best_score

    return best_id, best_score
