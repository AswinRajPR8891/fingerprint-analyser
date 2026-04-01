import cv2

def match_features(desc1, desc2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(desc1, desc2)
    matches = sorted(matches, key=lambda x: x.distance)
    return matches

def calculate_score(matches):
    if len(matches) == 0:
        return 0, []

    good_matches = [m for m in matches if m.distance < 50]
    score = len(good_matches) / len(matches)

    return score, good_matches
