import cv2

def extract_features(img):
    orb = cv2.ORB_create(nfeatures=2000)
    keypoints, descriptors = orb.detectAndCompute(img, None)
    return keypoints, descriptors
