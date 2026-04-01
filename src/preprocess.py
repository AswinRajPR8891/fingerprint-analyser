import cv2

def preprocess_image(path):
    img = cv2.imread(path, 0)
    img = cv2.equalizeHist(img)
    img = cv2.GaussianBlur(img, (5, 5), 0)
    return img
