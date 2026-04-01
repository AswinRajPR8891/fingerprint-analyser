import cv2
import numpy as np
from skimage.morphology import skeletonize

def get_minutiae(image):
    # Convert to binary
    _, thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

    # Skeletonize (convert to 0/1)
    skeleton = skeletonize(thresh // 255)
    skeleton = skeleton.astype(np.uint8)

    minutiae = []

    rows, cols = skeleton.shape

    for i in range(1, rows - 1):
        for j in range(1, cols - 1):
            if skeleton[i, j] == 1:
                neighbors = [
                    skeleton[i-1, j], skeleton[i-1, j+1],
                    skeleton[i, j+1], skeleton[i+1, j+1],
                    skeleton[i+1, j], skeleton[i+1, j-1],
                    skeleton[i, j-1], skeleton[i-1, j-1]
                ]

                transitions = 0
                for k in range(8):
                    if neighbors[k] == 0 and neighbors[(k+1) % 8] == 1:
                        transitions += 1

                if transitions == 1:
                    minutiae.append((j, i, 'ending'))
                elif transitions == 3:
                    minutiae.append((j, i, 'bifurcation'))

    return minutiae, skeleton
