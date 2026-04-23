import cv2
import numpy as np

def detect_plate_region(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Thresholding
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # 2. Edge Detection
    edges = cv2.Canny(gray, 100, 200)

    # 3. Contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    plate = None

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        if w > 80 and h > 25:
            plate = img[y:y+h, x:x+w]

    return plate


def detect_corners(img):

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    corners = cv2.goodFeaturesToTrack(gray, 50, 0.01, 10)

    if corners is not None:
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(img, (int(x), int(y)), 3, (0,255,0), -1)

    return img