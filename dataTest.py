import cv2
from streamlit import success

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    cv2.imshow("image", img)
    cv2.waitKey(1)