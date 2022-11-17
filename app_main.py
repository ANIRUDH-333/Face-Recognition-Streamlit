import cv2
from cvzone.FaceDetectionModule import FaceDetector
import streamlit as st

############################################################################################
st.title("Face Recognition System")
run = st.checkbox("Capture")
frame_window = st.image([])
############################################################################################
detector = FaceDetector()
cap = cv2.VideoCapture(0)
while True:
    ret, img = cap.read()
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    face, New_Image = detector.findFaces(img)
    frame_window.image(img)