import streamlit as st
import cv2
import numpy as np
from PIL import Image
import face_recognition

haar_model = "haarcascade_frontalface_default.xml"

st.title(' Face Recognition System')

st.subheader("Upload Your Photo")


# if st.button('Upload Image'):
uploaded_file = st.file_uploader("Upload image", type=['jpeg', 'png', 'jpg'])
if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    data = Image.fromarray(image)
    data.save('test.png')

st.subheader("Compare with the uploaded Image")
if st.button("Compare"):
    capture = cv2.VideoCapture(0)
    facedetect = cv2.CascadeClassifier(haar_model)
    count = 0
    while True:
        _,frame = capture.read()
        faces = facedetect.detectMultiScale(frame,1.6,5)
        
        for x,y,w,h in faces:
            
            count += 1
            img = cv2.cvtColor(frame[y:y+h,x:x+w],cv2.COLOR_BGR2RGB)
            if len(face_recognition.face_encodings(img)) == 0:
                continue
            else:     
                img_encode = face_recognition.face_encodings(img)[0] 
                
                known_image = face_recognition.load_image_file('test.png')
                encoding = face_recognition.face_encodings(known_image)[0]
                result = face_recognition.compare_faces([encoding],img_encode,0.6)
                if result[0] == True:
                    cv2.putText(frame,"Matched",(x-8,y),cv2.FONT_HERSHEY_PLAIN,1,(0,255,0),2)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)  
                else:
                    cv2.putText(frame,"Not Matched",(x-8,y),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
                    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
                
                    
        cv2.imshow("Frame",frame)
        key = cv2.waitKey(1)
        if key == 27:
            break
    capture.release()
    cv2.destroyAllWindows()
