import cv2
import numpy as np
cap = cv2.VideoCapture(0)
while True:
    sucess , video = cap.read()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_frontalface_default.xml")
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades+"haarcascade_eye.xml")
    detect_face =face_cascade.detectMultiScale(video,scaleFactor=1.1,minNeighbors=2)
    for x,y,w,h in detect_face:
        cv2.rectangle(video,(x,y) ,(x+w,y+h),(200,10,185),2)
        cv2.putText(video, "face", (x, y - 10), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 255, 0), 2)
        roi = video[y:y+h ,x:x+w]
        roi_image = video[y:y+h , x:x+w]
        eye_detect = eye_cascade.detectMultiScale(roi)
        for ex,ey,ew,eh in eye_detect:
            cv2.rectangle(roi_image,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
    cv2.imshow("main_window", video)
    if cv2.waitKey(1) & 0xff ==ord("q"):
        break
