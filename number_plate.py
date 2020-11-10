import cv2
import numpy as np

widthimg = 640
heightimg = 640
plate_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_russian_plate_number.xml")
minarea = 500
count=0
color = (255,0,0)
cap =cv2.VideoCapture(0)
cap.set(3,widthimg)
cap.set(4,heightimg)


while True:
    flag, img = cap.read()
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberplate = plate_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=1)
    for x, y, w, h in numberplate:
        area = w*h
        if area >minarea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(img,"numberplate",(x,y-5),cv2.FONT_ITALIC,1,color,2)
            imgroi = img[y:y+h,x:x+w]
            cv2.imshow("roi",imgroi)
    cv2.imshow("video ", img)
    if cv2.waitKey(1) & 0xff ==ord ("s"):
        cv2.imwrite("res/no_plate_"+str(count)+".jpg",imgroi)
        cv2.rectangle(img,(0,200),(640,300),(0,255,0),cv2.FILLED)
        cv2.putText(img,"scan saved",(150,265),cv2.FONT_ITALIC,2,(0,0,255),2)
        cv2.imshow("result",img)
        cv2.waitKey(500)
        count ++1
        break
