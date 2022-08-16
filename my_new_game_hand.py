import cv2
import time
import mediapipe as mp
import module_hand_tracking as htm
ptime = 0
ctime = 0
cap = cv2.VideoCapture(0)  # for internal camera
detector= htm.HandDetector()
while True:
    success, img = cap.read()
    img= detector.FindHands(img)
    lm_list=detector.FindPosition(img)
    if len(lm_list)!=0:
        print(lm_list[4])

    ctime = time.time()
    fps = 1 / (ctime - ptime)
    ptime = ctime

    cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_DUPLEX, 2
                    , (255, 0, 255), 2)

    cv2.imshow("Image", img)
    cv2.waitKey(1)