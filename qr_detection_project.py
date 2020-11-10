import numpy as np
import cv2
from pyzbar.pyzbar import decode

# img = cv2.imread("res/mithilesh_qr.png")
cap = cv2.VideoCapture(0)
cap.set(3,700)
cap.set(4,700)
while True:
    flag , img =cap.read()
    for barcode in decode(img):
        my_data = barcode.data.decode('utf-8')
        print(my_data)
        pts = np.array([barcode.polygon],np.int32)
        pts = pts.reshape((-1,1,2))
        cv2.polylines(img,[pts],True,(255,0,255),5)
        pts2 =barcode.rect
        cv2.putText(img,"scan",(pts2[0],pts2[1]),
                    cv2.FONT_HERSHEY_DUPLEX,0.9,(255,0,255),2)
    cv2.imshow("window",img)
    if cv2.waitKey(1) & 0xff ==ord("q"):
        break
