import cv2
import numpy as np
framewidth = 640
frameheigth = 640
cap = cv2.VideoCapture(0)
cap.set(3,framewidth)
cap.set(4, frameheigth)
cap.set(10,1300)
mycolor = [[5,107,0,19,255,255],
         [133,56,0,159,156,255],
         [57,76,0,100,255,255]]

mycolorvalue = [[51,153,255],
                [255,0,255],
                [0,255,0]]
mypoints =[]  #[x,y, colorid]
def find_color(img,mycolor,mycolorvalue ):
     imghsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
     count = 0
     newpoint = []
     for color in mycolor:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imghsv,lower,upper)
        x,y=getcontours(mask)
        cv2.circle(imgresult,(x,y),10,mycolorvalue[count],cv2.FILLED)
        if x!=0 and y!=0:
            newpoint.append([x,y,count])
        count +=1
        #cv2.imshow(str(color[0]),mask)
     return newpoint



def getcontours(img):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    x,y,w,h =0,0,0,0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if  area > 500:
            cv2.drawContours(imgresult, cnt, -1, (255, 0, 0), 2)
            perl = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*perl,True)
            x,y,w,h = cv2.boundingRect(approx)
    return x+w//2,y

def drawoncanvas(mypoints,mycolorvalue):
    for point in mypoints:
        cv2.circle(imgresult , (point[0],point[1]),10, mycolorvalue[point[2]], cv2.FILLED)

while True:
    success, img = cap.read()
    imgresult = img.copy()
    newpoint =find_color(img,mycolor,mycolorvalue)
    if len(newpoint)!=0:
        for newp in newpoint:
            mypoints.append(newp)
    if len(mypoints)!=0:
        drawoncanvas(mypoints,mycolorvalue)
        
    cv2.imshow("video",imgresult)
    if cv2.waitKey(1) & 0xff==ord("q"):
        break
