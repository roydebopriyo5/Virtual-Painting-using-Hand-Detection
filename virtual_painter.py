import cv2
import numpy as np
import time
import os
from cvzone.HandTrackingModule import HandDetector
import HandTrackingModule as htm

brushThickness = 15
eraserThickness = 50

folderPath = "Header_virtual_painter"
myList = os.listdir(folderPath)
#print(myList)
overlayList = []
for imPath in myList:
    image = cv2.imread(f'{folderPath}/{imPath}')
    overlayList.append(image)
#print(len(overlayList))
header = overlayList[0]
drawColor = (166,166,166)

cap = cv2.VideoCapture(0,  cv2.CAP_DSHOW)
cap.set(3, 1280)
cap.set(4, 720)

detector = HandDetector(detectionCon=0.85)
xp,yp = 0,0
imgCanvas = np.zeros((720,1280,3), np.uint8)

tipIds=[4, 8, 12, 16, 20]
def fingersUp():
    fingers =[]

    #-----Thumb
    if lmList[tipIds[0]][1] < lmList[tipIds[0]-1][1]:
        fingers.append(1)
    else:
        fingers.append(0)
    
    #-----4 Fingers
    for id in range(1,5):
        if lmList[tipIds[id]][2] < lmList[tipIds[id]-2][2]:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

while True:
    #-----Import image
    success, img = cap.read()
    img = cv2.flip(img,1)

    #-----find Hand Landmarks
    img = detector.findHands(img, draw=False)
    lmList, bboxInfo = detector.findPosition(img, draw=False)

    if len(lmList) != 0:
        #print(lmList)

        #-----Tip of Index and Middle Fingers
        x1, y1 = lmList[8][0:]
        x2, y2 = lmList[12][0:]

        #-----check which Fingers are Up
        fingers = detector.fingersUp()
        #print(fingers)

        #-----if Selection mode - 2 fingers are up
        if fingers[1] and fingers[2]:
            xp,yp = 0,0
            #print("Selection Mode")
            #-----checking for the Click
            if y1 < 125:
                if 50 < x1 < 250:
                    header = overlayList[0]
                    drawColor = (166,166,166)
                if 250 < x1 < 450:
                    header = overlayList[1]
                    drawColor = (87,87,255)
                elif 450 < x1 < 600:
                    header = overlayList[2]
                    drawColor = (235,23,94)
                elif 550 < x1 < 700:
                    header = overlayList[3]
                    drawColor = (87,217,126)
                elif 1050 < x1 < 1200:
                    header = overlayList[4]
                    drawColor = (0,0,0)
            cv2.rectangle(img, (x1,y1 - 25), (x2,y2 + 25), drawColor, cv2.FILLED)

        #-----if Drawing mode - Index finger is up
        if fingers[1] and fingers[2]==False:
            cv2.circle(img, (x1,y1), 15, drawColor, cv2.FILLED)
            #print("Drawing Mode")
            if xp==0 and yp==0:
                xp,yp = x1,y1

            if drawColor == (0,0,0):
                cv2.line(img, (xp,yp), (x1,y1), drawColor, eraserThickness)
                cv2.line(imgCanvas, (xp,yp), (x1,y1), drawColor, eraserThickness)
            elif drawColor != (166,166,166):
                cv2.line(img, (xp,yp), (x1,y1), drawColor, brushThickness)
                cv2.line(imgCanvas, (xp,yp), (x1,y1), drawColor, brushThickness)

            xp,yp = x1,y1
    
    imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray, 50, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, imgCanvas)

    #-----setting the Header Image
    img[0:125, 0:1280] = header
    #img = cv2.addWeighted(img, 0.5, imgCanvas, 0.5, 0)
    cv2.imshow("Image", img)
    cv2.imshow("Canvas", imgCanvas)
    cv2.imshow("Inv", imgInv)
    cv2.waitKey(1)
