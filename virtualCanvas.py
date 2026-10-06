import cv2 as cv
import mediapipe as mp
import time 
import handTrackingModule as ht
import numpy as np 

detector = ht.HandDetector()

pTime =0
cTime =0
cap = cv.VideoCapture(0)

canvas = None
preX, preY = 0, 0

while True:
    ret, frame = cap.read()
    if ret:
        frame = cv.flip(frame,1)
        canvas = np.zeros_like(frame)
        break

while True:
    ret, img = cap.read()
    img = cv.flip(img,1)
    
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv.putText(img, f"FPS: {int(fps)}", (30,40),1, cv.FONT_HERSHEY_COMPLEX,(256,0,0))

    img = detector.findHands(img, draw=False)
    lmlist,_ = detector.findPosition(img)
    if len(lmlist) != 0:
        
        #coordinates of index , middle fingertips

        x1,y1 = lmlist[8][1],lmlist[8][2]   #index finger tip coordinates
        x2,y2 = lmlist[12][1],lmlist[12][2]  
        x3,y3 = lmlist[4][1],lmlist[4][2]

        cv.circle(img, (x1,y1), 10, (255,0,255),-1)
        if canvas is None:
            canvas = np.zeros_like(img)
        # write with index finger tip => middle, thumb finger tip should be below index finger tip
        if y2 > y1 and y3> y1:
            if preX ==0 and preY ==0:
                preX,preY = x1,y1
            cv.line(canvas,(preX,preY), (x1,y1),(255,0,255),3)
            preX,preY = x1,y1

        #erase if y coordinate of index and middle finger is greater than thumb==> fist closed
        elif y1> y3 and y2 > y3:
            canvas = np.zeros_like(img)
        else: 
            preX,preY = 0,0

    #merge canvas and web frame
    grey_canvas = cv.cvtColor(canvas, cv.COLOR_BGR2GRAY)
    _, inv_canvas = cv.threshold(grey_canvas,50,255,cv.THRESH_BINARY_INV)
    inv_canvas = cv.cvtColor(inv_canvas, cv.COLOR_GRAY2BGR)

    img = cv.bitwise_and(img, inv_canvas)
    img = cv.bitwise_or(img, canvas)
    

    cv.imshow("video", img)
       
    if cv.waitKey(1) & 0xFF == ord('q'):
        break