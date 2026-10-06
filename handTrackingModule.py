import cv2 as cv
import mediapipe as mp
import time 

class HandDetector():
    def __init__(self, mode=False, maxHands=2,detectionConf=0.5, trackConf=0.5):

        self.mode = mode
        self.maxHands = maxHands
        self.detectionConf = detectionConf
        self.trackConf = trackConf

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
          static_image_mode=self.mode,
          max_num_hands=self.maxHands,
          min_detection_confidence=self.detectionConf,
          min_tracking_confidence=self.trackConf
          )
        self.mpDraw = mp.solutions.drawing_utils
    def findHands(self, img, draw=True ): 
        imgRGB = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if img is not None:
            if self.results.multi_hand_landmarks:
                for handLms in self.results.multi_hand_landmarks:
                    
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
            return img 
        
    def findPosition(self,img, draw= True):
        lmList1 = []    #first hand landmarks
        lmList2 = []    #second hand landmarks
        
        if self.results.multi_hand_landmarks:
            for i in range(len(self.results.multi_hand_landmarks)):

                myHand = self.results.multi_hand_landmarks[i]
                for id,lm in enumerate(myHand.landmark):
                    h,w,c = img.shape
                    cx, cy, cz = int(lm.x*w), int(lm.y*h), int(lm.z*h)
                    if i==0:
                        lmList1.append([id,cx,cy,cz])
                    if i==1:
                        lmList2.append([id,cx,cy,cz])
        return lmList1, lmList2
 


def main():
    pTime =0
    cTime =0
    cap = cv.VideoCapture(0)

    detector = HandDetector()
    while True:
        ret, img = cap.read()
    
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv.putText(img, f"FPS: {int(fps)}", (30,40),1, cv.FONT_HERSHEY_COMPLEX,(256,0,0))

        img = detector.findHands(img)
        print(detector.findPosition(img))
        cv.imshow("video", img)
       
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == "__main__":
    main()