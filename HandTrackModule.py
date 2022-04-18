import cv2
import mediapipe as mp
import time
import math
# import handtrackmodel

class handDetector():

    def __init__(self, mode = False, maxHands = 2, complexity = 1, detectionCon = 0.5, trackCon = 0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.complexity = complexity
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.complexity, self.detectionCon, self.trackCon)
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img

    def findPositions(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                # print(id, lm)
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                # print(id, cx,cy)
                lmList.append([id,cx,cy])
                if draw:
                    if id == 4 or id == 8:
                        cv2.circle(img, (cx, cy), 10, (255,0,255), cv2.FILLED)

        return lmList

    def distance(self, p1, p2):
        x = p1[1] - p2[1]
        y = p1[2] - p2[2]
        self.dis = math.sqrt(x * x + y * y)
        return self.dis

def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    detector = handDetector()

    pTime = 0

    while True:
        success, img = cap.read()

        img = detector.findHands(img)
        lmList = detector.findPositions(img)
        if len(lmList) != 0:
            if lmList[4] and lmList[8]:
                dis = detector.distance(lmList[4], lmList[8])
                print(lmList[4],lmList[8], dis)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(img,str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3,
        (255, 0, 255), 3)

        cv2.imshow("Image", img)
        cv2.waitKey(1)
        

if __name__ == '__main__':
    main()