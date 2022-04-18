from re import L
import cv2
import time
import numpy as np
import HandTrackModule as htm
from subprocess import call
import osascript

wcam, hcam = 640, 480

cap = cv2.VideoCapture(0)
cap.set(3, wcam)
cap.set(4, hcam)

detector = htm.handDetector(detectionCon=0.7)

def main():

    pTime = 0
    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPositions(img, draw=False)
        if len(lmList) != 0:
            if lmList[4] and lmList[8]:
                dis = detector.distance(lmList[4], lmList[8])
                print(lmList[4], lmList[8], dis)

                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(img, (x1, y1), 7, (255, 0, 0), cv2.FILLED)
                cv2.circle(img, (x2, y2), 7, (255, 0, 0), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 7, (255, 0, 0), cv2.FILLED)

                vol = dis // 3
                command  = 'set volume output volume ' + str(vol)
                osascript.osascript(command)

            


        cTime = time.time()
        fps = 1/(cTime - pTime)
        pTime = time.time()

        cv2.putText(img, f'FPS: {str(int(fps))}', (10,70), cv2.FONT_HERSHEY_PLAIN, 2,
        (255, 0, 0), 2)

        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()