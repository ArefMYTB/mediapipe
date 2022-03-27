import cv2
import mediapipe as mp
import time


class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands()
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # print(results.multi_hand_landmarks)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                # for id, lm in enumerate(handLms.landmark):
                #     height, weight, chanel = img.shape
                    # get position in the img
                    # cx, cy = int(lm.x*weight), int(lm.y*height)
                    # draw a circle around wrist point
                    # if id == 0:
                    #     cv2.circle(img, (cx, cy), 25, (255, 0, 255), cv2.FIllED)
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNum=0, draw=True):

        lmList = []
        if self.results.multi_hand_landmarks:
            # choose a particular hand
            myHand = self.results.multi_hand_landmarks[handNum]
            for id, lm in enumerate(myHand.landmark):
                height, weight, chanel = img.shape
                # get position in the img
                cx, cy = int(lm.x*weight), int(lm.y*height)
                lmList.append([id, cx, cy])
                # draw a circle around wrist point
                if draw:
                    cv2.circle(img, (cx, cy), 25, (255, 0, 255))

        return lmList

def main():
    cap = cv2.VideoCapture(0)

    prev_time = 0
    curr_time = 0
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)
        lmList = detector.findPosition(img)

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time

        cv2.putText(img, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 0), 3)

        cv2.imshow("image", img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()