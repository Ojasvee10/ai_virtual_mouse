
import cv2
import mediapipe as mp

class HandRecog:
    def __init__(self):
        self.hands = mp.solutions.hands.Hands(static_image_mode=False,
                                              max_num_hands=2,
                                              min_detection_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

    def findHands(self, img):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        hands = []

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS)
                hands.append(hand_landmarks)
        return img, hands

    def recognizeGesture(self, hand):
        # Dummy logic
        return "FIST"
