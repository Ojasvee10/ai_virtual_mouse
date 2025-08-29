
from core.hand_recognition import HandRecog
from core.controller import Controller
from core.gesture_enums import Gest, HLabel
from utils.mediapipe_utils import classify_hands
import mediapipe as mp
import cv2

class GestureController:
    def start(self):
        print("Starting gesture controller...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Webcam not found or cannot be opened.")
            return

        detector = HandRecog()
        controller = Controller()

        while True:
            success, img = cap.read()
            if not success:
                print("Failed to grab frame.")
                break

            img, hands = detector.findHands(img)
            hands = classify_hands(hands)

            if hands:
                hand = hands[0]
                gesture = detector.recognizeGesture(hand)
                if gesture:
                    controller.handle(gesture)

            cv2.imshow("Virtual Mouse", img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    gc1 = GestureController()
    gc1.start()
