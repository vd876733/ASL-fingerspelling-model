import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# ---------------------------- CONFIG ---------------------------- #
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model_1.h5", "Model/labels_1.txt")
offset = 20
imgSize = 300

labels = ["A","B","C","D","E","F","G","H","I","J","K","L","M","N","O","P",
          "Q","R","S","T","U","V","W","X","Y","Z",
          "1","2","3","4","5","6","7","8","9","NEXT"]

# ---------------------------- STATE ---------------------------- #
word = ""
current_label = ""

# ---------------------------- MAIN LOOP ---------------------------- #
while True:
    success, img = cap.read()
    if not success:
        continue

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Safe crop
        y1 = max(0, y - offset)
        y2 = min(img.shape[0], y + h + offset)
        x1 = max(0, x - offset)
        x2 = min(img.shape[1], x + w + offset)
        imgCrop = img[y1:y2, x1:x2]

        if imgCrop.size != 0:
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wGap + wCal] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hGap + hCal, :] = imgResize

            # Get prediction
            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            current_label = labels[index]

            # Draw label
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x + 90, y - offset), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, current_label, (x, y - 26),
                        cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)

    # ---------------------- KEYBOARD CONTROLS ---------------------- #
    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        if current_label != "":
            word += current_label
            print(f"Added letter: {current_label} → Word so far: {word}")

    elif key == ord('q'):
        print(f"\n✅ Final Word: {word}")
        break

    # Display current frame + word
    cv2.putText(imgOutput, f"Word: {word}", (50, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
    cv2.imshow("Image", imgOutput)

cap.release()
cv2.destroyAllWindows()
