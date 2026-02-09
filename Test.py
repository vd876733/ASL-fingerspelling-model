import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# ---------------------------- CONFIG ---------------------------- #
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=2)  # detect up to 2 hands
classifier = Classifier("Model/keras_model_1.h5", "Model/labels_1.txt")
offset = 20
imgSize = 300
labels = ["A", "B", "C", "D" , "E", "F" , "G", "H" , "I" , "J" ,"K" , "L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z","1","2","3","4","5","6","7","8","9","NEXT"]

# -------------------------- MAIN LOOP -------------------------- #
while True:
    success, img = cap.read()
    if not success or img is None:
        continue

    imgOutput = img.copy()
    hands, img = detector.findHands(img)

    if hands:
        for hand in hands:
            x, y, w, h = hand['bbox']
            handType = hand['type']  # "Left" or "Right"

            # Safe crop coordinates
            y1 = max(0, y - offset)
            y2 = min(img.shape[0], y + h + offset)
            x1 = max(0, x - offset)
            x2 = min(img.shape[1], x + w + offset)

            imgCrop = img[y1:y2, x1:x2]

            if imgCrop.size == 0:
                continue  # skip if crop failed

            # Create white background
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

            # Draw on output image
            cv2.rectangle(imgOutput, (x - offset, y - offset - 50),
                          (x - offset + 90, y - offset - 50 + 50), (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, labels[index], (x, y - 26),
                        cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset),
                          (x + w + offset, y + h + offset), (255, 0, 255), 4)

            # Show cropped and white images
            cv2.imshow(f"ImageCrop_{handType}", imgCrop)
            cv2.imshow(f"ImageWhite_{handType}", imgWhite)

    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
