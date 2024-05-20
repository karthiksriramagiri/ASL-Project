import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imageSize = 300
folder = "Data/C"
counter = 0
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V",
          "W", "X", "Y", "Z"]

while True:
    success, image = cap.read()
    display = image.copy()
    hands, image = detector.findHands(image)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imageWhite = np.ones((imageSize, imageSize, 3), np.uint8) * 255
        imageCrop = image[y - offset:y + h + offset, x - offset:x + w + offset]
        imageCropShape = imageCrop.shape
        aspectRatio = h / w
        if aspectRatio > 1:
            k = imageSize / h
            calculatedWidth = math.ceil(k * w)
            imgResize = cv2.resize(imageCrop, (calculatedWidth, imageSize))
            imgResizeShape = imgResize.shape
            widthGap = math.ceil((imageSize - calculatedWidth) / 2)
            imageWhite[:, widthGap:calculatedWidth + widthGap] = imgResize
            prediction, index = classifier.getPrediction(imageWhite, draw=False)
            print(prediction, index)
        else:
            k = imageSize / w
            calculatedHeight = math.ceil(k * h)
            imageResize = cv2.resize(imageCrop, (imageSize, calculatedHeight))
            imgResizeShape = imageResize.shape
            heightGap = math.ceil((imageSize - calculatedHeight) / 2)
            imageWhite[heightGap:calculatedHeight + heightGap, :] = imageResize
            prediction, index = classifier.getPrediction(imageWhite, draw=False)
        cv2.rectangle(display, (x - offset, y - offset-50),
                      (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
        cv2.putText(display, labels[index], (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(display, (x-offset, y-offset),
                      (x + w+offset, y + h+offset), (255, 0, 255), 4)
        cv2.imshow("ImageCrop", imageCrop)
        cv2.imshow("ImageWhite", imageWhite)
    cv2.imshow("Image", display)
    cv2.waitKey(1)