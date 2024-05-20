import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
offset = 20
imageSize = 300
folder = "Data/Z"
if not os.path.exists(folder):
    os.makedirs(folder)

i = 0
while True:
    success, image = cap.read()
    hands, image = detector.findHands(image)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']
        imageWhite = np.ones((imageSize, imageSize, 3), np.uint8) * 255
        imageCrop = image[y - offset:y + h + offset, x - offset:x + w + offset]
        imageCropShape = imageCrop.shape
        aspectRatio = h / w
        if aspectRatio > 1:
            constant = imageSize / h
            calculateWidth = math.ceil(constant * w)
            imageResize = cv2.resize(imageCrop, (calculateWidth, imageSize))
            imageResizeShape = imageResize.shape
            widthGap = math.ceil((300 - calculateWidth)/2)
            imageWhite[:, widthGap:calculateWidth+widthGap] = imageResize

        else:
            constant = imageSize / w
            calculatedHeight = math.ceil(constant * h)
            imageResize = cv2.resize(imageCrop, (imageSize, calculatedHeight))
            imageResizeShape = imageResize.shape
            heightGap = math.ceil((300 - calculatedHeight) / 2)
            imageWhite[heightGap: calculatedHeight + heightGap, :] = imageResize

        cv2.imshow("ImageCrop", imageCrop)
        cv2.imshow("ImageWhite", imageWhite)

    cv2.imshow("Image", image)
    key = cv2.waitKey(1)
    if key == ord("s"):
        i += 1
        filePath = f'{folder}/Image_{time.time()}.jpg'
        cv2.imwrite(filePath, imageWhite)
        print(i)
