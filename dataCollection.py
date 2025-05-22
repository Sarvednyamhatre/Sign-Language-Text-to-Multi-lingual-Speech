import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)


offset = 20
imgSize = 300


current_sign = 'A'  
folder = f"Data/{current_sign}"
os.makedirs(folder, exist_ok=True)
counter = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera. Exiting...")
        break

    
    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        
        x1 = max(0, x - offset)
        y1 = max(0, y - offset)
        x2 = min(img.shape[1], x + w + offset)
        y2 = min(img.shape[0], y + h + offset)
        imgCrop = img[y1:y2, x1:x2]

        
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        
        aspectRatio = h / w
        try:
            if aspectRatio > 1:  
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[:, wGap:wCal + wGap] = imgResize
            else:  
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize
        except Exception as e:
            print(f"Error during resizing or cropping: {e}")
            continue

        
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

   
    cv2.putText(img, f"Sign: {current_sign}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.imshow("Image", img)

    
    key = cv2.waitKey(1)
    if key == ord("s"):  
        counter += 1
        cv2.imwrite(f'{folder}/Image_{time.time()}.jpg', imgWhite)
        print(f"Saved {current_sign} image: {counter}")
    elif key >= ord('A') and key <= ord('Z'):  
        current_sign = chr(key)
        folder = f"Data/{current_sign}"
        os.makedirs(folder, exist_ok=True)
        counter = 0
        print(f"Switched to sign: {current_sign}")
    elif key == ord("q"): 
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
