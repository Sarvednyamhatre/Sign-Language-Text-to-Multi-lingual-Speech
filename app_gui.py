import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import tkinter as tk
from PIL import Image, ImageTk
from googletrans import Translator
from gtts import gTTS
import os
import pyttsx3


cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier(
    "D:\Mini project sem 7\sign-language-detection-and-conversion\Model\Sign_model.h5",
    "D:\Mini project sem 7\sign-language-detection-and-conversion\Model\labels.txt"
)
offset = 20
imgSize = 300
labels = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


sentence = ""
detected_letter = ""
engine = pyttsx3.init()
engine.setProperty('rate', 120)
translator = Translator()

# Language options
languages = {
    "English": "en",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Hindi": "hi",
}


window = tk.Tk()
window.title("Sign Language to Text")
window.geometry("800x600")


selected_language = tk.StringVar(value="English") 
video_label = tk.Label(window)
video_label.pack()

letter_label = tk.Label(window, text="Detected Letter: ", font=("Helvetica", 20))
letter_label.pack(pady=10)

sentence_label = tk.Label(window, text="Sentence: ", font=("Helvetica", 20))
sentence_label.pack(pady=10)


def add_to_sentence():
    global sentence
    sentence += detected_letter
    sentence_label.config(text="Sentence: " + sentence)


def add_space():
    global sentence
    sentence += " "
    sentence_label.config(text="Sentence: " + sentence)


def clear_sentence():
    global sentence
    sentence = ""
    sentence_label.config(text="Sentence: ")


def delete_last_letter():
    global sentence
    sentence = sentence[:-1]
    sentence_label.config(text="Sentence: " + sentence)


def speak_sentence():
    target_lang = languages.get(selected_language.get(), "en")
    try:
        translated = translator.translate(sentence, dest=target_lang)
        tts = gTTS(text=translated.text, lang=target_lang)
        tts.save("translated_sentence.mp3")
        os.system("start translated_sentence.mp3")  
    except Exception as e:
        print(f"Error during translation or TTS: {e}")


button_frame = tk.Frame(window)
button_frame.pack(pady=10)


add_button = tk.Button(button_frame, text="Add Letter", command=add_to_sentence, font=("Helvetica", 14), height=2, width=20)
add_button.grid(row=0, column=0, padx=5)

space_button = tk.Button(button_frame, text="Space", command=add_space, font=("Helvetica", 14), height=2, width=20)
space_button.grid(row=0, column=1, padx=5)

clear_button = tk.Button(button_frame, text="Clear Sentence", command=clear_sentence, font=("Helvetica", 14), height=2, width=20)
clear_button.grid(row=0, column=2, padx=5)

delete_button = tk.Button(button_frame, text="Delete Last Letter", command=delete_last_letter, font=("Helvetica", 14), height=2, width=20)
delete_button.grid(row=0, column=3, padx=5)

speak_button = tk.Button(button_frame, text="Speak Sentence", command=speak_sentence, font=("Helvetica", 14), height=2, width=20)
speak_button.grid(row=0, column=4, padx=5)


language_label = tk.Label(window, text="Select Language: ", font=("Helvetica", 16))
language_label.pack(pady=10)


language_menu = tk.OptionMenu(window, selected_language, *languages.keys())
language_menu.config(font=("Helvetica", 14), width=15)
language_menu.pack(pady=10)


hand_present_last_frame = False


import time


last_detected_time = None

def update_frame():
    global detected_letter, sentence, hand_present_last_frame, last_detected_time
    success, img = cap.read()

    if not success or img is None:
        print("Error: Could not capture frame from webcam.")
        window.after(10, update_frame)
        return

    imgOutput = img.copy()
    hands, img = detector.findHands(imgOutput)

    if hands:

        hand_present_last_frame = True


        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[max(0, y - offset):y + h + offset, max(0, x - offset):x + w + offset]

        if imgCrop.size != 0:
            aspectRatio = h / w

            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = math.ceil((imgSize - imgResize.shape[1]) / 2)
                imgWhite[:, wGap:wGap + imgResize.shape[1]] = imgResize
            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = math.ceil((imgSize - imgResize.shape[0]) / 2)
                imgWhite[hGap:hGap + imgResize.shape[0], :] = imgResize

            prediction, index = classifier.getPrediction(imgWhite, draw=False)
            detected_letter = labels[index]
            letter_label.config(text=f"Detected Letter: {detected_letter}")


            if last_detected_time is None:
                last_detected_time = time.time()

         
            if time.time() - last_detected_time >= 5:
                sentence += detected_letter
                sentence_label.config(text="Sentence: " + sentence)
                last_detected_time = None 
                detected_letter = ""  

            cv2.rectangle(imgOutput, (x - offset, y - offset - 50), (x - offset + 90, y - offset - 50 + 50),
                          (255, 0, 255), cv2.FILLED)
            cv2.putText(imgOutput, detected_letter, (x, y - 26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
            cv2.rectangle(imgOutput, (x - offset, y - offset), (x + w + offset, y + h + offset), (255, 0, 255), 4)

    else:

        last_detected_time = None
        hand_present_last_frame = False

    imgRGB = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
    imgPIL = Image.fromarray(imgRGB)
    imgtk = ImageTk.PhotoImage(image=imgPIL)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    window.after(10, update_frame)





update_frame()
window.mainloop()


cap.release()
cv2.destroyAllWindows()
