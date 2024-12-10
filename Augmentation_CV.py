#####################################################################################
#
# Augmentation
#https://gautamaditee.medium.com/hand-recognition-using-opencv-a7b109941c88
#####################################################################################

import cv2
import mediapipe as mp
from numpy import asarray
import random
import torch
from torchvision import transforms
from PIL import Image


class AugmentHandFocus:
    def __init__(self, blur_strength=15):
        # Die Unschärfe-Stärke für den Hintergrund
        self.blur_strength = blur_strength
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands()
        self.mpDraw = mp.solutions.drawing_utils
        self.tipIds = [4, 8, 12, 16, 20]

    def __call__(self, image, draw = True, handNo = 0):
        # Image as
        image = asarray(image)

        print(type(image))
 
        #  shape
        print(image.shape)

        # Handerkennung und Verarbeitung des Bildes
        results = self.hands.process(image)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                    # Extrahiere die Hand-Landmarken (Handkeypoints)
                    xList = []
                    yList = []
                    for id, lm in enumerate(handLms.landmark):
                        # Berechne die Höhe und Breite des Bildes mit PIL
                        width, height = image.size[1], image.size[0]
                        cx, cy = int(lm.x * width), int(lm.y * height)
                        xList.append(cx)
                        yList.append(cy)

                    # Bestimme die Bounding Box der Hand
                    xmin, xmax = min(xList), max(xList)
                    ymin, ymax = min(yList), max(yList)

                    # Wende Unschärfe auf den Hintergrund an
                    blurred_image = cv2.GaussianBlur(image, (self.blur_strength, self.blur_strength), 0)

                    # Extrahiere den scharfen Handbereich
                    hand_area = image[ymin:ymax, xmin:xmax]

                    # Kombiniere den unscharfen Hintergrund mit dem scharfen Handbereich
                    result_image = blurred_image.copy()  # Erstelle eine Kopie des unscharfen Bildes
                    result_image[ymin:ymax, xmin:xmax] = hand_area  # Füge den scharfen Handbereich in das unscharfe Bild ein

                    return result_image  # Rückgabe des bearbeiteten Bildes mit Fokus auf die Hand

        return image  # Rückgabe des Bildes ohne Veränderung, wenn keine Hand gefunden wurde


    
    def save_augmented_image(self, img, save_path):
        """Speichert das augmentierte Bild im angegebenen Pfad"""
        img = img.cpu()  # Sicherstellen, dass das Bild auf der CPU ist

        if img.max() <= 1.0:  # Falls das Bild im Bereich [0, 1] ist, auf [0, 255] skalieren
            img = img * 255
        img = img.to(torch.uint8)

        transform = transforms.ToPILImage()
        pil_image = transform(img)

        pil_image.save(save_path)  # Speichern des Bildes