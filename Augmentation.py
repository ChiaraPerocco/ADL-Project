#####################################################################################
#
# Augmentation
#
#####################################################################################

from PIL import Image, ImageEnhance, ImageFilter
import os
from torchvision import transforms
import torch
import mediapipe as mp
import numpy as np
import cv2

def ensure_rgb(img):
    """Überprüft, ob das Bild im RGB-Format vorliegt. Falls nicht, wird es konvertiert."""
    # Wenn das Bild ein PIL-Image ist, umwandeln in ein NumPy-Array
    if isinstance(img, Image.Image):
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    
    if False:
        if len(img.shape) == 3 and img.shape[2] == 3:  # Prüfen, ob das Bild 3 Kanäle hat (kann entweder RGB oder BGR sein)
            # Überprüfen, ob das Bild im BGR-Format vorliegt (dies passiert oft bei OpenCV)
            if isinstance(img, np.ndarray) and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Konvertiere von BGR nach RGB
            return img
        else:
            raise ValueError("Das Bild muss 3 Kanäle (RGB oder BGR) haben")

class AugmentHandFocus:
    if True:
        def __init__(self, brightness_factor=1.2, contrast_factor=1.5, blur_radius=10, vignette_strength=0.5):
            #self.brightness_factor = brightness_factor
            #self.contrast_factor = contrast_factor
            #self.blur_radius = blur_radius
            #self.vignette_strength = vignette_strength
            self.blur_strength = 15

    def __call__(self, img):
        # Stelle sicher, dass das Bild im RGB-Format vorliegt
        img = ensure_rgb(img)

        # MediaPipe Hand Modul initialisieren
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils

        # Hand-Erkennung anwenden
        results = hands.process(img)

        # Erstelle ein Bild, das die Handregion sichtbar lässt und den Rest schwarz macht
        output_image = np.zeros_like(img)

        # Wenn Hände erkannt wurden, berechne die Handregion und erstelle die Maske
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Berechne das minimale und maximale Koordinaten der Hand (Bounding Box)
                x_min = min([landmark.x for landmark in hand_landmarks.landmark])
                x_max = max([landmark.x for landmark in hand_landmarks.landmark])
                y_min = min([landmark.y for landmark in hand_landmarks.landmark])
                y_max = max([landmark.y for landmark in hand_landmarks.landmark])

                # Konvertiere die Normalisierten Koordinaten (0-1) in Bildkoordinaten
                height, width, _ = img.shape
                x_min, x_max = int(x_min * width), int(x_max * width)
                y_min, y_max = int(y_min * height), int(y_max * height)

                if False:
                    # Erstelle eine Maske für den Handbereich
                    hand_mask = np.zeros_like(img, dtype=np.uint8)
                    hand_mask[y_min:y_max, x_min:x_max] = img[y_min:y_max, x_min:x_max]

                    # Kopiere den Bereich mit der Hand in das Ausgangsbild
                    output_image = cv2.bitwise_or(output_image, hand_mask)

                    # Zeichne auch die Handlandmarks auf das Bild
                    mp_drawing.draw_landmarks(output_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    #output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)

                blurred_image = cv2.GaussianBlur(img, (self.blur_strength, self.blur_strength), 0)
                hand_area = img[y_min:y_max, x_min:x_max]

                result_image = blurred_image.copy()
                result_image[y_min:y_max, x_min:x_max] = hand_area


        # Rückgabe des modifizierten Bildes
        return result_image



    def save_augmented_image(self, img, save_path):
        """Speichert das augmentierte Bild im angegebenen Pfad"""
        img = img.cpu()  # Sicherstellen, dass das Bild auf der CPU ist

        if img.max() <= 1.0:  # Falls das Bild im Bereich [0, 1] ist, auf [0, 255] skalieren
            img = img * 255
        img = img.to(torch.uint8)

        transform = transforms.ToPILImage()
        pil_image = transform(img)

        pil_image.save(save_path)  # Speichern des Bildes