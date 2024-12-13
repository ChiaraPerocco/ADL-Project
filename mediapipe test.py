import cv2
import mediapipe as mp
import numpy as np
import os

# Absoluten Pfad des aktuellen Skripts ermitteln
current_dir = os.path.dirname(__file__)
print(current_dir)

# MediaPipe Hand Modul initialisieren
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Pfade anpassen
input_folder =  os.path.join(current_dir, "Sign Language", "val")
output_folder = os.path.join(current_dir, "Sign Language", "val_processed")

# Erstelle den Zielordner, falls er nicht existiert
os.makedirs(output_folder, exist_ok=True)

# Durchlaufe alle Unterordner
for subfolder in os.listdir(input_folder):
    subfolder_path = os.path.join(input_folder, subfolder)
    if os.path.isdir(subfolder_path):
        # Ziel-Unterordner erstellen
        target_subfolder = os.path.join(output_folder, subfolder)
        os.makedirs(target_subfolder, exist_ok=True)

        # Alle Bilder im aktuellen Unterordner durchlaufen
        for image_name in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_name)
            if os.path.isfile(image_path) and image_name.lower().endswith(('jpg', 'jpeg', 'png')):
                # Bild laden
                image = cv2.imread(image_path)

                # Hand-Erkennung anwenden
                results = hands.process(image)

                # Erstelle ein Bild, das die Handregion sichtbar lässt und den Rest schwarz macht
                output_image = np.zeros_like(image)

                # Wenn Hände erkannt wurden, berechne die Handregion und erstelle die Maske
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Berechne das minimale und maximale Koordinaten der Hand (Bounding Box)
                        x_min = min([landmark.x for landmark in hand_landmarks.landmark])
                        x_max = max([landmark.x for landmark in hand_landmarks.landmark])
                        y_min = min([landmark.y for landmark in hand_landmarks.landmark])
                        y_max = max([landmark.y for landmark in hand_landmarks.landmark])

                        # Konvertiere die Normalisierten Koordinaten (0-1) in Bildkoordinaten
                        height, width, _ = image.shape
                        x_min, x_max = int(x_min * width), int(x_max * width)
                        y_min, y_max = int(y_min * height), int(y_max * height)

                        # Füge einen kleinen Rand um die Bounding Box hinzu
                        margin = 10  # Anpassbar je nach Bedarf
                        x_min = max(0, x_min - margin)
                        x_max = min(width, x_max + margin)
                        y_min = max(0, y_min - margin)
                        y_max = min(height, y_max + margin)

                        # Erstelle eine Maske für den Handbereich
                        hand_mask = np.zeros_like(image, dtype=np.uint8)
                        hand_mask[y_min:y_max, x_min:x_max] = image[y_min:y_max, x_min:x_max]

                        # Kopiere den Bereich mit der Hand in das Ausgangsbild
                        output_image = cv2.bitwise_or(output_image, hand_mask)

                # Originalbild und bearbeitetes Bild speichern
                original_output_path = os.path.join(target_subfolder, f"original_{image_name}")
                processed_output_path = os.path.join(target_subfolder, f"processed_{image_name}")
                cv2.imwrite(original_output_path, image)  # Original speichern
                cv2.imwrite(processed_output_path, output_image)  # Bearbeitetes Bild speichern