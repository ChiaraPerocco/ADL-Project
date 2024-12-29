import cv2
import mediapipe as mp
import numpy as np
import os
import random


# Absoluten Pfad des aktuellen Skripts ermitteln
current_dir = os.path.dirname(__file__)
print(current_dir)

# MediaPipe Hand Modul initialisieren
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.5)

# Pfade anpassen
input_folder = os.path.join(current_dir, "Sign Language", "val")
output_folder = os.path.join(current_dir, "Sign Language", "val_processed")

# Erstelle den Zielordner, falls er nicht existiert
os.makedirs(output_folder, exist_ok=True)

# Definiere ein Limit für die Anzahl neuer fokussierter Bilder
max_processed_images = 500
processed_count = 0

def image_processing(input_folder, output_folder):
    # MediaPipe Hand Modul initialisieren
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.5)

    # Durchlaufe alle Unterordner
    for subfolder in os.listdir(input_folder):
        subfolder_path = os.path.join(input_folder, subfolder)
        if os.path.isdir(subfolder_path):
            # Ziel-Unterordner erstellen
            target_subfolder = os.path.join(output_folder, subfolder)
            os.makedirs(target_subfolder, exist_ok=True)

            # Alle Bilder im aktuellen Unterordner durchlaufen
            for image_name in os.listdir(subfolder_path):
                if processed_count >= max_processed_images:
                    break  # Beende, wenn das Limit erreicht ist

                image_path = os.path.join(subfolder_path, image_name)
                if os.path.isfile(image_path) and image_name.lower().endswith(('jpg', 'jpeg', 'png')):
                    # Bild laden
                    image = cv2.imread(image_path)

                    # Hand-Erkennung anwenden
                    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                    # Wenn Hände erkannt wurden, speichere nur zufällig ausgewählte fokussierte Bilder
                    if results.multi_hand_landmarks and random.random() < 0.5:  # 50% Wahrscheinlichkeit
                        for hand_landmarks in results.multi_hand_landmarks:
                            # Berechne Bounding Box der Hand (bleibt unverändert)
                            x_min = min([landmark.x for landmark in hand_landmarks.landmark])
                            x_max = max([landmark.x for landmark in hand_landmarks.landmark])
                            y_min = min([landmark.y for landmark in hand_landmarks.landmark])
                            y_max = max([landmark.y for landmark in hand_landmarks.landmark])

                            # Konvertiere Koordinaten in Bildkoordinaten
                            height, width, _ = image.shape
                            x_min, x_max = int(x_min * width), int(x_max * width)
                            y_min, y_max = int(y_min * height), int(y_max * height)

                            # Rand hinzufügen
                            margin = 15
                            x_min = max(0, x_min - margin)
                            x_max = min(width, x_max + margin)
                            y_min = max(0, y_min - margin)
                            y_max = min(height, y_max + margin)

                            # Extrahiere und vergrößere den Bereich mit der Hand
                            hand_region = image[y_min:y_max, x_min:x_max]
                            zoomed_hand = cv2.resize(hand_region, (width, height), interpolation=cv2.INTER_CUBIC)

                            # Speichere nur das gezoomte Bild
                            processed_output_path = os.path.join(target_subfolder, f"processed_{image_name}")
                            cv2.imwrite(processed_output_path, zoomed_hand)

                            # Zähle die Anzahl der generierten fokussierten Bilder
                            processed_count += 1
                            if processed_count >= max_processed_images:
                                break

                    else:
                        # Falls keine Hand erkannt wird, speichere nur das Original
                        original_output_path = os.path.join(target_subfolder, f"original_{image_name}")
                        cv2.imwrite(original_output_path, image)

print("Verarbeitung abgeschlossen!")
