import os
import cv2
import mediapipe as mp


def image_processing(input_folder, output_folder):
    # MediaPipe Hand Modul initialisieren
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.5)

    # Durchlaufe alle Dateien und Unterordner im input_folder
    for root, dirs, files in os.walk(input_folder):
        # Berechne den relativen Pfad für das Ziel
        relative_path = os.path.relpath(root, input_folder)
        target_subfolder = os.path.join(output_folder, relative_path)
        os.makedirs(target_subfolder, exist_ok=True)

        # Alle Bilder im aktuellen Verzeichnis (root) durchlaufen
        for image_name in files:
            image_path = os.path.join(root, image_name)
            if os.path.isfile(image_path) and image_name.lower().endswith(('jpg', 'jpeg', 'png')):
                # Bild laden
                image = cv2.imread(image_path)

                # Hand-Erkennung anwenden
                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                # Wenn Hände erkannt wurden, berechne die Handregion und erstelle das gezoomte Bild
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
                        margin = 15  # Anpassbar je nach Bedarf
                        x_min = max(0, x_min - margin)
                        x_max = min(width, x_max + margin)
                        y_min = max(0, y_min - margin)
                        y_max = min(height, y_max + margin)

                        # Extrahiere und vergrößere den Bereich mit der Hand
                        hand_region = image[y_min:y_max, x_min:x_max]
                        zoomed_hand = cv2.resize(hand_region, (width, height), interpolation=cv2.INTER_CUBIC)

                        # Pfad für die Speicherung des verarbeiteten Bildes
                        processed_output_path = os.path.join(target_subfolder, f"processed_{image_name}")

                        # Speichere nur das verarbeitete Bild (gezoomte Hand)
                        cv2.imwrite(processed_output_path, zoomed_hand)

                else:
                    # Falls keine Hand erkannt wird, speichere nur das Originalbild
                    #original_output_path = os.path.join(target_subfolder, f"original_{image_name}")
                    #cv2.imwrite(original_output_path, image)
                    print("Hand wurde nicht erkannt. Nehme noch ein Foto auf.")

    print("Verarbeitung abgeschlossen!")
