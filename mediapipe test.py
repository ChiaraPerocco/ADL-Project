import cv2
import mediapipe as mp
import numpy as np

# MediaPipe Hand Modul initialisieren
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

path = r"\\nas.ads.mwn.de\hm-aweb10ac\Benutzer\Dokumente\ADLProjekt\ADL-Project\Sign Language\train\Q\Q (287).jpg"

# Bild laden
image = cv2.imread(path)

# Das Bild umkehren (von BGR zu RGB, da MediaPipe im RGB-Format arbeitet)
#image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

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

        # Erstelle eine Maske für den Handbereich
        hand_mask = np.zeros_like(image, dtype=np.uint8)
        hand_mask[y_min:y_max, x_min:x_max] = image[y_min:y_max, x_min:x_max]

        # Kopiere den Bereich mit der Hand in das Ausgangsbild
        output_image = cv2.bitwise_or(output_image, hand_mask)

        # Zeichne auch die Handlandmarks auf das Bild
        mp_drawing.draw_landmarks(output_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

# Das Bild anzeigen
cv2.imshow('Hand Recognition', output_image)

# Warten auf eine Taste, um das Fenster zu schließen
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: Speichern des Ergebnisses
cv2.imwrite('hand_erkanntes_bild.jpg', output_image)
