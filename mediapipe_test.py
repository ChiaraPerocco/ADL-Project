import cv2
import mediapipe as mp
import numpy as np
import os
 
current_dir = os.path.dirname(__file__)
print(current_dir)

# MediaPipe Hand Modul 
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.5)

# adjust paths
#input_folder = os.path.join(current_dir, "Sign Language", "train")
#output_folder = os.path.join(current_dir, "Sign Language", "train_processed")
input_folder = os.path.join(current_dir, "Sign Language 2", "test")
output_folder = os.path.join(current_dir, "Sign Language 2", "test_processed")

# create folder
os.makedirs(output_folder, exist_ok=True)

for subfolder in os.listdir(input_folder):
    subfolder_path = os.path.join(input_folder, subfolder)
    if os.path.isdir(subfolder_path):
        target_subfolder = os.path.join(output_folder, subfolder)
        os.makedirs(target_subfolder, exist_ok=True)

        for image_name in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, image_name)
            if os.path.isfile(image_path) and image_name.lower().endswith(('jpg', 'jpeg', 'png')):
                image = cv2.imread(image_path)

                results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # calculate bounding box of the hand
                        x_min = min([landmark.x for landmark in hand_landmarks.landmark])
                        x_max = max([landmark.x for landmark in hand_landmarks.landmark])
                        y_min = min([landmark.y for landmark in hand_landmarks.landmark])
                        y_max = max([landmark.y for landmark in hand_landmarks.landmark])

                        # convert the coorodinates of the hand
                        height, width, _ = image.shape
                        x_min, x_max = int(x_min * width), int(x_max * width)
                        y_min, y_max = int(y_min * height), int(y_max * height)

                        margin = 15  
                        x_min = max(0, x_min - margin)
                        x_max = min(width, x_max + margin)
                        y_min = max(0, y_min - margin)
                        y_max = min(height, y_max + margin)

                        hand_region = image[y_min:y_max, x_min:x_max]
                        zoomed_hand = cv2.resize(hand_region, (width, height), interpolation=cv2.INTER_CUBIC)

                        original_output_path = os.path.join(target_subfolder, f"original_{image_name}")
                        processed_output_path = os.path.join(target_subfolder, f"processed_{image_name}")

                        cv2.imwrite(original_output_path, image)
                        cv2.imwrite(processed_output_path, zoomed_hand)

                else:
                    original_output_path = os.path.join(target_subfolder, f"original_{image_name}")
                    cv2.imwrite(original_output_path, image)

print("Verarbeitung abgeschlossen!")