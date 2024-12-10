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

# Import Train Loader and Valid Loader from Vision Transformer
# Import packages
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
#from torchcam.methods import SmoothGradCAMpp
#from torchcam.utils import overlay_mask

import os

# Absoluten Pfad des aktuellen Skripts ermitteln
current_dir = os.path.dirname(__file__)
print(current_dir)

# Relativen Pfad zum Zielordner setzen
dataset_train = os.path.join(current_dir, "Sign Language", "train")
dataset_val = os.path.join(current_dir, "Sign Language", "val")
dataset_test = os.path.join(current_dir, "Sign Language", "test")

num_classes = 26

# Device configuration
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
# Der Ordner, in dem die augmentierten Bilder gespeichert werden sollen
save_augmented_dir = os.path.join(current_dir, "Augmentation Images")

# Erstelle den Ordner, falls er nicht existiert
os.makedirs(save_augmented_dir, exist_ok=True)

class CustomTransform:
    def __init__(self):
        self.original_mode = None

    def __call__(self, img):
        # Speichern des Originalmodus
        self.original_mode = img.mode
        # Konvertiere Bild zu RGB, falls notwendig
        if img.mode != 'RGB':
            img = img.convert('RGB')
        return img

    def revert_to_original(self, img):
        # Rückwandeln in den ursprünglichen Modus (falls erforderlich)
        if self.original_mode:
            img = img.convert(self.original_mode)
        return img


def ensure_rgb(img):
    """Überprüft, ob das Bild im RGB-Format vorliegt. Falls nicht, wird es konvertiert."""
    # Wenn das Bild ein PIL-Image ist, umwandeln in ein NumPy-Array
    if isinstance(img, Image.Image):
        img = np.array(img)
        print(img.mode)
    
    if True:
        if len(img.shape) == 3 and img.shape[2] == 3:  # Prüfen, ob das Bild 3 Kanäle hat (kann entweder RGB oder BGR sein)
            # Überprüfen, ob das Bild im BGR-Format vorliegt (dies passiert oft bei OpenCV)
            if isinstance(img, np.ndarray) and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Konvertiere von BGR nach RGB
            return img
        else:
            raise ValueError("Das Bild muss 3 Kanäle (RGB oder BGR) haben")

class AugmentHandFocus:
    def __init__(self, brightness_factor=1.2, contrast_factor=1.5, blur_radius=10, vignette_strength=0.5):
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.blur_radius = blur_radius
        self.vignette_strength = vignette_strength
        self.blur_strength = 15

    #def __call__(self, img):
    def __call__(self, img, draw = True, handNo = 0):
        img = np.array(img)
        
        # Stelle sicher, dass das Bild im RGB-Format vorliegt
        #img = ensure_rgb(img)

        # MediaPipe Hand Modul initialisieren
        mp_hands = mp.solutions.hands
        hands = mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.5)
        mp_drawing = mp.solutions.drawing_utils

        # Hand-Erkennung anwenden
        results = hands.process(img)

        # Erstelle ein Bild, das die Handregion sichtbar lässt und den Rest schwarz macht
        output_image = np.zeros_like(img) # Empty image (output_image) of the same shape as the input image (img).Initially filled with zeros (black background).
        if True:
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

                    
                    # Erstelle eine Maske für den Handbereich
                    #hand_mask = np.zeros_like(img, dtype=np.uint8)
                    #hand_mask[y_min:y_max, x_min:x_max] = img[y_min:y_max, x_min:x_max]

                    # Kopiere den Bereich mit der Hand in das Ausgangsbild
                    #output_image = cv2.bitwise_or(output_image, hand_mask)

                    # Zeichne auch die Handlandmarks auf das Bild
                    #mp_drawing.draw_landmarks(output_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Extrahiere den scharfen Handbereich
                    hand_mask = img[y_min:y_max, x_min:x_max]

                    # Kombiniere schwarzen Hintergrund mit dem scharfen Handbereich
                    output_image[y_min:y_max, x_min:x_max] = hand_mask  # Füge den scharfen Handbereich in das unscharfe Bild ein


        transformer = CustomTransform()

        # Wieder zurück ins ursprüngliche Format konvertieren
        output_image = transformer.revert_to_original(img)


        # Rückgabe des modifizierten Bildes
        return output_image



    def save_augmented_image(self, img, save_path):
        """Speichert das augmentierte Bild im angegebenen Pfad"""
        img = img.cpu()  # Sicherstellen, dass das Bild auf der CPU ist

        if img.max() <= 1.0:  # Falls das Bild im Bereich [0, 1] ist, auf [0, 255] skalieren
            img = img * 255
        img = img.to(torch.uint8)

        transform = transforms.ToPILImage()
        pil_image = transform(img)

        pil_image.save(save_path)  # Speichern des Bildes




def get_train_valid_loader(data_dir_train, 
                           data_dir_valid,
                           batch_size, 
                           augment, 
                           save_augmented_dir,  # Optionaler Parameter für die Speicherung
                           shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # Transformationen für Validierungsdaten
    valid_transform = transforms.Compose([
        CustomTransform(),
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        normalize,
    ])

    # Augmentierung für Trainingsdaten (falls aktiviert)
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224, padding=4),
            AugmentHandFocus(), 
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

    # Lade den Trainingsdatensatz
    train_dataset = datasets.ImageFolder(root=data_dir_train, transform=train_transform)

    # Lade den Validierungsdatensatz
    valid_dataset = datasets.ImageFolder(root=data_dir_valid, transform=valid_transform)

    # DataLoader für Training und Validierung
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Wenn ein Ordner zum Speichern der augmentierten Bilder angegeben wurde, speichern wir die Bilder
    if save_augmented_dir:
        os.makedirs(save_augmented_dir, exist_ok=True)  # Erstelle den Ordner, falls er nicht existiert
        
        augment = AugmentHandFocus()  # Initialisiere die Augmentierungslogik
        for idx, (img, label) in enumerate(train_dataset):
            # Berechne den Dateipfad zum Speichern
            save_path = os.path.join(save_augmented_dir, f"augmented_{idx}.png")
            augment.save_augmented_image(img, save_path)


    return train_loader, valid_loader


# Load data with current batch size
train_loader, valid_loader = get_train_valid_loader(
    data_dir_train=dataset_train,
    data_dir_valid=dataset_val,
    batch_size=64,
    augment=True,
    save_augmented_dir=save_augmented_dir,
    shuffle=True
    )