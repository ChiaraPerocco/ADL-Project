import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from PIL import Image, ImageEnhance, ImageFilter
import torch
from torchvision import transforms
import random


# Absoluten Pfad des aktuellen Skripts ermitteln
current_dir = os.path.dirname(__file__)
print(current_dir)

# Relativen Pfad zum Zielordner setzen
dataset_train = os.path.join(current_dir, "Sign Language", "train")

# Ordner für augmentierte Bilder
save_augmented_dir = os.path.join(current_dir, "Augmentation Images")

# Augmentierungsklasse
class AugmentHandFocus:
    def __init__(self, brightness_factor=1, contrast_factor=1, blur_radius=10000000, vignette_strength=500000):
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.blur_radius = blur_radius
        self.vignette_strength = vignette_strength

    def __call__(self, img):
        # Helligkeit und Kontrast anpassen
        enhancer_brightness = ImageEnhance.Brightness(img)
        img = enhancer_brightness.enhance(self.brightness_factor)

        enhancer_contrast = ImageEnhance.Contrast(img)
        img = enhancer_contrast.enhance(self.contrast_factor)

        # Hintergrund weichzeichnen (Blur)
        blurred = img.filter(ImageFilter.GaussianBlur(self.blur_radius))
        
        # Maske für den Fokusbereich (zentraler Bereich, angepasst an die Handposition)
        width, height = img.size
        mask = Image.new("L", (width, height), 0)
        vignette = Image.new("L", (width, height), 0)
        for x in range(width):
            for y in range(height):
                distance = ((x - width // 2) ** 2 + (y - height // 2) ** 2) ** 0.5
                vignette.putpixel((x, y), int(255 * (1 - min(1, distance / (width // 1.5)))) )

        # Wenden Sie die Maske an, um einen Vignette-Effekt zu erzeugen
        img = Image.composite(img, blurred, vignette)
        
        return img

# Funktion zum Speichern von augmentierten Bildern
def save_augmented_images(data_dir_train, save_dir, augment, num_images_to_save=10):
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Dataset laden
    dataset = datasets.ImageFolder(root=data_dir_train, transform=None)

    augment = AugmentHandFocus()  # Initialisierung der Augmentierung

    for idx, (img, label) in enumerate(dataset):
        if idx >= num_images_to_save:
            break

        # Anwenden der Augmentierung auf jedes Bild
        augmented_img = augment(img)
        
        # Speichern des augmentierten Bildes
        save_path = os.path.join(save_dir, f"augmented_{idx}.png")
        augmented_img.save(save_path)

        print(f"Gespeichertes Bild: {save_path}")

# Augmentierung und Speichern durchführen
save_augmented_images(dataset_train, save_augmented_dir, augment=True)


