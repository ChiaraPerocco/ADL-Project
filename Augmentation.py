#####################################################################################
#
# Augmentation
#
#####################################################################################

from PIL import Image, ImageEnhance, ImageFilter
import os
from torchvision import transforms
import torch

class AugmentHandFocus:
    def __init__(self, brightness_factor=1.2, contrast_factor=1.5, blur_radius=10, vignette_strength=0.5):
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
                vignette.putpixel((x, y), int(255 * (1 - min(1, distance / (width // 1.5)))))


        # Wenden Sie die Maske an, um einen Vignette-Effekt zu erzeugen
        img = Image.composite(img, blurred, vignette)
        
        return img

    def save_augmented_image(self, img, save_path):
        """Speichert das augmentierte Bild im angegebenen Pfad"""
        img = img.cpu()  # Sicherstellen, dass das Bild auf der CPU ist

        if img.max() <= 1.0:  # Falls das Bild im Bereich [0, 1] ist, auf [0, 255] skalieren
            img = img * 255
        img = img.to(torch.uint8)

        transform = transforms.ToPILImage()
        pil_image = transform(img)

        pil_image.save(save_path)  # Speichern des Bildes