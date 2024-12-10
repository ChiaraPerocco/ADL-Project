#####################################################################################
#
# Augmentation
#
#####################################################################################

from PIL import Image, ImageEnhance, ImageFilter
import os

"""
class AugmentHandFocus:
    def __init__(self, brightness_factor=1.2, contrast_factor=1.5, sharpness_factor = -2, blur_radius=10, vignette_strength=0.5):
        self.brightness_factor = brightness_factor
        self.contrast_factor = contrast_factor
        self.sharpness_factor = sharpness_factor
        self.blur_radius = blur_radius
        self.vignette_strength = vignette_strength

    def __call__(self, img):
        # Helligkeit anpassen
        enhancer_brightness = ImageEnhance.Brightness(img)
        img = enhancer_brightness.enhance(self.brightness_factor)

        # Kontrast anpassen
        enhancer_contrast = ImageEnhance.Contrast(img)
        img = enhancer_contrast.enhance(self.contrast_factor)

        # Schärfe anpassen
        # Creating object of Sharpness class 
        enhancer_sharpness = ImageEnhance.Sharpness(img) 
  
        # showing resultant image 
        img = enhancer_sharpness.enhance(self.sharpness_factor)

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
        img.save(save_path)  # Speichern des Bildes an einem angegebenen Pfad
"""
python -m torch --version

# Using OpenCV
class AugmentHandFocus:
    def __init__(self, scale_range=(0.8, 1.2), crop_size=(224, 224)):
        self.scale_range = scale_range  # Range for random zooming (scaling)
        self.crop_size = crop_size  # Desired crop size (height, width)

    def __call__(self, image):
        # Convert PIL image to numpy array for OpenCV processing
        image = np.array(image)
        
        # Zoom (scale) the image randomly
        scale_factor = random.uniform(self.scale_range[0], self.scale_range[1])
        height, width = image.shape[:2]
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Resize image using OpenCV
        image = cv2.resize(image, (new_width, new_height))
        
        # Randomly crop the image to the target size
        crop_x = random.randint(0, new_width - self.crop_size[1])
        crop_y = random.randint(0, new_height - self.crop_size[0])

        image = image[crop_y:crop_y + self.crop_size[0], crop_x:crop_x + self.crop_size[1]]

        # Convert back to PIL Image
        image = Image.fromarray(image)

        return image