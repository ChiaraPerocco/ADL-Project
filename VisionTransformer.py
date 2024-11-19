import torch
import os
from transformers import ViTForImageClassification, ViTImageProcessor
from IPython.display import display, Image
from PIL import Image as img


#File_Name = r"C:\Studium\Data Analytics, M.Sc\Advanced Deep Learning\Team Project Empty\images\basketball.jpeg"

#display(Image(File_Name, width=700, height=400))

#image_array = img.open(File_Name)


directory = r'C:\Studium\Data Analytics, M.Sc\Advanced Deep Learning\dataset_final\test'

feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

'''
inputs = feature_extractor(image_array, return_tensors = "pt")
outputs = model(**inputs)
logits = outputs.logits

logits.shape

predicted_class_idx = logits.argmax(-1).item()x)
print("Predicted Clas
print(predicted_class_ids :", model.config.id2label[predicted_class_idx])
'''

model.eval()

# Über alle Bilder im Verzeichnis iterieren
for filename in os.listdir(directory):
    if filename.endswith(('.png', '.jpg', '.jpeg')):  # Nur Bilddateien
        file_path = os.path.join(directory, filename)
        
        # Bild laden und vorverarbeiten
        image_array = img.open(file_path).convert("RGB")
        inputs = feature_extractor(image_array, return_tensors="pt")
        
        # Vorhersage durchführen
        outputs = model(**inputs)
        
        logits = outputs.logits
        predicted_class_idx = logits.argmax(-1).item()
        
        # Ergebnis ausgeben
        print(f"Datei: {filename} - Predicted Class: {model.config.id2label[predicted_class_idx]}")