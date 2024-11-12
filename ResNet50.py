###############################################################################################
#
# Resnet-50
# source: https://moiseevigor.github.io/software/2022/12/18/one-pager-training-resnet-on-imagenet/
# source: https://pytorch.org/vision/stable/models.html
#
###############################################################################################
# Import packages
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from PIL import Image
from torchvision.models import resnet50, ResNet50_Weights
import torch.optim as optim
from torchvision import models
from sklearn import metrics  # for confusion matrix
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns # Confusion Matrix

# Import data set
dataset_train = r'C:\Users\annar\Documents\Master\Advanced Deep Learning\Facial emotion data set\facial_emotion_dataset\dataset_output\train'  # Pfad zu den Trainingsdaten
dataset_val = r'C:\Users\annar\Documents\Master\Advanced Deep Learning\Facial emotion data set\facial_emotion_dataset\dataset_output\val'  # Pfad zu den Validierungsdaten
dataset_test = r'C:\Users\annar\Documents\Master\Advanced Deep Learning\Facial emotion data set\facial_emotion_dataset\dataset_output\test'  # Pfad zu den Testdaten


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set hyperparameters
num_epochs = 3
batch_size = 128
learning_rate = 0.001

def get_train_valid_loader(data_dir_train, #Verzeichnis, in dem der Datensatz gespeichert wird (oder heruntergeladen werden soll).
                           data_dir_valid,
                           batch_size, #Anzahl der Bilder pro Batch (Mini-Batch) während des Trainings oder der Validierung.
                           augment, #Boolescher Wert, der angibt, ob Datenaugmentation verwendet werden soll (z. B. zufälliges Beschneiden oder Spiegeln der Bilder).
                           shuffle=True): #Ob die Daten vor dem Aufteilen in Trainings- und Validierungssets zufällig gemischt werden sollen
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # Transformationen für Validierungsdaten
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)), # Ändern der Größe auf 224x224 Pixel
        transforms.ToTensor(), # Umwandlung in einen Tensor
        normalize,
    ])

    # Augmentierung für Trainingsdaten (falls aktiviert)
    if augment:
        train_transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256 first
        transforms.RandomCrop(224, padding=4), #zufälliges zuschneiden auf 224x224
        transforms.RandomHorizontalFlip(), #zufälliges horizontales spiegeln
        transforms.ToTensor(),
        normalize,
    ])
        
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

    # Lade den Trainingsdatensatz (ImageFolder passt zur Ordnerstruktur)
    train_dataset = datasets.ImageFolder(root=data_dir_train, transform=train_transform)

    # Lade den Validierungsdatensatz (separate Validierungsdaten)
    valid_dataset = datasets.ImageFolder(root=data_dir_valid, transform=valid_transform)

    # DataLoader für Training und Validierung
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader

def get_test_loader(data_dir,
                    batch_size,
                    shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),   #Bildgröße anpassen
        transforms.ToTensor(),
        normalize,
    ])
    

    # Lade den Testdatensatz
    test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Erstellen des DataLoaders für Testdaten
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return test_loader

#Datensätze laden 
train_loader, valid_loader = get_train_valid_loader(
    data_dir_train= dataset_train, # Pfad zu den Trainingsdaten
    data_dir_valid= dataset_val, # Pfad zu den Validierungsdaten
    batch_size=batch_size,
    augment=True,  # Augmentierung für Trainingsdaten
    shuffle=True # Daten werden vor jedem Training zufällig durchmischt
)

test_loader = get_test_loader(
    data_dir= dataset_test, # Pfad zu den Testdaten
    batch_size=batch_size
)

# Lädt das vortrainierte Resnet50 Modell 
#    vortrainierte Gewichte
#model = resnet50(pretrained=True) 
model = resnet50(weights=ResNet50_Weights.DEFAULT) # or ResNet50_Weights.IMAGENET1K_V2

# Initialize the Weight Transforms
#weights = ResNet50_Weights.DEFAULT
#preprocess = weights.transforms()

# Apply it to the input image
#img_transformed = preprocess(img)

# Set the model to run on the device
model = model.to(device)

#Wir trainieren nur die letzte Schicht (den Fully-Connected Layer), die für die Klassifikation verantwortlich ist.
# Trainieren der letzten Schicht (Fully-Connected Layer), die für Klassifikation verantwortlich ist
# Anpassen des letzten Fully-Connected-Layers an die Anzahl der Klassen
#num_classes = len(dataset_test.classes)  # Anzahl der Klassen 
#model.fc = nn.Linear(model.fc.in_features, num_classes)

# Trainingsparameter definieren
# Verlustfunktion und Optimierer
criterion = nn.CrossEntropyLoss() # Verlustfunktion
optimizer = optim.Adam(model.parameters(), lr=learning_rate) # Adam-Optimierer nur für den Klassifikator (die letzten Schichten)
#Der Adam-Optimierer: Gewichte der Fully-Connected-Schicht (die neuen trainierbaren Parameter) zu aktualisieren. Die Lernrate (lr=0.001) gibt die Schrittweite bei jedem Optimierungsschritt an.

# Trainings- und Validierungsfunktion
def train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs=num_epochs):
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')

        # Training
        model.train() # Setzt das Modell in den Trainingsmodus
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            # Move input and label tensors to the device
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Gradienten zurücksetzen
            optimizer.zero_grad()

            # Vorwärtsdurchlauf
            outputs = model(inputs) # Modellvorhersagen für das Eingabebatch
            _, preds = torch.max(outputs, 1) # Vorhersagen (Klassen mit dem höchsten Score) aus den Ausgaben
                                             # preds: Enthält die vorhergesagte Klasse für jedes Bild im Batch.
                                             # torch.max(outputs, 1): Diese Funktion findet den Index der maximalen Logit-Werte (die wahrscheinlichste Klasse) entlang der Dimension 1 (also der Klassen).
            loss = criterion(outputs, labels) # Berechnung des Verlusts zwischen Vorhersagen und tatsächlichen Labels

            # Rückwärtsdurchlauf und Optimierung
            loss.backward() # Gradientenberechnung durch Backpropagation
            optimizer.step() # Optimierungsschritt: Aktualisieren der Modellparameter

            # Statistiken berechnen
            running_loss += loss.item() * inputs.size(0) # Akkumulierung des Verlusts für das gesamte Batch
            running_corrects += torch.sum(preds == labels.data) # Akkumulierung der korrekten Vorhersagen

        epoch_loss = running_loss / len(train_loader.dataset)  # Durchschnittlicher Verlust über den gesamten Datensatz
        epoch_acc = running_corrects.double() / len(train_loader.dataset) # Genauigkeit über den gesamten Datensatz

        print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # Validation
        model.eval() # Setzt Modell in den Evaluationsmodus
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

                val_loss = val_loss / len(valid_loader.dataset)
                val_acc = val_corrects.double() / len(valid_loader.dataset)

        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}')

    return model

# Trainiere das Modell
num_epochs = num_epochs
model = train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs=num_epochs)

# Assuming class_names list is defined as follows
class_names = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

# Testen des Modells auf den Testdaten
def test_model(model, test_loader):
    model.eval()
    test_corrects = 0
    all_labels_resNet50 = []
    all_preds_resNet50 = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)


            # Speichern der Labels und Vorhersagen für spätere Auswertungen
            all_labels_resNet50.extend(labels.cpu().numpy())
            all_preds_resNet50.extend(preds.cpu().numpy())

    # Convert numerical labels and predictions to class names
    true_labels = [class_names[i] for i in all_labels_resNet50]
    predicted_labels = [class_names[i] for i in all_preds_resNet50]


    # Berechnung der Test Accuracy
    test_acc_resNet50 = test_corrects.double() / len(test_loader.dataset)
    print(f'Test Accuracy: {test_acc_resNet50:.4f}')
    
    # Berechnung von Precision, Recall und F1-Score
    precision_resNet50, recall_resNet50, f1_resNet50, _ = precision_recall_fscore_support(all_labels_resNet50, all_preds_resNet50, average='weighted')
    
    print(f'Test Accuracy: {test_acc_resNet50:.4f}')
    print(f'Precision: {precision_resNet50:.4f}')
    print(f'Recall: {recall_resNet50:.4f}')
    print(f'F1-Score: {f1_resNet50:.4f}')
    print(f'Labels Testdaten: {true_labels}')
    print(f'vorhergesagte Testdaten: {predicted_labels}')

    # Rückgabe der Metriken
    return test_acc_resNet50.item(), precision_resNet50, recall_resNet50, f1_resNet50, all_labels_resNet50, all_preds_resNet50, true_labels, predicted_labels

# Testen auf Testdaten und Speichern der Metriken
test_acc_resNet50, precision_resNet50, recall_resNet50, f1_resNet50, all_labels_resNet50, all_preds_resNet50, true_labels, predicted_labels = test_model(model, test_loader)


# Testen auf Testdaten
test_model(model, test_loader)


# Confusion Matrix: ResNet50
import matplotlib.pyplot as plt 

conf_matrix_resNet50 = confusion_matrix(true_labels, predicted_labels)

# Visualisierung der Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_resNet50, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.title('Confusion Matrix ResNet50')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
