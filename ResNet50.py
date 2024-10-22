###############################################################################################
#
# Resnet-50
# source: https://moiseevigor.github.io/software/2022/12/18/one-pager-training-resnet-on-imagenet/
# source: https://pytorch.org/vision/stable/models.html
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

# import data set
dataset_train = r'C:\Users\annar\Documents\Master\Advanced Deep Learning\European License data set\dataset_final\trainneu'  # Pfad zu den Trainingsdaten
dataset_val = r'C:\Users\annar\Documents\Master\Advanced Deep Learning\European License data set\dataset_final\valneu'  # Pfad zu den Validierungsdaten
dataset_test = r'C:\Users\annar\Documents\Master\Advanced Deep Learning\European License data set\dataset_final\testneu'  # Pfad zu den Testdaten

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set hyperparameters
num_epochs = 10
batch_size = 64
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

# Testen des Modells auf den Testdaten
def test_model(model, test_loader):
    model.eval()
    test_corrects = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)

    test_acc = test_corrects.double() / len(test_loader.dataset)
    print(f'Test Accuracy: {test_acc:.4f}')

# Testen auf Testdaten
test_model(model, test_loader)

# Evaluate the model

# Calculate prediction for test data
#predictions = model.predict(test_loader)

# Confusion Matrix
#confusion_matrix = metrics.confusion_matrix(actual, predictions) 