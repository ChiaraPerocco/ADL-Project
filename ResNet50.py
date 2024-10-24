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
dataset_train = r'C:\Users\annar\Documents\Master\Advanced Deep Learning\Facial emotion data set\facial_emotion_dataset\dataset_output - Kopie\train'  # Pfad zu den Trainingsdaten
dataset_val = r'C:\Users\annar\Documents\Master\Advanced Deep Learning\Facial emotion data set\facial_emotion_dataset\dataset_output - Kopie\val'  # Pfad zu den Validierungsdaten
dataset_test = r'C:\Users\annar\Documents\Master\Advanced Deep Learning\Facial emotion data set\facial_emotion_dataset\dataset_output - Kopie\test'  # Pfad zu den Testdaten


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set hyperparameters
num_epochs = 3
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


    # Berechnung der Test Accuracy
    test_acc_resNet50 = test_corrects.double() / len(test_loader.dataset)
    print(f'Test Accuracy: {test_acc_resNet50:.4f}')
    
    # Berechnung von Precision, Recall und F1-Score
    precision_resNet50, recall_resNet50, f1_resNet50, _ = precision_recall_fscore_support(all_labels_resNet50, all_preds_resNet50, average='weighted')
    
    print(f'Precision: {precision_resNet50:.4f}')
    print(f'Recall: {recall_resNet50:.4f}')
    print(f'F1-Score: {f1_resNet50:.4f}')
    
    # Rückgabe der Metriken
    return test_acc_resNet50.item(), precision_resNet50, recall_resNet50, f1_resNet50, all_labels_resNet50, all_preds_resNet50

# Testen auf Testdaten und Speichern der Metriken
test_acc_resNet50, precision_resNet50, recall_resNet50, f1_resNet50, all_labels_resNet50, all_preds_resNet50 = test_model(model, test_loader)

# Zugriff auf die Metriken im weiteren Code
print(f'Saved Test Accuracy: {test_acc_resNet50:.4f}')
print(f'Saved Precision: {precision_resNet50:.4f}')
print(f'Saved Recall: {recall_resNet50:.4f}')
print(f'Saved F1-Score: {f1_resNet50:.4f}')
# Testen auf Testdaten
test_model(model, test_loader)


###################################################################################################
#
# AlexNet
#
###################################################################################################
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(6400, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))

    def forward(self, x):
        out = self.layer1(x)
        print("Layer 1 Output Shape:", out.shape)
        out = self.layer2(out)
        print("Layer 2 Output Shape:", out.shape)
        out = self.layer3(out)
        print("Layer 3 Output Shape:", out.shape)
        out = self.layer4(out)
        print("Layer 4 Output Shape:", out.shape)
        out = self.layer5(out)
        print("Layer 5 Output Shape:", out.shape)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out
    
#setting Hyperparameters
num_classes = 10
num_epochs = 20
batch_size = 64
learning_rate = 0.005

model = AlexNet(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

# Train the model
total_step = len(train_loader)

total_step = len(train_loader)


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):  
        
        print(images.shape)
    
        # Move tensors to the configured device
        images = images.to(device)
        labels = labels.to(device)

        print(images.shape)
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

    # Validation
    with torch.no_grad():
        correct = 0
        total = 0

        all_val_labels = []
        all_val_preds = []

        for images, labels in valid_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Speichern der Labels und Vorhersagen für Berechnung
            all_val_labels.extend(labels.cpu().numpy())
            all_val_preds.extend(predicted.cpu().numpy())
            
            del images, labels, outputs

        # Berechnung der Validation Accuracy
        val_accuracy = 100 * correct / total

        print('Accuracy of the network on the {} validation images: {} %'.format(total, val_accuracy)) 

        # Berechnung von Precision, Recall, F1-Score für Validation
        val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(all_val_labels, all_val_preds, average='weighted')
        
        # Ausgabe und Speicherung der Validierungsmetriken
        print(f'Validation Precision: {val_precision:.4f}')
        print(f'Validation Recall: {val_recall:.4f}')
        print(f'Validation F1-Score: {val_f1:.4f}')

# Testing
with torch.no_grad():
    correct = 0
    total = 0

    all_test_labels_alexNet = []
    all_test_preds_alexNet = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Speichern der Labels und Vorhersagen für spätere Berechnung
        all_test_labels_alexNet.extend(labels.cpu().numpy())
        all_test_preds_alexNet.extend(predicted.cpu().numpy())

        del images, labels, outputs

    # Berechnung der Test Accuracy
    test_accuracy_alexNet = correct / total
    print('Accuracy of the network on the {} test images: {} %'.format(total, test_accuracy_alexNet))
    
    # Berechnung von Precision, Recall, F1-Score für den Test
    test_precision_alexNet, test_recall_alexNet, test_f1_alexNet, _ = precision_recall_fscore_support(all_test_labels_alexNet, all_test_preds_alexNet, average='weighted')
    
    # Ausgabe und Speicherung der Testmetriken
    print(f'Test Precision AlexNet: {test_precision_alexNet:.4f}')
    print(f'Test Recall AlexNet: {test_recall_alexNet:.4f}')
    print(f'Test F1-Score AlexNet: {test_f1_alexNet:.4f}')




###################################################################################################
#
# Evaluation of the models 
#
###################################################################################################
###################################################################################################
###Comparison of the classification methods in the testing stage
###################################################################################################
import matplotlib.pyplot as plt 

# set width of bar 
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8)) 

# set height of bar 
AlexNet_bar = [test_f1_alexNet, test_accuracy_alexNet, test_precision_alexNet, test_recall_alexNet] 
ResNet50_bar = [f1_resNet50, test_acc_resNet50, precision_resNet50, recall_resNet50] 

# Set position of bar on X axis 
br1 = np.arange(len(AlexNet_bar)) 
br2 = [x + barWidth for x in br1] 

# Make the plot
plt.bar(br1, AlexNet_bar, color ='r', width = barWidth, 
        edgecolor ='grey', label ='AlexNet') 
plt.bar(br2, ResNet50_bar, color ='g', width = barWidth, 
        edgecolor ='grey', label ='ResNet50') 

# Adding Xticks 
plt.xlabel('Comparison of the classification methods in the testing stage', fontweight ='bold', fontsize = 15) 
plt.ylabel('Accuracy', fontweight ='bold', fontsize = 15) 
plt.xticks([r + barWidth for r in range(len(AlexNet_bar))], 
        ['F-Measure', 'Accuracy', 'Precision', 'Recall'])

plt.legend()
plt.show() 
plt.clf()  # Löscht die Figur für den nächsten Plot



###################################################################################################
###Confusion Matrix
###################################################################################################
# Confusion Matrix: ResNet50
conf_matrix_resNet50 = confusion_matrix(all_labels_resNet50, all_preds_resNet50)

# Visualisierung der Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_resNet50, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.title('Confusion Matrix ResNet50')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

plt.clf()  # Löscht die Figur für den nächsten Plot

# Confusion Matrix: AlexNet
conf_matrix_alexNet = confusion_matrix(all_test_labels_alexNet, all_test_preds_alexNet)

# Visualisierung der Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_alexNet, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.title('Confusion Matrix AlexNet')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()