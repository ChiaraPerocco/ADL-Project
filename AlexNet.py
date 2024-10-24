import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support

# import data set
dataset_train = r'C:\Users\annar\Documents\Master\Advanced Deep Learning\Facial emotion data set\facial_emotion_dataset\dataset_output - Kopie\train'  # Pfad zu den Trainingsdaten
dataset_val = r'C:\Users\annar\Documents\Master\Advanced Deep Learning\Facial emotion data set\facial_emotion_dataset\dataset_output - Kopie\val'  # Pfad zu den Validierungsdaten
dataset_test = r'C:\Users\annar\Documents\Master\Advanced Deep Learning\Facial emotion data set\facial_emotion_dataset\dataset_output - Kopie\test'  # Pfad zu den Testdaten

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        transforms.Resize((224, 224)), # AlexNet erwartet eine Bildgröße von 227x227
        transforms.ToTensor(),
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
    batch_size=64,
    augment=True,  # Augmentierung für Trainingsdaten
    shuffle=True
)

test_loader = get_test_loader(
    data_dir= dataset_test, # Pfad zu den Testdaten
    batch_size=64
)


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

#Testing
with torch.no_grad():
    correct = 0
    total = 0

    all_test_labels = []
    all_test_preds = []

    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        # Speichern der Labels und Vorhersagen für Berechnung
        all_test_labels.extend(labels.cpu().numpy())
        all_test_preds.extend(predicted.cpu().numpy())

        del images, labels, outputs

            # Berechnung der Test Accuracy
    test_accuracy = 100 * correct / total
    print('Accuracy of the network on the {} test images: {} %'.format(total, test_accuracy))
    
    # Berechnung von Precision, Recall, F1-Score für Test
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(all_test_labels, all_test_preds, average='weighted')
    
    # Ausgabe und Speicherung der Testmetriken
    print(f'Test Precision: {test_precision:.4f}')
    print(f'Test Recall: {test_recall:.4f}')
    print(f'Test F1-Score: {test_f1:.4f}')


