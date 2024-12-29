import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.optim as optim
import os
import wandb
import copy
from sklearn.metrics import precision_recall_fscore_support

# Definiere den absoluten Pfad des aktuellen Skripts
current_dir = os.path.dirname(__file__)
print(current_dir)

# Pfade zu den Datensätzen
dataset_train = os.path.join(current_dir, "Sign Language", "train_processed")
dataset_val = os.path.join(current_dir, "Sign Language", "val_processed")
dataset_test = os.path.join(current_dir, "Sign Language", "test_processed")

# Hyperparameter
num_classes = 26

batch_size = 64
learning_rate = 1e-4
num_epochs = 30
num_workers = 2  # Dies bleibt in der main Funktion, wie du es gewünscht hast

# Device configuration
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

# W&B Initialisierung – nur einmal in der main() Funktion
def initialize_wandb():
    wandb.init(project="ViT_model_dataset2_7", config={
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'num_epochs': num_epochs
    })
    wandb.run.name = "Final_Run"  # Optional, benenne den Run

# Transformationen und DataLoader
def get_train_valid_loader(data_dir_train, data_dir_valid, batch_size, augment, shuffle=True):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    # Validierungstransformation
    valid_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    # Trainingsdaten Augmentierung
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),
            transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0)),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    train_dataset = datasets.ImageFolder(root=data_dir_train, transform=train_transform)
    valid_dataset = datasets.ImageFolder(root=data_dir_valid, transform=valid_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader

def get_test_loader(data_dir, batch_size, shuffle=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

    return test_loader

# Modell initialisieren
def initialize_model(num_classes):
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

    # Alle Schichten einfrieren
    for param in model.parameters():
        param.requires_grad = False

    # Den Klassifikator (head) anpassen
    model.heads.head = nn.Sequential(
        nn.Dropout(0.3),  # Dropout für Regularisierung
        nn.Linear(model.heads.head.in_features, num_classes)
    )

    model = model.to(device)
    return model

# Training des finalen Modells
def train_final_model(model, train_loader, valid_loader, best_params):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.heads.head.parameters(), lr=best_params['learning_rate'], weight_decay=0.005)

    train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

    best_loss = float('inf')
    best_model_weights = None
    patience = 3  # Für Early Stopping

    for epoch in range(best_params['num_epochs']):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())

        # Validierung
        model.eval()
        val_loss = 0.0
        val_corrects = 0
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                val_corrects += torch.sum(preds == labels.data)

        val_loss = val_loss / len(valid_loader.dataset)
        val_acc = val_corrects.double() / len(valid_loader.dataset)

        val_losses.append(val_loss)
        val_accuracies.append(val_acc.item())

        print(f"Epoch {epoch+1}/{best_params['num_epochs']}, "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Logge Metriken zu W&B
        wandb.log({
            "train_loss": epoch_loss, "train_acc": epoch_acc,
            "val_loss": val_loss, "val_acc": val_acc
        })

        # Early Stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience = 3  # Resette Patience
        else:
            patience -= 1
            if patience == 0:
                print("Early Stopping")
                break

    # Beste Modellgewichte laden
    model.load_state_dict(best_model_weights)

    # Speichern des besten Modells
    torch.save(model.state_dict(), "ViT_model_dataset2_7.pth")
    print("Bestes Modell gespeichert als 'ViT_model_dataset2_7.pth'.")

    return model

# Testen des Modells
def test_model(model, test_loader):
    model.eval()
    test_corrects = 0
    all_labels, all_preds = [], []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    test_acc = test_corrects.double() / len(test_loader.dataset)
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')

    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")

    return test_acc, precision, recall, f1

# Hauptfunktion
def main():
    # Lade Trainings- und Validierungsdaten
    train_loader, valid_loader = get_train_valid_loader(dataset_train, dataset_val, batch_size, augment=True)
    test_loader = get_test_loader(dataset_test, batch_size)

    # Initialisiere das Modell
    model = initialize_model(num_classes)

    # Initialisiere W&B
    initialize_wandb()

    # Trainiere das Modell
    best_params = {'learning_rate': learning_rate, 'num_epochs': num_epochs}
    model = train_final_model(model, train_loader, valid_loader, best_params)

    # Teste das Modell
    test_acc, precision, recall, f1 = test_model(model, test_loader)

    # Logge die Testergebnisse zu W&B
    wandb.log({
        "test_acc": test_acc, "precision": precision,
        "recall": recall, "f1": f1
    })

    # Rufe noch wandb.finish() auf!!

if __name__ == "__main__":
    main() 