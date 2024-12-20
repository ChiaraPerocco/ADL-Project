import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import os
import wandb
import copy
import torch.multiprocessing as mp

# Absolut path of current script
current_dir = os.path.dirname(__file__)
print(current_dir)

# relative path of data sets
dataset_train = os.path.join(current_dir, "Sign Language", "train_processed")
dataset_val = os.path.join(current_dir, "Sign Language", "val_processed")
dataset_test = os.path.join(current_dir, "Sign Language", "test_processed")

# Device-Konfiguration
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# Number of classes
num_classes = 26

# Funktion für das Laden von Daten mit DataLoader
def get_train_valid_loader(data_dir_train, data_dir_valid, batch_size, augment, shuffle=True):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])

    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

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
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = datasets.ImageFolder(root=data_dir_train, transform=train_transform)
    valid_dataset = datasets.ImageFolder(root=data_dir_valid, transform=valid_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, valid_loader

# Funktion für das Laden von Testdaten
def get_test_loader(data_dir, batch_size, shuffle=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return test_loader

# Definition des AlexNet Modells
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2))
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
            nn.MaxPool2d(kernel_size=3, stride=2))
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
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

# Erstelle die besten Hyperparameter als Dictionary
best_params = {
    'learning_rate': 0.01,  # Beispielwert
    'batch_size': 64,      # Beispielwert
    'num_epochs': 50       # Beispielwert
}

# Trainiere das Modell
def train_model(rank, world_size, best_params, dataset_train, dataset_val, device):
    # Initialisieren des Moduls für DistributedDataParallel
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)

    # Hyperparameter ausbestimmen
    batch_size = best_params['batch_size']
    learning_rate = best_params['learning_rate']
    num_epochs = best_params['num_epochs']

    # Lade Trainings- und Validierungsdaten
    train_loader, valid_loader = get_train_valid_loader(
        data_dir_train=dataset_train,
        data_dir_valid=dataset_val,
        batch_size=batch_size,
        augment=True,
        shuffle=True
    )

    # Initialisiere das Modell und sende es an das richtige Gerät
    model = AlexNet(num_classes).to(device)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    # Definiere Verlustfunktion und Optimierer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialisiere wandb
    wandb.init(project="alexNet_model", config=best_params)
    wandb.watch(model, log="all")

    # Scheduler für OneCycleLR
    scheduler = OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=num_epochs)

    # Training
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # Logging der Metriken
        wandb.log({
            "train_loss": epoch_loss,
            "train_acc": epoch_acc
        })

        # Evaluation auf dem Validierungsdatensatz
        model.eval()
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

        val_loss /= len(valid_loader.dataset)
        val_acc = val_corrects.double() / len(valid_loader.dataset)

        wandb.log({
            "val_loss": val_loss,
            "val_acc": val_acc
        })

    # Schließen von wandb
    wandb.finish()

# Funktion für paralleles Training
def main():
    world_size = 4  # Die Anzahl der Prozesse (Anzahl der GPUs oder CPUs)

    # Wenn Sie auf mehreren GPUs trainieren, setzen Sie den Rank und World Size
    mp.spawn(train_model, args=(world_size, best_params, dataset_train, dataset_val, device), nprocs=world_size)

# Starte das Training
if __name__ == "__main__":
    main()
