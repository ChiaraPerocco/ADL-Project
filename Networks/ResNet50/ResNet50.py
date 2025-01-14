###############################################################################################
#
# Resnet-50
# source: https://moiseevigor.github.io/software/2022/12/18/one-pager-training-resnet-on-imagenet/
# source: https://pytorch.org/vision/stable/models.html
# source: https://optuna.org/
# source: https://medium.com/@boukamchahamdi/fine-tuning-a-resnet18-model-with-optuna-hyperparameter-optimization-2e3eab0bcca7
# source: https://towardsdatascience.com/designing-your-neural-networks-a5e4617027ed
# source: https://pytorch.org/docs/stable/optim.html
# source: https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
# source: https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
###############################################################################################
import os
import copy
import torch
import wandb
import torch.optim as optim
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights


# Absoluter Pfad des aktuellen Skripts
current_dir = os.path.dirname(__file__)
print(current_dir)

# Verzeichnisse der Datensätze
dataset_train = os.path.join(current_dir, "Sign Language", "train_processed")
dataset_val = os.path.join(current_dir, "Sign Language", "val_processed")
dataset_test = os.path.join(current_dir, "Sign Language", "test_processed")

# Anzahl der Klassen im Datensatz
num_classes = 26

# Geräte-Konfiguration (CPU oder GPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)


# Datenloader für Trainings- und Validierungsdatensätze
def get_train_valid_loader(data_dir_train, data_dir_valid, batch_size, augment, shuffle=True):
    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    
    # Transformation für den Validierungsdatensatz
    valid_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    
    # Transformation für Trainingsdaten mit Datenaugmentation
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
            normalize
        ])
    else:
        train_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])

    # Lade den Trainings- und Validierungsdatensatz
    train_dataset = datasets.ImageFolder(root=data_dir_train, transform=train_transform)
    valid_dataset = datasets.ImageFolder(root=data_dir_valid, transform=valid_transform)

    # Erstelle DataLoader für Training und Validierung
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader

def get_test_loader(data_dir, batch_size, shuffle=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), normalize])
    test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
    return test_loader


# Erstelle die besten Hyperparameter als Dictionary
best_params = {
    'learning_rate': 0.001,  # Beispielwert (Startwert für den Optimierer)
    'batch_size': 64,      # Beispielwert
    'num_epochs': 50       # Beispielwert
}


# Trainiere das finale Modell mit den besten Hyperparametern
def train_final_model(best_params, dataset_train, dataset_val, device):
    learning_rate = best_params['learning_rate']
    batch_size = best_params['batch_size']
    num_epochs = best_params['num_epochs']
    
    # Lade Trainings- und Validierungsdatensatz
    train_loader, valid_loader = get_train_valid_loader(dataset_train, dataset_val, batch_size, augment=True, shuffle=True)
    
    # Lade das vortrainierte Modell ResNet50
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # Alle Schichten einfrieren, außer der letzten
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    # Ersetze die letzte Schicht (Fully Connected Layer)
    model.fc = nn.Sequential(
        nn.Dropout(p=0.5),  # Dropout hinzugefügt
        nn.Linear(model.fc.in_features, num_classes)
    )
    
    model = model.to(device)
    
    # Initialisiere wandb und überwache
    wandb.init(project='resnet50_model_dataset2_5', config=best_params)
    wandb.watch(model, log="all")  # Dies funktioniert jetzt nach wandb.init()
    
    # Verlustfunktion und Optimierer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.005)

    # Scheduler für OneCycleLR
    scheduler = OneCycleLR(
        optimizer, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=num_epochs,
        pct_start=0.3, anneal_strategy='cos', cycle_momentum=True,
        base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=1000.0
    )

    # Variablen für Metriken und EarlyStopping
    best_loss = float('inf')
    best_model_weights = None
    patience = 5
    patience_counter = patience
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []
    
    # Funktion, um die Lernrate zu bekommen
    def get_lr(optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    # Funktion, um das Momentum zu bekommen
    def get_momentum(optimizer):
        return optimizer.param_groups[0]['momentum']

    # Beste Lernrate und Momentum speichern
    best_lr = None
    best_momentum = None

    # Trainingsloop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        current_lr = get_lr(optimizer)
        current_momentum = get_momentum(optimizer)
        print(f"Epoch {epoch+1}/{num_epochs}, Current Learning Rate: {current_lr:.6f}, Current Momentum: {current_momentum:.6f}")

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
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
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())

        print(f"Training Loss: {epoch_loss:.4f}, Training Accuracy: {epoch_acc:.4f}")

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

        print(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.4f}")

        # Logge Metriken zu wandb
        wandb.log({
            "train acc": epoch_acc,
            "train loss": epoch_loss,
            "val acc": val_acc,
            "val loss": val_loss,
            "lr": current_lr,
            "momentum": current_momentum
        })

        # Speichern der besten Lernrate und des besten Momentums bei der besten Validierungsgenauigkeit
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())
            patience_counter = patience
            best_lr = current_lr
            best_momentum = current_momentum
            print(f"Best model updated with lr={best_lr:.6f}, momentum={best_momentum:.6f} (val_loss < best_loss)")
        else:
            patience_counter -= 1
            print(f"Patience: {patience} (val_loss > best_loss)")
            if patience_counter == 0:
                print("Early Stopping")
                break

    # Modell mit den besten Gewichtungen speichern
    model.load_state_dict(best_model_weights)

    # Speichern des besten Lernraten- und Momentumwertes
    checkpoint = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'best_lr': best_lr,  # Speichern der besten Lernrate
        'best_momentum': best_momentum,  # Speichern des besten Momentums
        'hyper_params': best_params,
    }

    # Speichern des Modells und der Metriken
    eval_folder_path = os.path.join(current_dir, "Evaluation_folder")
    os.makedirs(eval_folder_path, exist_ok=True)
    torch.save(checkpoint, os.path.join(eval_folder_path, "resnet_values_dataset2_5.pth"))

   # # W&B beenden
   # wandb.finish()

    return model

# Trainiere das finale Modell mit den besten Hyperparametern
final_model = train_final_model(best_params, dataset_train, dataset_val, device)


# Load test data
test_loader = get_test_loader(
    data_dir= dataset_test, # Pfad zu den Testdaten
    batch_size=best_params['batch_size']
)

# Class name list
class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Test the model on the test data
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

            all_labels_resNet50.extend(labels.cpu().numpy())
            all_preds_resNet50.extend(preds.cpu().numpy())

    test_acc = test_corrects.double() / len(test_loader.dataset)
    print(f"Test Accuracy: {test_acc:.4f}")

    return test_acc, all_labels_resNet50, all_preds_resNet50

# Testen des finalen Modells
test_acc, all_labels_resNet50, all_preds_resNet50 = test_model(final_model, test_loader)

# Save the entire model
#torch.save(final_model, 'resnet50_model_dataset2_5.pth')
wandb.save('resnet50_model_dataset2_5.pth')  # Speichert das Modell in W&B

wandb.finish()
###################################################################################################
#
# Saliency Maps with Grad-CAM
# https://github.com/idiap/fullgrad-saliency/blob/26fb91b2f9ef616d15d080644b9f8a656a68204f/dump_images.py
#
###################################################################################################
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" 
    Implement GradCAM

    Original Paper: 
    Selvaraju, Ramprasaath R., et al. "Grad-cam: Visual explanations from deep networks 
    via gradient-based localization." ICCV 2017.

"""

import torch.nn.functional as F
from math import isclose
import numpy as np
from sklearn.metrics import precision_recall_fscore_support


class GradCAMExtractor:
    #Extract tensors needed for Gradcam using hooks
    
    def __init__(self, model):
        self.model = model

        self.features = None
        self.feat_grad = None

        prev_module = None
        self.target_module = None

        # Iterate through layers
        for m in self.model.modules():
            if isinstance(m, nn.Conv2d):
                prev_module = m
            elif isinstance(m, nn.Linear):
                self.target_module = prev_module
                break

        if self.target_module is not None:
            # Register feature-gradient and feature hooks for each layer
            handle_g = self.target_module.register_backward_hook(self._extract_layer_grads)
            handle_f = self.target_module.register_forward_hook(self._extract_layer_features)

    def _extract_layer_grads(self, module, in_grad, out_grad):
        # function to collect the gradient outputs
        self.feature_grads = out_grad[0]
    
    def _extract_layer_features(self, module, input, output):
        # function to collect the layer outputs
        self.features = output

    def getFeaturesAndGrads(self, x, target_class):

        out = self.model(x)

        if target_class is None:
            target_class = out.data.max(1, keepdim=True)[1]

        output_scalar = -1. * F.nll_loss(out, target_class.flatten(), reduction='sum')

        # Compute gradients
        self.model.zero_grad()
        output_scalar.backward()

        return self.features, self.feature_grads


class GradCAM():
    """
    Compute GradCAM 
    """

    def __init__(self, model):
        self.model = model
        self.model_ext = GradCAMExtractor(self.model)


    def saliency(self, image, target_class=None):
        #Simple FullGrad saliency
        
        self.model.eval()
        features, intermed_grad = self.model_ext.getFeaturesAndGrads(image, target_class=target_class)

        # GradCAM computation
        grads = intermed_grad.mean(dim=(2,3), keepdim=True)
        cam = (F.relu(features)* grads).sum(1, keepdim=True)
        cam_resized = F.interpolate(F.relu(cam), size=image.size(2), mode='bilinear', align_corners=True)
        return cam_resized


""" Misc helper functions """

import cv2
import subprocess
import torchvision.transforms as transforms

class NormalizeInverse(transforms.Normalize):
    # Undo normalization on images

    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super(NormalizeInverse, self).__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super(NormalizeInverse, self).__call__(tensor.clone())


def create_folder(folder_name):
    try:
        subprocess.call(['mkdir','-p',folder_name])
    except OSError:
        None

def save_saliency_map(image, saliency_map, filename):
    """ 
    Save saliency map on image.
    
    Args:
        image: Tensor of size (3,H,W)
        saliency_map: Tensor of size (1,H,W) 
        filename: string with complete path and file extension

    """

    image = image.data.cpu().numpy()
    saliency_map = saliency_map.data.cpu().numpy()

    saliency_map = saliency_map - saliency_map.min()
    saliency_map = saliency_map / saliency_map.max()
    saliency_map = saliency_map.clip(0,1)

    saliency_map = np.uint8(saliency_map * 255).transpose(1, 2, 0)
    saliency_map = cv2.resize(saliency_map, (224,224))

    image = np.uint8(image * 255).transpose(1,2,0)
    image = cv2.resize(image, (224, 224))

    # Apply JET colormap
    color_heatmap = cv2.applyColorMap(saliency_map, cv2.COLORMAP_JET)
    
    # Combine image with heatmap
    img_with_heatmap = np.float32(color_heatmap) + np.float32(image)
    img_with_heatmap = img_with_heatmap / np.max(img_with_heatmap)

    cv2.imwrite(filename, np.uint8(255 * img_with_heatmap))


#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Compute saliency maps of images from dataset folder
    and dump them in a results folder """

from torchvision import datasets, utils, models


# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
dataset = os.path.join(current_dir, "Sign Language", "test_processed")

unnormalize = NormalizeInverse(mean = [0.485, 0.456, 0.406],
                           std = [0.229, 0.224, 0.225])

def compute_saliency_and_save():
    for batch_idx, (data, _) in enumerate(test_loader):
        data = data.to(device).requires_grad_()

        # Compute saliency maps for the input data
        saliency_map = GradCAM(final_model).saliency(data)

        # Save saliency maps
        for i in range(data.size(0)):
            filename = save_path + str( (batch_idx+1) * (i+1))
            image = unnormalize(data[i].cpu())
            save_saliency_map(image, saliency_map[i], filename + '_' + '.jpg')



#save_path = os.path.join(current_dir, "Saliency Map", "results")
save_path = os.path.join(current_dir, "Saliency Maps_resnet_dataset2_5", "results_resnet_dataset2_5")
os.makedirs(save_path, exist_ok=True)
create_folder(save_path)
compute_saliency_and_save()
print('Saliency maps saved.')


"""
Ausführen des Modells auf den Testdaten
# Modell und Metriken laden
checkpoint = torch.load('path_to_saved_model/resnet_values.pth')

# Lade das Modell mit den besten Gewichtungen
model.load_state_dict(checkpoint['best_model_weights'])

# Lade die besten Hyperparameter und setze sie wieder
best_lr = checkpoint['best_lr']
best_momentum = checkpoint['best_momentum']

# Setze den Optimierer und Scheduler mit den gespeicherten Werten
optimizer = optim.SGD(model.fc.parameters(), lr=best_lr, momentum=best_momentum, weight_decay=0.005)
scheduler = OneCycleLR(
    optimizer, max_lr=best_lr, steps_per_epoch=len(train_loader), epochs=best_params['num_epochs'],
    pct_start=0.3, anneal_strategy='cos', cycle_momentum=True,
    base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0
)

# Teste das Modell auf neuen Testdaten
test_acc, all_labels_resNet50, all_preds_resNet50 = test_model(model, test_loader)
"""