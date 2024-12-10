###############################################################################################
#
# VisionTransformer
# source: https://pytorch.org/vision/main/models/generated/torchvision.models.vit_b_16.html
# source: https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
# source: https://optuna.org/
# source: https://medium.com/@boukamchahamdi/fine-tuning-a-resnet18-model-with-optuna-hyperparameter-optimization-2e3eab0bcca7
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
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torch.optim as optim
from torchvision import models
from sklearn import metrics  # for confusion matrix
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns # Confusion Matrix
import optuna
from torchvision.utils import save_image
#from torchcam.methods import SmoothGradCAMpp
#from torchcam.utils import overlay_mask
import matplotlib.pyplot as plt
from Augmentation_CV import AugmentHandFocus

import os

# Absoluten Pfad des aktuellen Skripts ermitteln
current_dir = os.path.dirname(__file__)
print(current_dir)

# Relativen Pfad zum Zielordner setzen
dataset_train = os.path.join(current_dir, "Sign Language", "train")
dataset_val = os.path.join(current_dir, "Sign Language", "val")
dataset_test = os.path.join(current_dir, "Sign Language", "test")

num_classes = 26

# Device configuration
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)
# Der Ordner, in dem die augmentierten Bilder gespeichert werden sollen
save_augmented_dir = os.path.join(current_dir, "Augmentation Images")

# Erstelle den Ordner, falls er nicht existiert
os.makedirs(save_augmented_dir, exist_ok=True)


def get_train_valid_loader(data_dir_train, 
                           data_dir_valid,
                           batch_size, 
                           augment, 
                           save_augmented_dir,  # Optionaler Parameter für die Speicherung
                           shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # Transformationen für Validierungsdaten
    valid_transform = transforms.Compose([
        transforms.Resize((224, 224)), 
        transforms.ToTensor(),
        normalize,
    ])

    # Augmentierung für Trainingsdaten (falls aktiviert)
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224, padding=4),
            transforms.RandomHorizontalFlip(),        
            #AugmentHandFocus(), 
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            normalize,
        ])

    # Lade den Trainingsdatensatz
    train_dataset = datasets.ImageFolder(root=data_dir_train, transform=train_transform)

    # Lade den Validierungsdatensatz
    valid_dataset = datasets.ImageFolder(root=data_dir_valid, transform=valid_transform)

    # DataLoader für Training und Validierung
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    # Wenn ein Ordner zum Speichern der augmentierten Bilder angegeben wurde, speichern wir die Bilder
    #if save_augmented_dir:
    #    os.makedirs(save_augmented_dir, exist_ok=True)  # Erstelle den Ordner, falls er nicht existiert
    #    
    #    augment = AugmentHandFocus()  # Initialisiere die Augmentierungslogik
    #    for idx, (img, label) in enumerate(train_dataset):
    #        # Berechne den Dateipfad zum Speichern
    #        save_path = os.path.join(save_augmented_dir, f"augmented_{idx}.png")
    #        augment.save_augmented_image(img, save_path)


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

# Optuna objective function for hyperparameter tuning
# https://medium.com/@boukamchahamdi/fine-tuning-a-resnet18-model-with-optuna-hyperparameter-optimization-2e3eab0bcca7
def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 0.0008, 0.0008, log=True)
    batch_size = trial.suggest_categorical('batch_size', [64])
    num_epochs = trial.suggest_int('num_epochs', 10, 10)

    # Load data with current batch size
    train_loader, valid_loader = get_train_valid_loader(
        data_dir_train=dataset_train,
        data_dir_valid=dataset_val,
        batch_size=batch_size,
        augment=True,
        save_augmented_dir=save_augmented_dir,
        shuffle=True
    )
    # DataLoader
    #train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
    #valid_loader = torch.utils.data.DataLoader(dataset_val, batch_size=batch_size, shuffle=False)

    # Load pretrained model
    # https://python.plainenglish.io/how-to-freeze-model-weights-in-pytorch-for-transfer-learning-step-by-step-tutorial-a533a58051ef
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze last layer
    for param in model.heads.head.parameters():
        param.requires_grad = True
        
    # Replace last layer
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)

    # Move the final layer to the device
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.heads.head.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
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
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)

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

        val_loss = val_loss / len(valid_loader.dataset)
        val_acc = val_corrects.double() / len(valid_loader.dataset)
        
        return val_acc

# Run the Optuna study
study = optuna.create_study(direction="maximize") # is it minimize or maximize
study.optimize(objective, n_trials=1)

# Print the best hyperparameters
best_params = study.best_params
print("Beste Hyperparameter:", best_params)
print("Bester Validierungsverlust:", study.best_value)

# Verwende die besten Hyperparameter zum Trainieren des finalen Modells
def train_final_model(best_params, dataset_train, dataset_val, device):
    # Extrahiere die Hyperparameter
    learning_rate = best_params['learning_rate']
    batch_size = best_params['batch_size']
    num_epochs = best_params['num_epochs']
    
    # Lade die Trainings- und Validierungsdaten mit dem besten Batch-Size
    train_loader, valid_loader = get_train_valid_loader(
        data_dir_train=dataset_train,
        data_dir_valid=dataset_val,
        batch_size=batch_size,
        save_augmented_dir=save_augmented_dir,
        augment=True,
        shuffle=True
    )
    
    # Load pretrained model
    # https://python.plainenglish.io/how-to-freeze-model-weights-in-pytorch-for-transfer-learning-step-by-step-tutorial-a533a58051ef
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Unfreeze last layer
    for param in model.heads.head.parameters():
        param.requires_grad = True

    # Replace last layer
    model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)

    # Move the final layer to the device
    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.heads.head.parameters(), lr=learning_rate)

    # Initialize lists to track loss and accuracy for all epochs
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    # Trainiere das Modell mit den besten Hyperparametern
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

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

        # Store the training loss and accuracy for this epoch
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")

    # Evaluieren auf den Validierungsdaten
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

    val_loss = val_loss / len(valid_loader.dataset)
    val_acc = val_corrects.double() / len(valid_loader.dataset)

    # Store the validation loss and accuracy for this epoch
    val_losses.append(val_loss)
    val_accuracies.append(val_acc.item())
    
    print(f"Final Validation Loss: {val_loss:.4f}, Final Validation Accuracy: {val_acc:.4f}")

    # Save the training and validation metrics for all epochs
    checkpoint = {
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies,
        'hyper_params': best_params,
    }
    # Create Evaluation folder
    # Define the directory path
    eval_folder_path = os.path.join(current_dir, "Evaluation_folder")

    # Create the folder if it doesn't exist
    os.makedirs(eval_folder_path, exist_ok=True)    # Save the checkpoint
    torch.save(checkpoint, os.path.join(current_dir, "Evaluation_folder", "ViT_values.pth"))

    return model

# Trainiere das finale Modell mit den besten Hyperparametern
final_model = train_final_model(best_params, dataset_train, dataset_val, device)

# Load test data
test_loader = get_test_loader(
    data_dir= dataset_test, # Pfad zu den Testdaten
    batch_size=best_params['batch_size']
)
#test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=best_params['batch_size'], shuffle=False)

if False:
    # Class name list
    class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    # Testen des Modells auf den Testdaten
    def test_model(model, test_loader):
        model.eval()
        test_corrects = 0
        all_labels_ViT = []
        all_preds_ViT = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                test_corrects += torch.sum(preds == labels.data)


                # Speichern der Labels und Vorhersagen für spätere Auswertungen
                all_labels_ViT.extend(labels.cpu().numpy())
                all_preds_ViT.extend(preds.cpu().numpy())

        # Berechnung der Test Accuracy
        test_acc_ViT = test_corrects.double() / len(test_loader.dataset)
        print(f'Test Accuracy: {test_acc_ViT:.4f}')

        # Berechnung von Precision, Recall und F1-Score
        precision_ViT, recall_ViT, f1_ViT, _ = precision_recall_fscore_support(all_labels_ViT, all_preds_ViT, average='weighted')
        
        print(f'Test Accuracy: {test_acc_ViT:.4f}')
        print(f'Precision: {precision_ViT:.4f}')
        print(f'Recall: {recall_ViT:.4f}')
        print(f'F1-Score: {f1_ViT:.4f}')
        
        print(f'Labels Testdaten: {all_labels_ViT}')
        print(f'vorhergesagte Testdaten: {all_preds_ViT}')
        # Rückgabe der Metriken
        return test_acc_ViT.item(), precision_ViT, recall_ViT, f1_ViT, all_labels_ViT, all_preds_ViT

    # Testen auf Testdaten und Speichern der Metriken und label
    test_acc_ViT, precision_ViT, recall_ViT, f1_ViT, all_labels_ViT, all_preds_ViT = test_model(final_model, test_loader)

    # Testen auf Testdaten
    # test_model(final_model, test_loader)

# Save the entire model
torch.save(final_model, 'ViT_model_1.pth')
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


######### misc function.py
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

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

#########dump images .py
#
# Copyright (c) 2019 Idiap Research Institute, http://www.idiap.ch/
# Written by Suraj Srinivas <suraj.srinivas@idiap.ch>
#

""" Compute saliency maps of images from dataset folder
    and dump them in a results folder """

from torchvision import datasets, utils, models


# PATH variables
PATH = os.path.dirname(os.path.abspath(__file__)) + '/'
dataset = os.path.join(current_dir, "Sign Language", "test")

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


#if __name__ == "__main__":
# Create folder to saliency maps
#save_path = PATH + 'results/'
save_path = os.path.join(current_dir, "Saliency Maps", "results")
create_folder(save_path)
compute_saliency_and_save()
print('Saliency maps saved.')

