###################################################################################################################################### 
#
# Vision Transformer
#
# source: https://medium.com/@brianpulfer/vision-transformers-from-scratch-pytorch-a-step-by-step-guide-96c3313c2e0c
# source: https://huggingface.co/google/vit-base-patch16-224
# source: https://huggingface.co/blog/fine-tune-vit
# source: https://colab.research.google.com/drive/1BG_8peLIzpbQxztz2GNSptok0g4QrOiN?usp=sharing#scrollTo=XLKA1dnC4O1d
# source: https://optuna.org/
# source: https://medium.com/@boukamchahamdi/fine-tuning-a-resnet18-model-with-optuna-hyperparameter-optimization-2e3eab0bcca7
#
#####################################################################################################################################
# import packages
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import torch.optim as optim
from torchvision import models
from sklearn import metrics  # for confusion matrix
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
import seaborn as sns # Confusion Matrix
import optuna
import os

# Absoluten Pfad des aktuellen Skripts ermitteln
current_dir = os.path.dirname(__file__)
print(current_dir)

# Relativen Pfad zum Zielordner setzen
dataset_train = os.path.join(current_dir, "facial_emotion_dataset", "dataset_output - Kopie", "train")
dataset_val = os.path.join(current_dir, "facial_emotion_dataset", "dataset_output - Kopie", "val")
dataset_test = os.path.join(current_dir, "facial_emotion_dataset", "dataset_output - Kopie", "test")

# Import data set
#dataset_train = r"C:\Studium\Data Analytics, M.Sc\Advanced Deep Learning\Team Project Empty\facial_emotion_dataset\dataset_output - Kopie\train"  # Pfad zu den Trainingsdaten
#dataset_val = r"C:\Studium\Data Analytics, M.Sc\Advanced Deep Learning\Team Project Empty\facial_emotion_dataset\dataset_output - Kopie\val"  # Pfad zu den Validierungsdaten
#dataset_test = r"C:\Studium\Data Analytics, M.Sc\Advanced Deep Learning\Team Project Empty\facial_emotion_dataset\dataset_output - Kopie\test"  # Pfad zu den Testdaten

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set number of classes in data
num_classes = 5

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

# Modell und Optimierer initialisieren
#model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
#model = model.to(device)
#model.classifier = nn.Linear(model.classifier.in_features, num_classes)

# Optuna-Zielfunktion für Hyperparameter-Optimierung
def objective(trial):
    # Hyperparameter anpassen
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    num_epochs = trial.suggest_int("num_epochs", 1, 5)

    # Datensätze mit den aktuellen Batch-Größen laden
    train_loader, valid_loader = get_train_valid_loader(
        data_dir_train=dataset_train,
        data_dir_valid=dataset_val,
        batch_size=batch_size,
        augment=True,
        shuffle=True
    )

    # Modell und Optimierer initialisieren
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    # Replace last layer    
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    # Unfreeze last layer
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    model = model.to(device)
    
    # Loss and optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    model.train()
    # Training und Validierung
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

    # Validierung
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs.logits, 1)
            val_loss += criterion(outputs.logits, labels).item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
                
        val_loss = val_loss / len(valid_loader.dataset)
        val_acc = val_corrects.double() / len(valid_loader.dataset)
        
        return val_acc

# Starte die Optimierung
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=1)

# Beste Hyperparameter und Wert
best_params = study.best_params
print("Best hyperparameters:", best_params)
print("Best validation accuracy:", study.best_value)

if False:
    # Testen des Modells mit den besten Hyperparametern
    best_batch_size = study.best_params['batch_size']
    best_learning_rate = study.best_params['learning_rate']
    best_num_epochs = study.best_params['num_epochs']

if False:
    # Erstelle Modell mit besten Hyperparametern
    train_loader, valid_loader = get_train_valid_loader(
        data_dir_train=dataset_train,
        data_dir_valid=dataset_val,
        batch_size=best_batch_size,
        augment=True,
        shuffle=True
    )
    test_loader = get_test_loader(data_dir=dataset_test, batch_size=best_batch_size)

    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    model = model.to(device)
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

    optimizer = optim.AdamW(model.parameters(), lr=best_learning_rate)
    criterion = nn.CrossEntropyLoss()

    # Trainingsfunktion
    def train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs=best_num_epochs):
        for epoch in range(num_epochs):
            print(f'Epoch {epoch+1}/{num_epochs}')

            model.train()
            running_loss = 0.0
            running_corrects = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs.logits, 1)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = running_corrects.double() / len(train_loader.dataset)

            print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        return model

    model = train_model(model, criterion, optimizer, train_loader, valid_loader, num_epochs=best_num_epochs)

    # Testfunktion
    def test_model(model, test_loader):
        model.eval()
        test_corrects = 0
        all_labels = []
        all_preds = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, preds = torch.max(outputs.logits, 1)
                test_corrects += torch.sum(preds == labels.data)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())

        test_acc = test_corrects.double() / len(test_loader.dataset)
        precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
        print(f'Test Accuracy: {test_acc:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1-Score: {f1:.4f}')
        
        return test_acc.item(), precision, recall, f1, all_labels, all_preds

    # Modell auf Testdaten auswerten
    test_acc, precision, recall, f1, all_labels, all_preds = test_model(model, test_loader)

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
        augment=True,
        shuffle=True
    )
    
    # Modell und Optimierer initialisieren
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False

    model.classifier = nn.Linear(model.classifier.in_features, num_classes)

    # Unfreeze last layer
    for param in model.classifier.parameters():
        param.requires_grad = True
        
    model = model.to(device)
    
    # Loss and optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
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
            _, preds = torch.max(outputs.logits, 1)
            loss = criterion(outputs.logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)

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
            _, preds = torch.max(outputs.logits, 1)
            loss = criterion(outputs.logits, labels)

            val_loss += loss.item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)

    val_loss = val_loss / len(valid_loader.dataset)
    val_acc = val_corrects.double() / len(valid_loader.dataset)
    
    print(f"Final Validation Loss: {val_loss:.4f}, Final Validation Accuracy: {val_acc:.4f}")

    return model

# Trainiere das finale Modell mit den besten Hyperparametern
final_model = train_final_model(best_params, dataset_train, dataset_val, device)

# Load test data
test_loader = get_test_loader(
    data_dir= dataset_test, # Pfad zu den Testdaten
    batch_size=best_params['batch_size']
)


# Class name list
class_names = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]
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
            _, preds = torch.max(outputs.logits, 1)
            test_corrects += torch.sum(preds == labels.data)


            # Speichern der Labels und Vorhersagen für spätere Auswertungen
            all_labels_ViT.extend(labels.cpu().numpy())
            all_preds_ViT.extend(preds.cpu().numpy())

    # Convert numerical labels and predictions to class names
    #true_labels = [class_names[i] for i in all_labels_ViT]
    #predicted_labels = [class_names[i] for i in all_preds_ViT]

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
            target_class = out.logits.detach().max(1, keepdim=True)[1]

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
dataset = os.path.join(current_dir, "facial_emotion_dataset", "dataset_output - Kopie", "test")

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
save_path = r"C:\Studium\Data Analytics, M.Sc\Advanced Deep Learning\Team Project Empty\Output_images\results"
create_folder(save_path)
compute_saliency_and_save()
print('Saliency maps saved.')

######################################################################################################
if False: 
    x = 1
    # get the input image index
    #from tf_keras_vis.utils.score import CategoricalScore
    #score = CategoricalScore([281, 235, 8, 292]) # value for each class???? 


    # Create GradCAM object 
    #gradcam = Gradcam(final_model, clone = True)

    # Generate heatmap with Grad CAM
    #cam = gradcam(score, input_images, penultimate_layer = -1)

    # show generated images


    #from tf_keras_vis.saliency import Saliency
    #from tf_keras.vis.utils import normalize

    # Create saliency object 
    #saliency = Saliency(final_model, clone = False)

    # Generate saliency map
    #saliency_map = saliency(score, input_images)
    #saliency_map = normalize(saliency_map)



'''
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
    
# Modell initialisieren
model = AlexNet(num_classes=6).to(device)

# Optuna-Zielfunktion für Hyperparameter-Optimierung
def objective(trial):
    # Hyperparameter anpassen
    batch_size = trial.suggest_int("batch_size", 32, 128, step=32)
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    num_epochs = trial.suggest_int("num_epochs", 1, 5)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-2, log = True)
    momentum = trial.suggest_float('momentum', 0.7, 0.9)

    # Datensätze mit den aktuellen Batch-Größen laden
    train_loader, valid_loader = get_train_valid_loader(
        data_dir_train=dataset_train,
        data_dir_valid=dataset_val,
        batch_size=batch_size,
        augment=True,
        shuffle=True
    )

    # Modell initialisieren
    model = AlexNet(num_classes=6).to(device)

    # Loss und Optimierer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)

    model.train()
    # Training und Validierung
    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

    # Validierung
    model.eval()
    val_loss = 0.0
    val_corrects = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_loss += criterion(outputs, labels).item() * inputs.size(0)
            val_corrects += torch.sum(preds == labels.data)
                
        val_loss = val_loss / len(valid_loader.dataset)
        val_acc = val_corrects.double() / len(valid_loader.dataset)
        
        return val_acc

# Starte die Optimierung
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=2)

# Beste Hyperparameter und Wert
best_params = study.best_params
print("Best hyperparameters:", best_params)
print("Best validation accuracy:", study.best_value)

# Verwende die besten Hyperparameter zum Trainieren des finalen Modells
def train_final_model(best_params, dataset_train, dataset_val, device):
    # Extrahiere die Hyperparameter
    learning_rate = best_params['learning_rate']
    batch_size = best_params['batch_size']
    num_epochs = best_params['num_epochs']
    weight_decay = best_params['weight_decay'] 
    momentum = best_params['momentum']
    
    # Lade die Trainings- und Validierungsdaten mit dem besten Batch-Size
    train_loader, valid_loader = get_train_valid_loader(
        data_dir_train=dataset_train,
        data_dir_valid=dataset_val,
        batch_size=batch_size,
        augment=True,
        shuffle=True
    )
    
     # Modell initialisieren
    model = AlexNet(num_classes=6).to(device)

    # Loss und Optimierer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=momentum)
    
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
    
    print(f"Final Validation Loss: {val_loss:.4f}, Final Validation Accuracy: {val_acc:.4f}")

    return model

# Trainiere das finale Modell mit den besten Hyperparametern
final_model = train_final_model(best_params, dataset_train, dataset_val, device)

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
AlexNet_bar = [val_loss, ] 
ViT_bar = [f1, test_acc, precision, recall] 

# Set position of bar on X axis 
br1 = np.arange(len(AlexNet_bar)) 
br2 = [x + barWidth for x in br1] 

# Make the plot
plt.bar(br1, AlexNet_bar, color ='r', width = barWidth, 
        edgecolor ='grey', label ='AlexNet') 
plt.bar(br2, ViT_bar, color ='g', width = barWidth, 
        edgecolor ='grey', label ='ViT') 

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
# Confusion Matrix: Vision Transformer
conf_matrix = confusion_matrix(all_labels, all_preds)

# Visualisierung der Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.title('Confusion Matrix ViT')
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
'''