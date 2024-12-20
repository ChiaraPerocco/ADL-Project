###############################################################################################
#
# AlexNet
# source: https://www.digitalocean.com/community/tutorials/alexnet-pytorch
# source: https://optuna.org/
# source: https://medium.com/@boukamchahamdi/fine-tuning-a-resnet18-model-with-optuna-hyperparameter-optimization-2e3eab0bcca7
#
###############################################################################################
# import packages
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
from torch.optim.lr_scheduler import OneCycleLR
import optuna
import os
import wandb
import copy

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
            transforms.RandomRotation(15),                # Random rotation between -15 to 15 degrees
            transforms.ColorJitter(brightness=0.2,        # Random color variations
                                contrast=0.2, 
                                saturation=0.2, 
                                hue=0.1),
            transforms.RandomAffine(degrees=0,            # Random affine transformations (scale, shear)
                                    translate=(0.1, 0.1), 
                                    scale=(0.9, 1.1), 
                                    shear=10),
            transforms.RandomGrayscale(p=0.1),            # Randomly convert images to grayscale
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

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, valid_loader


def get_test_loader(data_dir, batch_size, shuffle=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return test_loader


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
    
# Erstelle die besten Hyperparameter als Dictionary
best_params = {
    'learning_rate': 0.01,  # Beispielwert
    'batch_size': 64,      # Beispielwert
    'num_epochs': 50       # Beispielwert
}


if False:
    batch_size = 32
    learning_rate = 10**-5
    num_epochs = 50

    best_params = {
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'num_epochs': num_epochs
    }

    # Load data with current batch size
    train_loader, valid_loader = get_train_valid_loader(
        data_dir_train=dataset_train,
        data_dir_valid=dataset_val,
        batch_size=batch_size,
        augment=True,
        shuffle=True
    )

    # Load model
    model = AlexNet(num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

    # Train the model
    total_step = len(train_loader)

    total_step = len(train_loader)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
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
                


if False:
    # Optuna objective function for hyperparameter tuning
    # https://medium.com/@boukamchahamdi/fine-tuning-a-resnet18-model-with-optuna-hyperparameter-optimization-2e3eab0bcca7
    def objective(trial):
        # Suggest hyperparameters
        learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [64])
        num_epochs = trial.suggest_int('num_epochs', 1, 50)

        # Load data with current batch size
        train_loader, valid_loader = get_train_valid_loader(
            data_dir_train=dataset_train,
            data_dir_valid=dataset_val,
            batch_size=batch_size,
            augment=True,
            shuffle=True
        )

        # Load model
        model = AlexNet(num_classes).to(device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

        # Train the model
        total_step = len(train_loader)

        total_step = len(train_loader)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
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

# Use the best hyperparameters to train the final model
def train_final_model(best_params, dataset_train, dataset_val, device):
    # Extract the hyperparameters
    learning_rate = best_params['learning_rate']
    batch_size = best_params['batch_size']
    num_epochs = best_params['num_epochs']
    
    # Load the training and validation data with the best batch size
    train_loader, valid_loader = get_train_valid_loader(
        data_dir_train=dataset_train,
        data_dir_valid=dataset_val,
        batch_size=batch_size,
        augment=True,
        shuffle=True
    )
    
    # Load model
    model = AlexNet(num_classes).to(device)

    # Initialisiere wandb und überwache
    wandb.init(project='alexNet_model_dataset2_4', config=best_params)
    # Überwacht das Modell und protokolliert Gradienten und Gewichte
    wandb.watch(model, log="all") 

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

    # Scheduler für OneCycleLR
    scheduler = OneCycleLR(
        optimizer, max_lr=0.1, steps_per_epoch=len(train_loader), epochs=num_epochs,
        pct_start=0.3, anneal_strategy='cos', cycle_momentum=True,
        base_momentum=0.85, max_momentum=0.95, div_factor=25.0, final_div_factor=10000.0
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

    # Train the model
    total_step = len(train_loader)

    total_step = len(train_loader)

    # Loss and optimizer
    #criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)

    #Initialize Variables for EarlyStopping
    best_loss = float('inf')
    best_model_weights = None
    patience = 5
    
    # Train the model with the best hyperparameters
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_corrects = 0

        current_lr = get_lr(optimizer)
        current_momentum = get_momentum(optimizer)
        print(f"Epoch {epoch+1}/{num_epochs}, Current Learning Rate: {current_lr:.6f}, Current Momentum: {current_momentum:.6f}")

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

        # Store the training loss and accuracy for this epoch
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc.item())

        print(f"Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}")
        

        # Evaluate on the validation data
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
    
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {val_loss:.4f}, Accuracy: {val_acc:.4f}")

        # Logge Metriken zu wandb
        wandb.log({
            "train acc": epoch_acc,
            "train loss": epoch_loss,
            "val acc": val_acc,
            "val loss": val_loss,
            "lr": current_lr,
            "momentum": current_momentum
        })

        # Prüfe auf Early Stopping
        # Early stopping
        if val_loss < best_loss:
            best_loss = val_loss
            best_model_weights = copy.deepcopy(model.state_dict())  # Deep copy here      
            patience_counter = patience
            best_lr = current_lr
            best_momentum = current_momentum
            print(f"Best model updated with lr={best_lr:.6f}, momentum={best_momentum:.6f} (val_loss < best_loss)")
        else:
            patience_counter -= 1
            print(f"Patience val_loss > best_loss {patience}")
            if patience_counter == 0:
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
    torch.save(checkpoint, os.path.join(eval_folder_path, "alexNet_values_dataset2_4.pth"))

    # W&B beenden
    wandb.finish()

    return model

# Trainiere das finale Modell mit den besten Hyperparametern
final_model = train_final_model(best_params, dataset_train, dataset_val, device)

# Load test data
test_loader = get_test_loader(
    data_dir= dataset_test, # Pfad zu den Testdaten
    batch_size=best_params['batch_size']
)

if False:
    # Class name list
    #class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

    # Test the model on the test data
    def test_model(model, test_loader):
        model.eval()
        test_corrects = 0
        all_labels_alexNet = []
        all_preds_alexNet = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)

                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                test_corrects += torch.sum(preds == labels.data)


                # Speichern der Labels und Vorhersagen für spätere Auswertungen
                all_labels_alexNet.extend(labels.cpu().numpy())
                all_preds_alexNet.extend(preds.cpu().numpy())

        # Convert numerical labels and predictions to class names
        #true_labels = [class_names[i] for i in all_labels_alexNet]
        #predicted_labels = [class_names[i] for i in all_preds_alexNet]

        # Berechnung der Test Accuracy
        test_acc_alexNet = test_corrects.double() / len(test_loader.dataset)
        print(f'Test Accuracy: {test_acc_alexNet:.4f}')

        # Berechnung von Precision, Recall und F1-Score
        precision_alexNet, recall_alexNet, f1_alexNet, _ = precision_recall_fscore_support(all_labels_alexNet, all_preds_alexNet, average='weighted')
        
        print(f'Test Accuracy: {test_acc_alexNet:.4f}')
        print(f'Precision: {precision_alexNet:.4f}')
        print(f'Recall: {recall_alexNet:.4f}')
        print(f'F1-Score: {f1_alexNet:.4f}')
        
        print(f'Labels Testdaten: {all_labels_alexNet}')
        print(f'vorhergesagte Testdaten: {all_preds_alexNet}')
        # Rückgabe der Metriken
        return test_acc_alexNet.item(), precision_alexNet, recall_alexNet, f1_alexNet, all_labels_alexNet, all_preds_alexNet

    # Testen auf Testdaten und Speichern der Metriken und label
    test_acc_alexNet, precision_alexNet, recall_alexNet, f1_alexNet, all_labels_alexNet, all_preds_alexNet = test_model(final_model, test_loader)


# Save the entire model
torch.save(final_model, 'alexNet_model_dataset2_4.pth')

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
save_path = os.path.join(current_dir, "Saliency Maps_alexNet_dataset2_4")
os.makedirs(save_path, exist_ok=True)
create_folder(save_path)
compute_saliency_and_save()
print('Saliency maps saved.')



