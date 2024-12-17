###################################################################################################
#
# Evaluation of the models 
#
###################################################################################################
### Import packages
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import  confusion_matrix
import seaborn as sns
from sklearn.metrics import precision_recall_fscore_support
import os
#from AlexNet import AlexNet


### Import data set
# Absolut path of current script
current_dir = os.path.dirname(__file__)
print(current_dir)
# relative path of test data set
#dataset_test = os.path.join(current_dir, "Sign Language", "test_processed")
dataset_test = os.path.join(current_dir, "Sign Language", "test")

# Load AlexNet class
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
    
### Import the models
resnet50_model = torch.load('resnet50_model_dataset2.pth')
alexnet_model = torch.load('alexnet_model.pth')
ViT_model = torch.load('ViT_model_dataset2.pth')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
###################################################################################################
#
# Test model on resnet50 model
#
###################################################################################################
### Import validation accuracy 
resnet50_params = torch.load(os.path.join(current_dir, "Evaluation_folder", "resnet_values_dataset2.pth"))
# Retrieve saved variables
resnet50_hyper_params = resnet50_params['hyper_params']
resnet50_num_epoch = resnet50_hyper_params['num_epochs']
resnet50_val_loss = resnet50_params['val_losses']
resnet50_val_acc = resnet50_params['val_accuracies']
resnet50_train_loss = resnet50_params['train_losses']
resnet50_train_acc = resnet50_params['train_accuracies']
print(resnet50_train_acc)
print(resnet50_val_acc)
print(resnet50_val_loss)
print(resnet50_train_loss)

epochs = list(range(1, resnet50_num_epoch + 1))
print(epochs)
(len(epochs) == len(resnet50_train_loss) == len(resnet50_val_loss) == len(resnet50_train_acc) == len(resnet50_val_acc))

###################################################################################################
#
# Test model on AlexNet model
#
###################################################################################################
### Import validation accuracy 
alexNet_params = torch.load(os.path.join(current_dir, "Evaluation_folder", "alexNet_values.pth"))
# Retrieve saved variables
alexNet_hyper_params = alexNet_params['hyper_params']
alexNet_num_epoch = alexNet_hyper_params['num_epochs']
alexNet_val_loss = alexNet_params['val_losses']
alexNet_val_acc = alexNet_params['val_accuracies']
alexNet_train_loss = alexNet_params['train_losses']
alexNet_train_acc = alexNet_params['train_accuracies']
print(alexNet_train_acc)
print(alexNet_val_acc)
print(alexNet_val_loss)
print(alexNet_train_loss)

epochs = list(range(1, alexNet_num_epoch + 1))
print(epochs)
(len(epochs) == len(alexNet_train_loss) == len(alexNet_val_loss) == len(alexNet_train_acc) == len(alexNet_val_acc))

###################################################################################################
#
# Test model on Vision Transformer model
#
###################################################################################################
### Import validation accuracy 
ViT_params = torch.load(os.path.join(current_dir, "Evaluation_folder", "ViT_values.pth"))
# Retrieve saved variables
ViT_hyper_params = ViT_params['hyper_params']
ViT_num_epoch = ViT_hyper_params['num_epochs']
ViT_val_loss = ViT_params['val_losses']
ViT_val_acc = ViT_params['val_accuracies']
ViT_train_loss = ViT_params['train_losses']
ViT_train_acc = ViT_params['train_accuracies']
print(ViT_train_acc)
print(ViT_val_acc)
print(ViT_val_loss)
print(ViT_train_loss)

epochs = list(range(1, ViT_num_epoch + 1))
print(epochs)
(len(epochs) == len(ViT_train_loss) == len(ViT_val_loss) == len(ViT_train_acc) == len(ViT_val_acc))

###################################################################################################
#
# Test model on AlexNet
#
###################################################################################################
if True:
        # get test loader
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

    # Load test data
    test_loader = get_test_loader(
        data_dir= dataset_test, # Pfad zu den Testdaten
        batch_size=alexNet_hyper_params['batch_size']
    )

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
    test_acc_alexNet, precision_alexNet, recall_alexNet, f1_alexNet, all_labels_alexNet, all_preds_alexNet = test_model(alexnet_model, test_loader)

###################################################################################################
#
# Test model on ViT model
#
###################################################################################################
if True:
    # get test loader
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

    # Load test data
    test_loader = get_test_loader(
        data_dir= dataset_test, # Pfad zu den Testdaten
        batch_size=ViT_hyper_params['batch_size']
    )


    # Test the model on the test data
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

        # Convert numerical labels and predictions to class names
        #true_labels = [class_names[i] for i in all_labels_resNet50]
        #predicted_labels = [class_names[i] for i in all_preds_resNet50]

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
    test_acc_ViT, precision_ViT, recall_ViT, f1_ViT, all_labels_ViT, all_preds_ViT = test_model(ViT_model, test_loader)

###################################################################################################
#
# Test model on ResNet50
#
###################################################################################################
# get test loader
if True:
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
    # Load test data
    test_loader = get_test_loader(
        data_dir= dataset_test, # Pfad zu den Testdaten
        batch_size=resnet50_hyper_params['batch_size']
    )
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
                # Speichern der Labels und Vorhersagen für spätere Auswertungen
                all_labels_resNet50.extend(labels.cpu().numpy())
                all_preds_resNet50.extend(preds.cpu().numpy())
        # Convert numerical labels and predictions to class names
        #true_labels = [class_names[i] for i in all_labels_resNet50]
        #predicted_labels = [class_names[i] for i in all_preds_resNet50]
        # Berechnung der Test Accuracy
        test_acc_resNet50 = test_corrects.double() / len(test_loader.dataset)
        print(f'Test Accuracy: {test_acc_resNet50:.4f}')
        # Berechnung von Precision, Recall und F1-Score
        precision_resNet50, recall_resNet50, f1_resNet50, _ = precision_recall_fscore_support(all_labels_resNet50, all_preds_resNet50, average='weighted')
        
        print(f'Test Accuracy: {test_acc_resNet50:.4f}')
        print(f'Precision: {precision_resNet50:.4f}')
        print(f'Recall: {recall_resNet50:.4f}')
        print(f'F1-Score: {f1_resNet50:.4f}')
        
        print(f'Labels Testdaten: {all_labels_resNet50}')
        print(f'vorhergesagte Testdaten: {all_preds_resNet50}')
        # Rückgabe der Metriken
        return test_acc_resNet50.item(), precision_resNet50, recall_resNet50, f1_resNet50, all_labels_resNet50, all_preds_resNet50
    # Testen auf Testdaten und Speichern der Metriken und label
    test_acc_resNet50, precision_resNet50, recall_resNet50, f1_resNet50, all_labels_resNet50, all_preds_resNet50 = test_model(resnet50_model, test_loader)



###################################################################################################
###Comparison of the classification methods in the testing stage
###################################################################################################
# set width of bar 
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8)) 

# set height of bar 
AlexNet_bar = [f1_alexNet, test_acc_alexNet, precision_alexNet, recall_alexNet] 
ResNet50_bar = [f1_resNet50, test_acc_resNet50, precision_resNet50, recall_resNet50] 
ViT_bar = [f1_ViT, test_acc_ViT, precision_ViT, recall_ViT] 

# Set position of bar on X axis 
br1 = np.arange(len(AlexNet_bar)) 
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]

# Make the plot
plt.bar(br1, AlexNet_bar, color ='r', width = barWidth, 
        edgecolor ='grey', label ='AlexNet') 
plt.bar(br2, ResNet50_bar, color ='g', width = barWidth, 
        edgecolor ='grey', label ='ResNet50')
plt.bar(br2, ResNet50_bar, color ='b', width = barWidth, 
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
### Evaluation: ResNet50
###################################################################################################
"""
### Confusion Matrix:
conf_matrix_resNet50 = confusion_matrix(all_labels_resNet50, all_preds_resNet50)

# Visualizing Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_resNet50, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.title('Confusion Matrix ResNet50')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# Validation and training Loss/accuracy
epochs = list(range(1, resnet50_num_epoch + 1))
print(epochs)

plt.figure(figsize=(10, 6))

plt.plot(epochs, resnet50_train_loss, label = "train loss") # plotting the Loss curve
plt.plot(epochs, resnet50_val_loss, label = "validation loss")
plt.plot(epochs, resnet50_train_acc, label = "train accuracy") # plotting accuracy
plt.plot(epochs, resnet50_val_acc, label = "validation accuracy")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot with two y-axis one for accuracy, one for loss
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot loss on the left y-axis
ax1.plot(epochs, resnet50_train_loss, label="train loss", color='r')
ax1.plot(epochs, resnet50_val_loss, label="validation loss", color='b')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.tick_params(axis='y')

# Create another y-axis for accuracy
ax2 = ax1.twinx()
ax2.plot(epochs, resnet50_train_acc, label="train accuracy", color='g')
ax2.plot(epochs, resnet50_val_acc, label="validation accuracy", color='y')
ax2.set_ylabel('Accuracy')
ax2.tick_params(axis='y')

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='lower left')

plt.show()
"""
###################################################################################################
### Evaluation: AlexNet
###################################################################################################

### Confusion Matrix:
conf_matrix_alexNet = confusion_matrix(all_labels_alexNet, all_preds_alexNet)

# Visualizing Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_alexNet, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.title('Confusion Matrix AlexNet')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# Validation and training Loss/accuracy
epochs = list(range(1, alexNet_num_epoch + 1))
print(epochs)

plt.figure(figsize=(10, 6))

plt.plot(epochs, alexNet_train_loss, label = "train loss") # plotting the Loss curve
plt.plot(epochs, alexNet_val_loss, label = "validation loss")
plt.plot(epochs, alexNet_train_acc, label = "train accuracy") # plotting accuracy
plt.plot(epochs, alexNet_val_acc, label = "validation accuracy")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot with two y-axis one for accuracy, one for loss
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot loss on the left y-axis
ax1.plot(epochs, alexNet_train_loss, label="train loss", color='r')
ax1.plot(epochs, alexNet_val_loss, label="validation loss", color='b')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.tick_params(axis='y')

# Create another y-axis for accuracy
ax2 = ax1.twinx()
ax2.plot(epochs, alexNet_train_acc, label="train accuracy", color='g')
ax2.plot(epochs, alexNet_val_acc, label="validation accuracy", color='y')
ax2.set_ylabel('Accuracy')
ax2.tick_params(axis='y')

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='lower left')

plt.show()

###################################################################################################
### Evaluation: Vision Transformer
###################################################################################################
"""
### Confusion Matrix:
conf_matrix_ViT = confusion_matrix(all_labels_ViT, all_preds_ViT)

# Visualizing Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_ViT, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.title('Confusion Matrix Vision Transformer')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()

# Validation and training Loss/accuracy
epochs = list(range(1, ViT_num_epoch + 1))
print(epochs)

plt.figure(figsize=(10, 6))

plt.plot(epochs, ViT_train_loss, label = "train loss") # plotting the Loss curve
plt.plot(epochs, ViT_val_loss, label = "validation loss")
plt.plot(epochs, ViT_train_acc, label = "train accuracy") # plotting accuracy
plt.plot(epochs, ViT_val_acc, label = "validation accuracy")
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Plot with two y-axis one for accuracy, one for loss
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot loss on the left y-axis
ax1.plot(epochs, ViT_train_loss, label="train loss", color='r')
ax1.plot(epochs, ViT_val_loss, label="validation loss", color='b')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.tick_params(axis='y')

# Create another y-axis for accuracy
ax2 = ax1.twinx()
ax2.plot(epochs, ViT_train_acc, label="train accuracy", color='g')
ax2.plot(epochs, ViT_val_acc, label="validation accuracy", color='y')
ax2.set_ylabel('Accuracy')
ax2.tick_params(axis='y')

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='lower left')

plt.show()
"""
###################################################################################################
###Confusion Matrix
###################################################################################################
# Confusion Matrix: ResNet50
"""
if True:
    conf_matrix_resNet50 = confusion_matrix(all_labels_resNet50, all_preds_resNet50)

    # Visualisierung der Confusion Matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix_resNet50, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix ResNet50')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

    plt.clf()  # Löscht die Figur für den nächsten Plot
"""
if False:
    # Confusion Matrix: ViT
    conf_matrix_ViT = confusion_matrix(all_labels_ViT, all_preds_ViT)

    # Visualisierung der Confusion Matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix_ViT, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix Vision Transformer')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

    plt.clf()  # Löscht die Figur für den nächsten Plot


# Confusion Matrix: AlexNet
conf_matrix_alexNet = confusion_matrix(all_labels_alexNet, all_preds_alexNet)

# Visualisierung der Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_alexNet, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.title('Confusion Matrix AlexNet')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()


if False:

    # Confusion Matrix: ViT
    conf_matrix_ViT = confusion_matrix(all_labels_ViT, all_preds_ViT)

    # Visualisierung der Confusion Matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix_ViT, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix Vision Transformer')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

    plt.clf()  # Löscht die Figur für den nächsten Plot



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

final_model = alexnet_model

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


#if __name__ == "__main__":
# Create folder to saliency maps
#save_path = PATH + 'results/'
save_path = os.path.join(current_dir, "Saliency Maps", "results_alexnet_model_1")
create_folder(save_path)
compute_saliency_and_save()
print('Saliency maps saved.')

