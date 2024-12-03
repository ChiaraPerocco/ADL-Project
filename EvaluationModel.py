###################################################################################################
#
# Evaluation of the models 
#
###################################################################################################
# Import packages
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader


# Import packages
import torch
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.metrics import  confusion_matrix
import seaborn as sns
#from ResNet50 import test_acc_resNet50, precision_resNet50, recall_resNet50, f1_resNet50, all_labels_resNet50, all_preds_resNet50
#from ResNet50 import device, precision_recall_fscore_support, test_loader
from sklearn.metrics import precision_recall_fscore_support
import os
#from AlexNet import  test_precision_alexNet, test_recall_alexNet, test_f1_alexNet, test_accuracy_alexNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Absolut path of current script
current_dir = os.path.dirname(__file__)
print(current_dir)

# relative path of data sets
dataset_test = os.path.join(current_dir, "Sign Language", "test")

# Load the models
resnet50_model = torch.load('resnet50_model.pth')
#alexnet_model = torch.load('alexnet_model.pth')

###################################################################################################
#
# Test model on resnet50 model
#
###################################################################################################
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
    batch_size=64
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
#
# Test model on AlexNet
#
###################################################################################################
if False: 
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
###Comparison of the classification methods in the testing stage
###################################################################################################
"""
# set width of bar 
barWidth = 0.25
fig = plt.subplots(figsize =(12, 8)) 

# set height of bar 
AlexNet_bar = [f1_alexNet, test_acc_alexNet, precision_alexNet, recall_alexNet] 
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

"""

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

"""
# Confusion Matrix: AlexNet
conf_matrix_alexNet = confusion_matrix(all_labels_resNet50, all_preds_resNet50)

# Visualisierung der Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_alexNet, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.title('Confusion Matrix AlexNet')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()
"""