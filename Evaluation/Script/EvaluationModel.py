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
dataset_test = os.path.join(current_dir, "Sign Language", "test_processed")
#dataset_test = os.path.join(current_dir, "Sign Language", "test")

if False:
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

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

### Import the models
#resnet50_model = torch.load('resnet50_model_dataset2_2.pth', map_location=torch.device('cpu'))
#alexnet_model = torch.load('alexnet_model_dataset2_2.pth', map_location=torch.device('cpu'))
#ViT_model = torch.load('ViT_model_dataset2_2.pth', map_location=torch.device('cpu'))
ViT_model = torch.load('ViT_model_dataset2_7.pth')
resnet50_model = torch.load('resnet50_model_dataset2_2.pth')


###################################################################################################
#
# Test model on resnet50 model
#
###################################################################################################
if True:
    ### Import validation accuracy 
    resnet50_params = torch.load(os.path.join(current_dir, "Evaluation_folder", "resnet_values_dataset2_2.pth"))
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
if False:
    ### Import validation accuracy 
    alexNet_params = torch.load(os.path.join(current_dir, "Evaluation_folder", "alexNet_values_dataset2_2.pth"))
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
if True:
    ### Import validation accuracy 
    ViT_params = torch.load(os.path.join(current_dir, "Evaluation_folder", "ViT_values_dataset2_2.pth"))
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
if False:
        # get test loader
    def get_test_loader(data_dir,
                        batch_size,
                        shuffle=True):
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        )

        transform = transforms.Compose([
            transforms.Resize((224, 224)),   # adjust image size
            transforms.ToTensor(),
            normalize,
        ])
        

        # load test dataset
        test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

        # create dataloader for testlaoder
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

        return test_loader

    # Load test data
    test_loader = get_test_loader(
        data_dir= dataset_test, # path for testdata
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

                all_labels_alexNet.extend(labels.cpu().numpy())
                all_preds_alexNet.extend(preds.cpu().numpy())

                # Convert numerical labels and predictions to class names
                #true_labels = [class_names[i] for i in all_labels_alexNet]
                #predicted_labels = [class_names[i] for i in all_preds_alexNet]

        #calculate test Accuracy
        test_acc_alexNet = test_corrects.double() / len(test_loader.dataset)
        print(f'Test Accuracy: {test_acc_alexNet:.4f}')

        # calculate Precision, Recall und F1-Score
        precision_alexNet, recall_alexNet, f1_alexNet, _ = precision_recall_fscore_support(all_labels_alexNet, all_preds_alexNet, average='weighted')
                
        print(f'Test Accuracy: {test_acc_alexNet:.4f}')
        print(f'Precision: {precision_alexNet:.4f}')
        print(f'Recall: {recall_alexNet:.4f}')
        print(f'F1-Score: {f1_alexNet:.4f}')
                
        print(f'Labels Testdaten: {all_labels_alexNet}')
        print(f'vorhergesagte Testdaten: {all_preds_alexNet}')
        # return metrics
        return test_acc_alexNet.item(), precision_alexNet, recall_alexNet, f1_alexNet, all_labels_alexNet, all_preds_alexNet

    # test and save metrics and labels
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
            transforms.Resize((224, 224)),   # adjust image size
            transforms.ToTensor(),
            normalize,
        ])
        

        # load test dataset
        test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

        # create dataloader for testdata
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

        return test_loader

    # Load test data
    test_loader = get_test_loader(
        data_dir= dataset_test, # path for testdata
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

                all_labels_ViT.extend(labels.cpu().numpy())
                all_preds_ViT.extend(preds.cpu().numpy())

        # Convert numerical labels and predictions to class names
        #true_labels = [class_names[i] for i in all_labels_resNet50]
        #predicted_labels = [class_names[i] for i in all_preds_resNet50]

        # calculate test Accuracy
        test_acc_ViT = test_corrects.double() / len(test_loader.dataset)
        print(f'Test Accuracy: {test_acc_ViT:.4f}')

        # calculate Precision, Recall und F1-Score
        precision_ViT, recall_ViT, f1_ViT, _ = precision_recall_fscore_support(all_labels_ViT, all_preds_ViT, average='weighted')
        
        print(f'Test Accuracy: {test_acc_ViT:.4f}')
        print(f'Precision: {precision_ViT:.4f}')
        print(f'Recall: {recall_ViT:.4f}')
        print(f'F1-Score: {f1_ViT:.4f}')
        
        print(f'Labels Testdaten: {all_labels_ViT}')
        print(f'vorhergesagte Testdaten: {all_preds_ViT}')
        # return metrics
        return test_acc_ViT.item(), precision_ViT, recall_ViT, f1_ViT, all_labels_ViT, all_preds_ViT

    # test and save metrics and labels
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
            transforms.Resize((224, 224)),   # adjust image size
            normalize,
        ])
        
        # load test dataset
        test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
        # create dataloader for testdata
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        return test_loader
    # Load test data
    test_loader = get_test_loader(
        data_dir= dataset_test, #path for testdata
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
                all_labels_resNet50.extend(labels.cpu().numpy())
                all_preds_resNet50.extend(preds.cpu().numpy())
        # Convert numerical labels and predictions to class names
        #true_labels = [class_names[i] for i in all_labels_resNet50]
        #predicted_labels = [class_names[i] for i in all_preds_resNet50]
        # calculate test Accuracy
        test_acc_resNet50 = test_corrects.double() / len(test_loader.dataset)
        print(f'Test Accuracy: {test_acc_resNet50:.4f}')
        # calculate Precision, Recall und F1-Score
        precision_resNet50, recall_resNet50, f1_resNet50, _ = precision_recall_fscore_support(all_labels_resNet50, all_preds_resNet50, average='weighted')
        
        print(f'Test Accuracy: {test_acc_resNet50:.4f}')
        print(f'Precision: {precision_resNet50:.4f}')
        print(f'Recall: {recall_resNet50:.4f}')
        print(f'F1-Score: {f1_resNet50:.4f}')
        
        print(f'Labels Testdaten: {all_labels_resNet50}')
        print(f'vorhergesagte Testdaten: {all_preds_resNet50}')
        # return metrics
        return test_acc_resNet50.item(), precision_resNet50, recall_resNet50, f1_resNet50, all_labels_resNet50, all_preds_resNet50
    # test and save metrics and labels
    test_acc_resNet50, precision_resNet50, recall_resNet50, f1_resNet50, all_labels_resNet50, all_preds_resNet50 = test_model(resnet50_model, test_loader)



###################################################################################################
###Comparison of the classification methods in the testing stage
###################################################################################################
if False:
    # set width of bar 
    barWidth = 0.25
    fig = plt.subplots(figsize =(12, 8)) 

    # set height of bar 
    AlexNet_bar = [f1_alexNet, test_acc_alexNet, precision_alexNet] 
    ResNet50_bar = [f1_resNet50, test_acc_resNet50, precision_resNet50] 
    ViT_bar = [f1_ViT, test_acc_ViT, precision_ViT] 
    print(AlexNet_bar)
    print(ResNet50_bar)
    print(ViT_bar)

    # Set position of bar on X axis 
    br1 = np.arange(len(AlexNet_bar)) 
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]

    # Make the plot
    plt.bar(br1, AlexNet_bar, color ='r', width = barWidth, 
            edgecolor ='grey', label ='AlexNet') 
    plt.bar(br2, ResNet50_bar, color ='g', width = barWidth, 
            edgecolor ='grey', label ='ResNet50')
    plt.bar(br3, ViT_bar, color ='b', width = barWidth, 
            edgecolor ='grey', label ='ViT')

    # Adding values on top of the bars
    for i in range(len(AlexNet_bar)):
        plt.text(br1[i], AlexNet_bar[i], f'{round(AlexNet_bar[i], 4)}', ha='center', fontsize=18)
        plt.text(br2[i], ResNet50_bar[i], f'{round(ResNet50_bar[i], 4)}', ha='center', fontsize=18)
        plt.text(br3[i], ViT_bar[i], f'{round(ViT_bar[i], 4)}', ha='center', fontsize=18)


    # Adding Xticks 
    plt.xlabel('Comparison of the classification methods in the testing stage', fontweight ='bold', fontsize = 24) 
    plt.ylabel('values', fontweight ='bold', fontsize = 24) 
    plt.xticks([r + barWidth for r in range(len(AlexNet_bar))], 
            ['F-Measure', 'Accuracy', 'Precision'], fontsize = 24)

    plt.yticks(fontsize=24)  

    # Set larger font for legend and labels
    plt.legend(fontsize=24, bbox_to_anchor=(0.98, 1), loc='upper left')
    plt.show() 
    plt.clf()  


###################################################################################################
### Evaluation: ResNet50
###################################################################################################
if True:
    ### Confusion Matrix
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
    plt.title("ResNet50")
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

###################################################################################################
### Evaluation: AlexNet
###################################################################################################
if False:
    ### Confusion Matrix
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
    plt.title("AlexNet")
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
if True:
    ### Confusion Matrix
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
    plt.title("Vision Transformer")
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

###################################################################################################
###Confusion Matrix
###################################################################################################
# Confusion Matrix: ResNet50
if True:
    conf_matrix_resNet50 = confusion_matrix(all_labels_resNet50, all_preds_resNet50)

    # Visulizing Confusion Matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix_resNet50, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix ResNet50')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

    plt.clf()  

if True:
    # Confusion Matrix: ViT
    conf_matrix_ViT = confusion_matrix(all_labels_ViT, all_preds_ViT)

    # Visulizing Confusion Matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix_ViT, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix Vision Transformer')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

    plt.clf()  

if False:
    # Confusion Matrix: AlexNet
    conf_matrix_alexNet = confusion_matrix(all_labels_alexNet, all_preds_alexNet)

    # Visualizing Confusion Matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix_alexNet, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix AlexNet')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()


if False:

    # Confusion Matrix: ViT
    conf_matrix_ViT = confusion_matrix(all_labels_ViT, all_preds_ViT)

    # Visualizing Confusion Matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(conf_matrix_ViT, annot=True, fmt='g', cmap='Blues', cbar=False)
    plt.title('Confusion Matrix Vision Transformer')
    plt.xlabel('Predicted labels')
    plt.ylabel('True labels')
    plt.show()

    plt.clf()  