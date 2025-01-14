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

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.metrics import precision_recall_fscore_support
import os
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt 
from sklearn.metrics import  confusion_matrix
import seaborn as sns

# Absolut path of current script
current_dir = os.path.dirname(__file__)
print(current_dir)

# relative path of data sets
dataset_test = os.path.join(current_dir, "Sign Language 2", "test_processed")

# Number of classes in data set
num_classes = 26

# Device configuration
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

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
    

    # load testdata
    test_dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # create dataloader for testdata
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return test_loader

#Resnet model 5
batch_size = 64
learning_rate = 0.001
num_epochs = 50
num_workers = 2  
drop_out_rate = 0.2


best_params = {
    'batch_size': batch_size,
    'learning_rate': learning_rate,
    'num_epochs': num_epochs
}

# Load test data
test_loader = get_test_loader(
    data_dir= dataset_test, # Pfad zu den Testdaten
    batch_size=best_params['batch_size']
)

# initialize model
def initialize_model(num_classes):
    # load pretrained ResNet50 model
    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    # freeze all layers except the last one
    for param in model.parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True

    # replace the fully connected layer
    model.fc = nn.Sequential(
        nn.Dropout(drop_out_rate),  # Dropout 
        nn.Linear(model.fc.in_features, num_classes)
    )
    
    model = model.to(device)


    return model


model = initialize_model(26)
final_model = torch.load("resnet50_model_dataset2_5.pth", map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(final_model)
model.eval()
### Get the predicted label
# Run the model with the saved images from the webcam
class_names = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]


# Test model
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
    true_labels = [class_names[i] for i in all_labels_ViT]
    predicted_labels = [class_names[i] for i in all_preds_ViT]

    # calculate test Accuracy
    test_acc_ViT = test_corrects.double() / len(test_loader.dataset)
    print(f'Test Accuracy: {test_acc_ViT:.4f}')
    # calculate Precision, Recall und F1-Score
    precision_ViT, recall_ViT, f1_ViT, _ = precision_recall_fscore_support(all_labels_ViT, all_preds_ViT, average='weighted')
        
    print(f'Test Accuracy: {test_acc_ViT:.4f}')
    print(f'Precision: {precision_ViT:.4f}')
    print(f'Recall: {recall_ViT:.4f}')
    print(f'F1-Score: {f1_ViT:.4f}')
        
    print(f'Labels Testdaten: {true_labels}')
    print(f'vorhergesagte Testdaten: {predicted_labels}')
    # return metrics
    return test_acc_ViT.item(), precision_ViT, recall_ViT, f1_ViT, all_labels_ViT, all_preds_ViT

# test and save metrics and labels
test_acc_ViT, precision_ViT, recall_ViT, f1_ViT, all_labels_ViT, all_preds_ViT = test_model(model, test_loader)


### Confusion Matrix
conf_matrix_ViT = confusion_matrix(all_labels_ViT, all_preds_ViT)

# Visualizing Confusion Matrix
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix_ViT, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.title('Confusion Matrix ResNet50')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()


# Create train, val loss and train, val accuracy plots
import wandb
api = wandb.Api()
run = api.run("annaretr01-hochschule-m-nchen/resnet50_model_dataset2_5/zbs9sdhu")  # Beispiel: "username/projectname/run-id"
# Verfügbare Spalten anzeigen
history = run.history(keys=["train_loss", "train_acc", "val_loss", "val_acc"])
print(history["train_loss"])


import matplotlib.pyplot as plt

# Validation and training Loss/accuracy
epochs = range(len(history["train_loss"]))

# Plot with two y-axis one for accuracy, one for loss
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot loss on the left y-axis
ax1.plot(epochs, history["train_loss"], label="train loss", color='r')
ax1.plot(epochs, history["val_loss"], label="validation loss", color='b')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.tick_params(axis='y')

# Create another y-axis for accuracy
ax2 = ax1.twinx()
ax2.plot(epochs, history["train_acc"], label="train accuracy", color='g')
ax2.plot(epochs, history["val_acc"], label="validation accuracy", color='y')
ax2.set_ylabel('Accuracy')
ax2.tick_params(axis='y')

# Add legends
ax1.legend(loc='upper left')
ax2.legend(loc='lower left')

plt.show()

# Saliency Maps
if True:
        
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
            saliency_map = GradCAM(model).saliency(data)

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
