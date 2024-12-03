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
from sklearn.metrics import precision_recall_fscore_support
import optuna
import os

# Absolut path of current script
current_dir = os.path.dirname(__file__)
print(current_dir)

# relative path of data sets
dataset_train = os.path.join(current_dir, "facial_emotion_dataset", "train")
dataset_val = os.path.join(current_dir, "facial_emotion_dataset", "val")
dataset_test = os.path.join(current_dir, "facial_emotion_dataset", "test")
# Device-Konfiguration
device = torch.device('cuda')

# Number of classes
num_classes = 5

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
    

if False:
    test_loader = get_test_loader(
        data_dir = dataset_test, # Pfad zu den Testdaten
        batch_size = batch_size
    )


    model = AlexNet(num_classes).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

    # Train the model
    total_step = len(train_loader)

    total_step = len(train_loader)


    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):  
            
            print(images.shape)
        
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            print(images.shape)
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, num_epochs, i+1, total_step, loss.item()))

        # Validation
        with torch.no_grad():
            correct = 0
            total = 0

            all_val_labels = []
            all_val_preds = []

            for images, labels in valid_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                # Speichern der Labels und Vorhersagen für Berechnung
                all_val_labels.extend(labels.cpu().numpy())
                all_val_preds.extend(predicted.cpu().numpy())
                
                del images, labels, outputs

            # Berechnung der Validation Accuracy
            val_accuracy = 100 * correct / total

            print('Accuracy of the network on the {} validation images: {} %'.format(total, val_accuracy)) 

            # Berechnung von Precision, Recall, F1-Score für Validation
            val_precision, val_recall, val_f1, _ = precision_recall_fscore_support(all_val_labels, all_val_preds, average='weighted')
            
            # Ausgabe und Speicherung der Validierungsmetriken
            print(f'Validation Precision: {val_precision:.4f}')
            print(f'Validation Recall: {val_recall:.4f}')
            print(f'Validation F1-Score: {val_f1:.4f}')

    # Testing
    with torch.no_grad():
        correct = 0
        total = 0

        all_test_labels_alexNet = []
        all_test_preds_alexNet = []

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Speichern der Labels und Vorhersagen für spätere Berechnung
            all_test_labels_alexNet.extend(labels.cpu().numpy())
            all_test_preds_alexNet.extend(predicted.cpu().numpy())

            del images, labels, outputs

        # Berechnung der Test Accuracy
        test_accuracy_alexNet = correct / total
        print('Accuracy of the network on the {} test images: {} %'.format(total, test_accuracy_alexNet))
        
        # Berechnung von Precision, Recall, F1-Score für den Test
        test_precision_alexNet, test_recall_alexNet, test_f1_alexNet, _ = precision_recall_fscore_support(all_test_labels_alexNet, all_test_preds_alexNet, average='weighted')
        
        # Ausgabe und Speicherung der Testmetriken
        print(f'Test Precision AlexNet: {test_precision_alexNet:.4f}')
        print(f'Test Recall AlexNet: {test_recall_alexNet:.4f}')
        print(f'Test F1-Score AlexNet: {test_f1_alexNet:.4f}')



# Optuna objective function for hyperparameter tuning
# https://medium.com/@boukamchahamdi/fine-tuning-a-resnet18-model-with-optuna-hyperparameter-optimization-2e3eab0bcca7
def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128])
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
study.optimize(objective, n_trials=20)

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

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay = 0.005, momentum = 0.9)  

    # Train the model
    total_step = len(train_loader)

    total_step = len(train_loader)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Train the model with the best hyperparameters
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
    
    print(f"Final Validation Loss: {val_loss:.4f}, Final Validation Accuracy: {val_acc:.4f}")

    return model

# Train the final model with the best hyperparameters
final_model = train_final_model(best_params, dataset_train, dataset_val, device)

# Load test data
test_loader = get_test_loader(
    data_dir= dataset_test, # Pfad zu den Testdaten
    batch_size=best_params['batch_size']
)

# Class name list
#class_names = ["Angry", "Happy", "Neutral", "Sad", "Surprise"]

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
torch.save(final_model, 'alexnet_model.pth')


