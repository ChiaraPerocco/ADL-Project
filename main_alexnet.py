###################################################################################################
#
# Main File
#
###################################################################################################

import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from Webcam import save_frame_camera_key, dir_path
from mediapipe_webcam_images import image_processing
from LLM_final import generate_article


def initialize_model(num_classes):
    class AlexNet(nn.Module):
        def __init__(self, num_classes=num_classes):
            super(AlexNet, self).__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=0),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU(),
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(384),
                nn.ReLU(),
            )
            self.layer5 = nn.Sequential(
                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(6400, 4096),
                nn.ReLU(),
            )
            self.fc1 = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(),
            )
            self.fc2 = nn.Sequential(nn.Linear(4096, num_classes))

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

    return AlexNet(num_classes)


class WebcamImageDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_names = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        img_name = os.path.join(self.image_dir, self.image_names[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, img_name


def get_test_loader(data_dir, batch_size, shuffle=True):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    )
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    webcam_image = WebcamImageDataset(image_dir=data_dir, transform=transform)
    webcam_image_loader = DataLoader(webcam_image, batch_size=batch_size, shuffle=shuffle)
    return webcam_image_loader


def test_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
    return preds.item()


def letter(number):
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    index = number % 26
    return alphabet[index]


if __name__ == "__main__":
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    current_dir = os.path.dirname(__file__)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    input_folder = os.path.join(current_dir, "webcam_images")
    output_folder = os.path.join(current_dir, "webcam_images_processed")
    os.makedirs(output_folder, exist_ok=True)

    save_frame_camera_key('camera capture', dir_path=input_folder)

    for file in os.listdir(output_folder):
        if file.lower().endswith('.jpg'):
            os.remove(os.path.join(output_folder, file))

    image_processing(input_folder, output_folder)

    dir_path = os.path.join(current_dir, "webcam_images_processed")

    batch_size = 64
    learning_rate = 0.0008
    num_epochs = 50
    num_classes = 26


    current_dir = os.path.dirname(__file__)

    model_path = os.path.join(current_dir, "Models", "alexnet_model_dataset2_4.pth")
    model = initialize_model(num_classes)
    final_model = torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
    model.load_state_dict(final_model)
    model.eval()

    test_loader = get_test_loader(data_dir=dir_path, batch_size=batch_size)
    prediction_alexnet = test_model(model, test_loader)

    detected_letter = letter(prediction_alexnet)
    print(f"The predicted number {prediction_alexnet} stands for the letter '{detected_letter}'.")

    generate_article(detected_letter)
