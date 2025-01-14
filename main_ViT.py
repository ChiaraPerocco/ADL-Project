###################################################################################################
#
# Main File
#
###################################################################################################
### Import packages
import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from sklearn.metrics import precision_recall_fscore_support
from LLM_final import generate_article
#from LLM_ollama_agent_Newparser import generate_letter_article, create_chain
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] ='0' # enable oneDNN custom operations --> different numericl results due to floating-point round-off errors from different computation errors


### Import functions
from Webcam import save_frame_camera_key, dir_path
#from LLM import create_article_pdf, generate_image_caption, generate_answer_for_section, draw_wrapped_text
from mediapipe_webcam_images import image_processing

# get the path of current_dir
current_dir = os.path.dirname(__file__)

### Check for device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)

### Image processing folders
input_folder = os.path.join(current_dir, "webcam_images")
output_folder = os.path.join(current_dir, "webcam_images_processed")

# Erstelle den Zielordner, falls er nicht existiert
os.makedirs(output_folder, exist_ok=True)

# Aufnahme des Kamerabildes
#save_frame_camera_key('camera capture', dir_path=input_folder)

# Check if the directory contains any files and delete them if necessary
# Alle .jpg-Dateien im Ordner löschen
for datei in os.listdir(output_folder):
    if datei.lower().endswith('.jpg'):
        os.remove(os.path.join(output_folder, datei))

# Process image
# Aufruf der Funktion aus dem anderen Skript
image_processing(input_folder, output_folder)

# Path of image that we want to classify
dir_path = os.path.join(current_dir, "webcam_images_processed")

# Load Hyperparameters


# ViT
batch_size = 64
learning_rate = 0.0001
num_epochs = 30


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

# Webcam Image Class
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
        
        return image, img_name  # Bild und Dateiname (kann später zur Ausgabe verwendet werden)


# Get test loader
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
    webcam_image = WebcamImageDataset(image_dir=data_dir, transform=transform)

    # Erstellen des DataLoaders für Testdaten
    webcam_image_loader = DataLoader(webcam_image, batch_size=batch_size, shuffle=shuffle)

    return webcam_image_loader

# Load test data
test_loader = get_test_loader(
    data_dir= dir_path, # Pfad zu den Webcam Daten
    batch_size=batch_size  # needs to be modified???!!!
)

# Test the model on the test data
def test_model(model, test_loader):
    model.eval()

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            #labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

    return preds.item()



# Modell initialisieren
def initialize_model(num_classes):
    model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

    # Alle Schichten einfrieren
    for param in model.parameters():
        param.requires_grad = False

    # Den Klassifikator (head) anpassen
    model.heads.head = nn.Sequential(
        # Dropout für Regularisierung
        #nn.Dropout(0.5), # dropout ViT model 5
        nn.Dropout(0.3), # dropout ViT model 7
        nn.Linear(model.heads.head.in_features, num_classes)
    )

    model = model.to(device)
    return model

model = initialize_model(26)

final_model = torch.load("ViT_model_dataset2_8.pth", map_location=torch.device('cpu'), weights_only=True)
model.load_state_dict(final_model)

model.eval()


### Get the predicted label
# Run the model with the saved images from the webcam
prediction_ViT = test_model(model, test_loader)

print(f'ViT prediction: {prediction_ViT}')

# change the predicted class into the letter

# Funktion zur Umwandlung einer Zahl in einen Buchstaben
def letter(number):
    # Basis-Buchstaben von A-Z (ASCII 65-90)
    alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    # Die Zahl mod 26 nehmen, um sie in den Bereich 0-25 zu bringen
    index = number % 26
    # Den entsprechenden Buchstaben aus dem Alphabet auswählen
    return alphabet[index]

# Umwandlung in Buchstaben
detected_letter = letter(prediction_ViT)

print(f"Die vorhergesagte Zahl {prediction_ViT} entspricht dem Buchstaben '{detected_letter}'.")

generate_article(detected_letter)

### LLM


"""
# Load diffusion model picture
image_path = os.path.join(current_dir, "DiffusionModelOutput", 'generated_image.png')  # Change the path as needed
# Load image caption
caption = generate_image_caption(image_path)


### Main function that leads the user through each stage of the process.
def main(photo, photo_shot, run, model):
    # Captures and saves the image from the webcam
    if photo == "Ja":
        if photo_shot == "Ja":
            save_frame_camera_key('camera_capture', dir_path=dir_path)
        elif photo_shot == "Nein":
            print("Unbekannte Aktion")
    elif photo == "Nein":
        print("Abbruch der Bildaufnahme")
    else:
        print("Unbekannte Aktion")
    
    # Run the model with the saved photo
    if run == "Ja":
       # test_model(ViT, dir_path)
       print("Here need to add the test model function for specific model")
    elif run == "Nein":
        print("Selbst gewünschter Abbruch")
    else:
        print("Unbekannte Aktion")

    # Execute the model with the specific model
    if model == "AlexNet":
        prediction_alexNet
    elif model == "Resnet50":
        prediction_resnet
    elif model == "Vision Transformer":
        prediction_ViT
    else:
        print("Unknown model")


    # Create Images with Diffusion Model or take existing

    # Execute the LLM
    #if llm == "Ja":
    #    if question == "default":
     #       question = "How has the use of sign language evolved over the years?"
    #        pdf_path = os.path.join(current_dir, "Article", 'article.pdf') 
    #        create_article_pdf(question, image_path, caption, pdf_path)
    #    elif question == "Own":
    #        print("Not implemented yet")

    #elif llm == "Nein":
    #    print("Artikel war nicht erwünscht")

    #else:
    #    print("Unbekannte Eingabe")


if __name__ == "__main__":
    photo = input("Möchten sie ein Webcam Bild aufnehmen? -Ja/Nein \n")
    photo_shot = input("Zum aufnehmen des Bildes drücken sie die Taste 'c'. \n Zum Beenden der Webcam ohne Aufnahme drücken sie 'q'. \n Verstanden? Ja/Nein \n")
    run = input("Möchten sie nun das Bild klassifizieren? -Ja/Nein \n")
    model = input("Welches Modell möchten sie gerne verwenden? (AlexNet, Resnet50, Vision Transformer) \n")
    #llm = input("Möchten sie nun einen Artikel zu dem Bild erhalten?- Ja/Nein \n")
    #question = input("Soll der Artikel auf einer bestimmten Frage basieren oder soll ein default Artikel erstellt werden? -Default/Own")
    main(photo, photo_shot, run, model)


# Call main function with all inputs and necessary variables
#main(photo, photo_shot, run, model, llm, question, dir_path, current_dir, image_path, caption)
main(photo, photo_shot, run, model, dir_path, current_dir)
"""