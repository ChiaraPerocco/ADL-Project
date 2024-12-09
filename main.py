###################################################################################################
#
# Main File
#
###################################################################################################
### Import packages
import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] ='0' # enable oneDNN custom operations --> different numericl results due to floating-point round-off errors from different computation errors

# get the path of current_dir
current_dir = os.path.dirname(__file__)

### Import functions
from EvaluationModel import test_model
from Webcam import save_frame_camera_key, dir_path
from LLM import create_article_pdf, generate_image_caption, generate_answer_for_section, draw_wrapped_text


### Load models
#device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#ViT = torch.load('ViT_model.pth', map_location=torch.device('cpu'))

# just for a test run
batch_size = 64 # need to include hyperparameters
ViT = torch.load('ViT_model.pth')
#ResNet = torch.load('resnet50_model.pth')
#AlexNet = torch.load('alexnet_model.pth')

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
    data_dir= dir_path, # Pfad zu den Webcam Daten
    batch_size=64  # needs to be modified???!!!
)

### Get the predicted label
# Run the model with the saved images from the webcam
test_acc_ViT, precision_ViT, recall_ViT, f1_ViT, all_labels_ViT, all_preds_ViT = test_model(ViT, test_loader)


### LLM

# Load diffusion model picture
image_path = os.path.join(current_dir, "DiffusionModelOutput", 'generated_image.png')  # Change the path as needed
# Load image caption
caption = generate_image_caption(image_path)


### Main function that leads the user through each stage of the process.
def main(photo, photo_shot, run):
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
        #all_preds_alexnet
        print("at the moment not available")
    elif model == "Resnet50":
        #all_preds_resnet
        print("at the moment not available")
    elif model == "Vision Transformer":
        all_preds_ViT
    else:
        print("Unknown model")

    # Execute the LLM
    if llm == "Ja":
        if question == "default":
            question = "How has the use of sign language evolved over the years?"
            pdf_path = os.path.join(current_dir, "Article", 'article.pdf') 
            create_article_pdf(question, image_path, caption, pdf_path)
        elif question == "Own":
            print("Not implemented yet")

    elif llm == "Nein":
        print("Artikel war nicht erwünscht")

    else:
        print("Unbekannte Eingabe")


if __name__ == "__main__":
    photo = input("Möchten sie ein Webcam Bild aufnehmen? -Ja/Nein \n")
    photo_shot = input("Zum aufnehmen des Bildes drücken sie die Taste 'c'. \n Zum Beenden der Webcam ohne Aufnahme drücken sie 'q'. \n Verstanden? Ja/Nein \n")
    run = input("Möchten sie nun das Bild klassifizieren? -Ja/Nein \n")
    model = input("Welches Modell möchten sie gerne verwenden? (AlexNet, Resnet50, Vision Transformer) \n")
    llm = input("Möchten sie nun einen Artikel zu dem Bild erhalten?- Ja/Nein \n")
    question = input("Soll der Artikel auf einer bestimmten Frage basieren oder soll ein default Artikel erstellt werden? -Default/Own")
    main(photo, photo_shot, run)
