###############################################################################################
#
# Splitting facial emotion data set
#
# source: https://stackoverflow.com/questions/53074712/how-to-split-folder-of-images-into-test-training-validation-sets-with-stratified
# source: https://note.nkmk.me/en/python-random-choice-sample-choices/#random-sample-without-replacement-randomsample
# source: https://www.geeksforgeeks.org/python-shutil-copy-method/
#
###############################################################################################

import splitfolders # splitting data into test, val, train
import os # Ordner splitten
import random
import shutil

if False:
    dataset = r'C:\Studium\Data Analytics, M.Sc\Advanced Deep Learning\facial_emotion_dataset\dataset'
    output_folder = r'C:\Studium\Data Analytics, M.Sc\Advanced Deep Learning\Team Project Empty\facial_emotion_dataset'


    # Split data set
    splitfolders.ratio(dataset, output=output_folder, seed=1337, ratio=(.8, 0.1,0.1))


    # Definiere die Pfade für die Daten
    dataset_train = os.path.join('output_folder', 'train')
    dataset_val = os.path.join('output_folder', 'val')
    dataset_test = os.path.join('output_folder', 'test')

    print(f"Train Pfad: {dataset_train}")
    print(f"Val Pfad: {dataset_val}")
    print(f"Test Pfad: {dataset_test}")
    
# Absoluten Pfad des aktuellen Skripts ermitteln
current_dir = os.path.dirname(__file__)
print(current_dir)

# Pfade definieren
dataset = r'C:\Studium\Data Analytics, M.Sc\Advanced Deep Learning\facial_emotion_dataset\dataset'
output_folder = os.path.join(current_dir, "facial_emotion_dataset")

# Anzahl der Bilder pro Klasse
num_images_per_class = 500

# Temporärer Ordner für reduzierte Datenmenge
temp_dataset = os.path.join(output_folder, 'temp_dataset')

for class_folder in os.listdir(dataset):
    class_path = os.path.join(dataset, class_folder)
    temp_class_path = os.path.join(temp_dataset, class_folder)
    
    if os.path.isdir(class_path):
        # Liste aller Bilder im Klassenordner
        all_images = [img for img in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, img))]
        
        # Zufällige Auswahl von bis zu 500 Bildern
        selected_images = random.sample(all_images, min(len(all_images), num_images_per_class)) #random.sample ohne Duplikate
        
        # Erstellung des Zielordners und Kopie ausgewählter Bilder
        os.makedirs(temp_class_path, exist_ok=True)
        for img in selected_images:
            # shutil.copy(source, destination)
            shutil.copy(os.path.join(class_path, img), os.path.join(temp_class_path, img))

# Aufteilung des temporären Datasets in train, val und test
splitfolders.ratio(temp_dataset, output=output_folder, seed=1337, ratio=(.8, 0.1, 0.1))

# Definierung der Pfade für train, val und test
dataset_train = os.path.join(output_folder, 'train')
dataset_val = os.path.join(output_folder, 'val')
dataset_test = os.path.join(output_folder, 'test')

print(f"Train Pfad: {dataset_train}")
print(f"Val Pfad: {dataset_val}")
print(f"Test Pfad: {dataset_test}")

