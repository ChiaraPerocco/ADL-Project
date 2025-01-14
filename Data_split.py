###############################################################################################
#
# Splitting ASL data set
#
# source: https://stackoverflow.com/questions/53074712/how-to-split-folder-of-images-into-test-training-validation-sets-with-stratified
# source: https://note.nkmk.me/en/python-random-choice-sample-choices/#random-sample-without-replacement-randomsample
# source: https://www.geeksforgeeks.org/python-shutil-copy-method/
# source: https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset
#
###############################################################################################

import splitfolders 
import os 
import random
import shutil


current_dir = os.path.dirname(__file__)
print(current_dir)

# define paths
#dataset = r"C:\Studium\Data Analytics, M.Sc\Advanced Deep Learning\datasetSL\ASL_Alphabet_Dataset\asl_alphabet_train"
dataset = r"C:\Users\Christoph\Documents\Anna\ASL_Alphabet_Dataset\asl_alphabet_train"
save_path = os.path.join(current_dir, "Sign Language 2")
os.makedirs(save_path, exist_ok=True)
output_folder = os.path.join(current_dir, "Sign Language 2")

# number of images per class
num_images_per_class = 700

# temporary folder for reduced dataset
temp_dataset = os.path.join(output_folder, 'temp_dataset_2')
os.makedirs(temp_dataset, exist_ok=True)

for class_folder in os.listdir(dataset):
    class_path = os.path.join(dataset, class_folder)
    temp_class_path = os.path.join(temp_dataset, class_folder)
    
    if os.path.isdir(class_path):
    
        all_images = [img for img in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, img))]
        
        # random choice of images
        selected_images = random.sample(all_images, min(len(all_images), num_images_per_class)) 
        
        
        os.makedirs(temp_class_path, exist_ok=True)
        for img in selected_images:
            # shutil.copy(source, destination)
            shutil.copy(os.path.join(class_path, img), os.path.join(temp_class_path, img))


#splitfolders.ratio(temp_dataset, output=output_folder, seed=1337, ratio=(.8, 0.1, 0.1))
splitfolders.ratio(temp_dataset, output=output_folder, seed=1223, ratio=(.8, 0.1, 0.1))

# Definierung der Pfade für train, val und test
dataset_train = os.path.join(output_folder, 'train_2')
os.makedirs(dataset_train, exist_ok=True)

dataset_val = os.path.join(output_folder, 'val_2')
os.makedirs(dataset_val, exist_ok=True)

dataset_test = os.path.join(output_folder, 'test_2')
os.makedirs(dataset_test, exist_ok=True)

print(f"Train Pfad: {dataset_train}")
print(f"Val Pfad: {dataset_val}")
print(f"Test Pfad: {dataset_test}")

