###############################################################################################
#
# Splitting Mushroom data set
#
###############################################################################################

import splitfolders # splitting data into test, val, train
import os # Ordner splitten

dataset = r'C:\Users\annar\Documents\Master\Advanced Deep Learning\Facial emotion data set\facial_emotion_dataset\dataset'
output_folder = r'C:\Users\annar\Documents\Master\Advanced Deep Learning\Facial emotion data set\facial_emotion_dataset\dataset_output'


# Split data set
splitfolders.ratio(dataset, output=output_folder, seed=1337, ratio=(.8, 0.1,0.1))


# Definiere die Pfade für die Daten
dataset_train = os.path.join('output_folder', 'train')
dataset_val = os.path.join('output_folder', 'val')
dataset_test = os.path.join('output_folder', 'test')

print(f"Train Pfad: {dataset_train}")
print(f"Val Pfad: {dataset_val}")
print(f"Test Pfad: {dataset_test}")