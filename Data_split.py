###############################################################################################
#
# Splitting facial emotion data set
#
###############################################################################################

import splitfolders # splitting data into test, val, train
import os # Ordner splitten

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