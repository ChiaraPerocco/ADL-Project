# ADL-Project
Deep Learning project: Sign Language Recognition  - American Sign Language 

Authors: Anna Reiter, Chiara Perocco

#### Installation requirements 
    Before starting please install the requirements.txt file by running the command "pip install -r requirements.txt"

    To be able to run the Large Language Model follow the following steps:
-    **1.** Install the ollama-model from https://ollama.com/ (for more information installing ollama see: https://github.com/ollama/ollama)
-    **2**. In the command prompt check if the installation is correct using the command "ollama list" (ollama should return the path of the ollama installation)
-    **3**. Download the model llama3.1 via the command prompt using the command "ollama pull llama3.1"
-    **4**. Check in the command prompt if the version is downloaded, "ollama list" should show the available models
-    **5**. Add the path to the ollama installation to the system varaible of your device for using ollama in Visual Studio Code
-    **6**. Install pandoc from https://github.com/jgm/pandoc/releases/tag/3.6.1 and add it to system variables (add the whole installed pandoc folder to your system variables: C:\Program Files\Pandoc\ )
-    **7**. Install Miktex for using pandoc https://miktex.org/download and add it to your system variables (add only the file from the bin folder to the system variables: C:\Program Files\MiKTeX\miktex\bin\x64\ )

    New installation windows may open while the main file is being executed. These packages must be installed too.

#### Project Structure

- **Domain** 
  The folder `Domain` contains an Image of the American Sign Language Alphabet. These signs can be classificated through the project. The numbers in the image stands for the classes in the dataset.

- **Model Training**  
  The scripts for training the AlexNet, ResNet50, and Vision Transformer models are located in the `Networks` folder.  
  We have separate files for training each model. For some models, Optuna was used to automatically adjust the hyperparameters (see `AlexNet.py` and `VisionTransformer.py`). However, due to the high computational resources required, we decided to manually adjust the hyperparameters by running different experiments.

- **Model Evaluation**  
  The evaluation plots are stored in the `Evaluation` folder.  
  Saliency maps are saved in the `Saliency Maps` folder.

- **Dataset**  
  The `Sign Language` folder contains `ds1`, while the `Sign Language 2` folder contains `ds2`.
  The `Data_split.py` script is used to select a subset of the original datasets and split them into training, validation, and test datasets.
  The `mediapipe_test.py` script further extends these datasets by generating images that focus exclusively on the hand regions. The extended datasets are then saved with the suffix `_processed`. These processed datasets were subsequently used for training.

- **Webcam**  
  Webcam images are stored in the `webcam_images` folder. These images are processed by the `mediapipe_webcam_images.py` script, and the processed images are then used for letter classification.  
  The webcam is initialized with index 0 in the `Webcam.py` file, which uses the internal webcam of your device. To use an external webcam, change the index to 1.  
  If an error occurs while executing the main function, it might be due to the `mediapipe_webcam_images.py` script not recognizing a hand in the image. In this case, the message "Hand wurde nicht erkannt. Nehme noch ein Foto auf" will be printed.  
  Please check the processed image in the `webcam_images_processed` folder as it can sometimes be distorted. This distortion can make it difficult to recognise the correct letter. For best letter classification results, keep a reasonable distance from the webcam and exaggerate hand gestures to improve recognition.

- **Article Generation**  
  The final language model (LLM) is produced in the `LLM_final.py` file. This file contains the agent, tools, and article generation process.  
  The generated articles are saved in the `Article` folder when executed in the main function.

- **Execution**  
  To execute the program, you can run `main_alexnet.py`, `main_resnet50.py`, or `main_vit.py`. The best classification results are achieved using `main_alexnet.py` and `main_vit.py`.





    