# ADL-Project
Deep Learning project: Sign Language Recognition  - American Sign Language 

Authors: Anna Reiter, Chiara Perocco

Before starting please install the requirements.txt file by running the command "pip install -r requirements.txt"

To be able to run the Large Language Model follow the following steps:
1. Install the ollama-model from https://ollama.com/ (for more information installing ollama see: https://github.com/ollama/ollama)
2. In the command prompt check if the installation is correct using the command "ollama list" (ollama should return the path of the ollama installation)
3. Download the model llama3.1 via the command prompt using the command "ollama pull llama3.1"
4. Check in the command prompt if the version is downloaded, "ollama list" should show the available models
5. Add the path to the ollama installation to the system varaible of your device for using ollama in Visual Studio Code
6. Install pandoc from https://github.com/jgm/pandoc/releases/tag/3.6.1 and add it to system variables (add the whole installed pandoc folder to your system variables: C:\Program Files\Pandoc\ )
7. Install Miktex for using pandoc https://miktex.org/download and add it to your system variables (add only the file from the bin folder to the system variables: C:\Program Files\MiKTeX\miktex\bin\x64\ )

New installation windows may open while the main file is being executed. These packages must be installed too.


We have different files for training our models. For some training Optuna was used to adjust the hyperparameters automatically (see the files AlexNet.py, VisionTransfomer.Py). Because of high computing power we decided to adjust the hyperparameters manually by running different experiments. 

The webcam is initialized with the index 0 in the file "Webcam.py" for using the intern webcam of your device. Change it to index 1, if you would like to use an extern webcam.