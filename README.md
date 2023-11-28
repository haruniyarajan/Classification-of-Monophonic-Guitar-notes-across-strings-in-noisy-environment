#Classification of Monophonic Guitar Notes Across Strings in Noisy Environments

## Introduction
This project aims to classify monophonic guitar notes across strings in various noisy environments using Convolutional Neural Networks (CNNs). The goal is to accurately identify the notes and strings played, even in the presence of background noise.

#Running the Project in Google Collab

##Open the Project in Google Collab
Open the project notebook in Google Colab to access the interactive environment.

##Mount Google Drive
Run the following code to mount your Google Drive, which allows access to your files stored in Google Drive:

from google.colab import drive
drive.mount('/content/gdrive')

##Navigate to the Project Directory
Use the following code to navigate to the directory where your data is stored in Google Drive:

The below code is for downloading the audio_hex_pickup_debleeded file containing the audio recordings. This code will navigate to the directory to where you would want to store our data. Below given is an example. The data will be downloaded in a Guitarset folder under My drive in gdrive, provided the appropriate folders have been created in advance.

import os
os.chdir("/content/gdrive/My Drive/Guitarset/audio_hex-pickup_debleeded")

Post this, please follow the instructions in the dataset section below for downloading the audio_hex-pickup_debleeded and unzipping it.

Follow similar instructions for downloading annotations.zip file. The below code will navigate the directory to gdrive -> My Drive -> Guitarset -> Annotation, provided the mentioned folders have been created in advance.

import os
os.chdir("/content/gdrive/My Drive/Guitarset/annotation")

Post this, please follow the instructions in the dataset section below for downloading the annotations.zip and unzipping it.

The above given is an example used in the python notebook for implementation of the project and can be adjusted according to the needs of the user.

## Dataset
The data used in this project comes from the [GuitarSet dataset](https://guitarset.weebly.com/), a collection of audio recordings of different guitar performances, along with corresponding annotations. The dataset includes various features such as pitch, timing, and string information.

### Downloading the Dataset
The dataset should be downloaded from the above mentioned website to the local machine and run the below mentioned commands to import the dataset in Google Collab environment.

```bash
!wget "https://zenodo.org/record/3371780/files/audio_hex-pickup_debleeded.zip?download=1" -O audio_hex-pickup_debleeded.zip
!unzip -o audio_hex-pickup_debleeded.zip
!wget "https://zenodo.org/record/3371780/files/annotation.zip?download=1" -O annotation.zip
!unzip -o annotation.zip

##Dependencies
The following libraries and tools are required to run the project:

librosa
numpy
matplotlib
seaborn
pandas
jams
tensorflow
keras
sklearn
imblearn
tqdm
keras-tuner
pydot
graphviz

###Installation
You can install the required dependencies using the following commands:

```bash
!pip install librosa numpy matplotlib seaborn pandas jams tqdm scikit-learn imbalanced-learn tensorflow keras keras-tuner pydot graphviz


##Code Structure
Final Guitar Note Classification.ipynb: The python notebook containing the entire code implementation.It includes preprocessing the GuitarSet audio data and training and evaluating the CNN model.

##Usage
Clone the repository to your local machine.
Download the GuitarSet dataset as described in the Dataset section.
Install the required dependencies.
Run the Final Guitar Note Classification.ipynb notebook.

##Model Architecture
The project employs a Convolutional Neural Network (CNN) with various layers including convolutional, max-pooling, dropout, and dense layers. Hyperparameter tuning is performed using Keras Tuner, and the model is evaluated using metrics such as loss, precision, recall, accuracy, F1 score, and AUC.

##Results
The model demonstrates promising accuracy in string classification (78.63%), while note classification presents challenges with significant variations in performance across different classes. Further tuning and exploration of different neural network architectures may enhance performance.

##Acknowledgements
Credit to the creators of the GuitarSet dataset and the open-source libraries used in this project.



