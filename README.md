# ELEC0135 AMLS II assingment 2024/2025

There is one task for this assignment:
1. Cassava leaf disease multi-class classification.
https://www.kaggle.com/competitions/cassava-leaf-disease-classification/overview

## What are the folders and files
1. Main folder contains main.py file, which runs other instances of python scripts.
2. `model` folder which contains EfficientNet model files needed when running the scripts.
4. `src` folder contain `utils.py` script. This file is called by all ML model scripts in this assignment to perform a centralized tasks, such as: 
   - Downloading Cassava leaf disease dataset.
   - Split dataset into train, validation and test.
   - Output a subset of a dataset.
   - Plot accuracy and loss graphs.
   - Save EfficientNet model as json and .weight.h5 file
   - Load EfficientNet model.
   - Creating directories.

## Important note before the procedure.
1. `main.py` has one arguments set.
   - `decision`, which the user can define how to run the models. You can either run on the training, validation and test dataset (training the model from scratch) using `-d train` or test the loaded model using test dataset by adding no input. The default is set to `test`.
2. `Dataset` folder does not exist, but it will be created via `utils.py`. 
3.  The images and labels are imported via google drive within the `utils.py` file (no manual inetrvention needed). 
   - The data in the google drive is a zip file of images folder and their labels' csv file originally from the Kaggle dataset.

  
## Procedures

1. You should be able to see a `requirements.txt` file, which contains all the libraries needed for this assignment. To install all the libraries from it, simply use `pip install -r requirements.txt` from the command line.

2. Once installed, you can start running the tasks. There multiple ways to do it.
    - If you want to run the models for the tasks as running models on the test datasets only, then run `python main.py`
    - If you want to run the models for the tasks as performing the entire training and validation process , then run `python main.py -d train`