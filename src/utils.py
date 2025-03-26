""" Provide python functions which can be used by multiple python files


"""

import pandas as pd
import numpy as np
from pathlib import Path
from keras.models import model_from_json
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import cv2
from model import efficientNet 

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def create_directory(directory):
    """Create directory under the current path

    Args:
        directory: String formatted directory name
    
    """
    try:

        # Get the current path
        current_path = os.getcwd()

        # Merge the current path with the desired one
        path = os.path.join(current_path, directory)

        # If the directory exists, do nothing. Otherwise, create it
        if os.path.isdir(path):
            return
        else:
            os.mkdir(path)

    except Exception as e:
        print(f"Creating directory failed. Error: {e}")



def load_dataset(image_path):
    """Download dataset.

    This function downloads the BloodMNIST dataset from the medmnist library

    Args:
            dataset_name(str): The dataset name to be downloaded

    Returns:
            Training, validation and test datasets.

    """
    try:

        TARGET_SIZE = 224

        train = pd.read_csv(image_path+'/train.csv')

        image_name = train['image_id'].to_list()
        labels = train['label'].to_list()

        img = []
        for i in range(len(image_name)):
        
            img.append(np.array(cv2.resize(cv2.imread(image_path+'/train/'+image_name[i])
                                           ,(TARGET_SIZE,TARGET_SIZE))))
        
        train_img = np.array(img)

        print("Computer vision read checkpoint")

        print(len(train_img))
        x_train, x_temp, y_train, y_temp = train_test_split(train_img, labels,
                                                            test_size=0.3, random_state=42)
        
        x_val, x_test, y_val, y_test = train_test_split(x_temp, y_temp,
                                                            test_size=0.5,
                                                            random_state=42)
        
        print("Dataset split checkpoint")

        print(f"Training shape: {str(x_train.shape)}")
        print(f"Validation shape: {str(x_val.shape)}")
        print(f"Testing shape: {str(x_test.shape)}")

        return x_train, y_train, x_val, y_val, x_test, y_test

    except Exception as e:
        print(f"Downloading dataset failed. Error: {e}")



def normalize_dataset(train_dataset, validation_dataset, test_dataset):
    """Normalize dataset.

    This function performs data transform via normalization.

    Args:
            training, validation and test datasets.

    Returns:
            normalized training, validation and test datasets.

    """

    try:
        
        # Performing data transformation via normalization
        train_dataset = train_dataset/255.0
        validation_dataset = validation_dataset/255.0
        test_dataset= test_dataset/255.0

        return train_dataset, validation_dataset, test_dataset
    
    except Exception as e:
        print(f"Data normalization failed. Error: {e}")



def save_model(model, model_name):
    """Save CNN model.

    This function saves CNN model and weights as json and .h5 files respectively.

    Args:
            CNN model
            model_name(str)
            

    """

    try:

        # Convert the model structure into json
        model_structure = model.to_json()

        # Creates a json file and writes the json model structure
        file_path = Path(f"./model/{model_name}.json")
        file_path.write_text(model_structure)

        # Saves the weights as .h5 file
        model.save_weights(f"./model/{model_name}.weights.h5")

    except Exception as e:
        print(f"Saving the EfficientNet model failed. Error: {e}")



def load_model(model_name):
    """Save CNN model.

    This function loads the saved CNN model and weights to be used later on.

    Args:
            model_name(str)
            
    Returns:
            CNN model

    """

    try:
        
        # Locate the model structure file
        file_path = Path(f"./model/{model_name}.json")

        # Read the json file and extract the CNN model
        model_structure = file_path.read_text()
        model = model_from_json(model_structure)

        # Load the CNN weights
        model.load_weights(f"./model/{model_name}.weights.h5")

        return model
    
    except Exception as e:
        print(f"Loading the EfficientNet model failed. Error: {e}")



def plot_accuray_loss(model_history):
    """Plot accuracy loss graphs for the CNN model.

    This function plots the CNN model's accuracy and loss against epoch into a fig file.

    Args:
            model history

    """

    try:

        # Create the subplots variables.
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10,7))

        # Plot the accuracy subplot
        accuracy = model_history.history['accuracy']
        validation_accuracy = model_history.history['val_accuracy']
        epochs = range(1, len(accuracy)+1)
        ax1.plot(epochs, accuracy, label="Training Accuracy")
        ax1.plot(epochs, validation_accuracy, label="Validation Accuracy")
        ax1.set_title('Training and validation accuracy')
        ax1.set_xlabel('Number of Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid()

        # Plot the loss subplot
        loss = model_history.history['loss']
        val_loss = model_history.history['val_loss']
        ax2.plot(epochs, loss, label="Training loss")
        ax2.plot(epochs, val_loss, label="Validation loss")
        ax2.set_title('Training and validation loss')
        ax2.set_xlabel('Number of Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid()

        # Save the subplots file.`125   q`
        fig.savefig(f'./figures/Efficient_accuracy_loss_33.png')
    
    except Exception as e:
        print(f"Plotting accuracy and loss has failed. Error: {e}")


def visualise_subset(train_dataset, labels):
    """Visualise Subset.

    This function visualises a subset of the training dataset images data.

    Args:
            training datasets.

    Returns:
            Images saved in figures folder.

    """

    try:

        plt.figure(figsize=(12,10))
        for x in range(6):
            plt.subplot(2, 3, x+1)
            plt.imshow(train_dataset[x], cmap=plt.get_cmap('gray'))
            plt.title(f"Class {labels[x]}")

        plt.savefig("./figures/subset_images.jpeg")
        plt.close()

    except Exception as e:
        print(f"Visualising subset failed. Error: {e}")