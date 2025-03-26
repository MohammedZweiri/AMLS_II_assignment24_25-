""" main.py is a centralised module where the process for both tasks initiates


"""

from src import utils
from model import efficientNet
import argparse
import numpy as np
import os


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def task(image_path):
    """ Runs the CNN model for bloodMNIST dataset


    """

    print("################ Cassava leaf disease classification task is starting ################")
    print('\n')


    # Download the dataset
    print('Loading the dataset..')
    x_train, y_train, x_val, y_val, x_test, y_test = utils.load_dataset(image_path)

    # Transform data using normalization
    #normalised_train, normalised_val, normalised_test = utils.normalize_dataset(x_train, x_val, x_test)

    # visualise a subset of the dataset
    utils.visualise_subset(x_train, y_train)


    # Perform image augmentation
    # print('Performing image augmentation..')
    # augmented_data = efficientNet.image_augmentation(x_train)

    # visualise a subset of the dataset
    #utils.visualise_augmentation(augmented_data)

    # # Run the CNN model

    # if decision == 'train':
    print('Performing model testing..')
    #efficientNet.EfficientNet_model_training(x_train, y_train, x_val, y_val)
    
    efficientNet.EfficientNet_model_testing(x_test, y_test)
    # elif decision == 'test':
    #     CNN_B.CNN_model_testing(test_dataset)

    # print('\n')
    # print("################ Task B via CNN has finished ################")


if __name__ == "__main__":
    # Create Datasets folder
    utils.create_directory("figures")
    dataset_path = "./Dataset/"
    task(dataset_path)


    






