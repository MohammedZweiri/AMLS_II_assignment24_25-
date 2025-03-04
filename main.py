""" main.py is a centralised module where the process for both tasks initiates


"""

from src import utils
from model import efficientNet
import argparse
import numpy as np


def task(image_path):
    """ Runs the CNN model for bloodMNIST dataset


    """

    print("################ Task B via CNN is starting ################")
    print('\n')

    # Download the dataset
    x_train, y_train, x_val, y_val, x_test, y_test = utils.load_dataset(image_path)

    # Transform data using normalization
    #normalised_train, normalised_val, normalised_test = utils.normalize_dataset(x_train, x_val, x_test)

    # visualise a subset of the dataset
    utils.visualise_subset(x_train, y_train)

    # Perform image augmentation
    efficientNet.image_augmentation(x_train)

    # # Run the CNN model

    # if decision == 'train':
    #     CNN_B.CNN_model_training(train_dataset, validation_dataset, test_dataset)
        
    # elif decision == 'test':
    #     CNN_B.CNN_model_testing(test_dataset)

    # print('\n')
    # print("################ Task B via CNN has finished ################")


if __name__ == "__main__":
    # Create Datasets folder
    utils.create_directory("figures")
    dataset_path = "./Dataset/"
    task(dataset_path)


    






