""" main.py is a centralised module where the process for both tasks initiates


"""

from src import utils
from model import efficientNet
import argparse
import os
import warnings

warnings.filterwarnings("ignore")


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def task(image_path, decision):
    """ Runs the EffcientNet model for Cassava leaf disease dataset


    """

    print("################ Cassava leaf disease classification task is starting ################\n")


    # Download the dataset
    print('Loading the dataset..')
    x_train, y_train, x_val, y_val, x_test, y_test = utils.load_dataset(image_path)

    # visualise a subset of the dataset
    utils.visualise_subset(x_train, y_train)


    # Run the efficientNet model
    if decision == 'train':
        print('Performing model training, then testing..')
        efficientNet.EfficientNet_model_training(x_train, y_train, x_val, y_val)
        efficientNet.EfficientNet_model_testing(x_test, y_test)
    elif decision == 'test':
        print('Performing model testing..')
        efficientNet.EfficientNet_model_testing(x_test, y_test)

    print("\n################ Cassava leaf disease classification task via efficientNet has finished ################")


if __name__ == "__main__":

    # Decision argument
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--decision', default='test',
                        help ='select the task')
    
    args = parser.parse_args()
    decision = args.decision

    # Create figures folder
    utils.create_directory("figures")

    # Run the required functions depending on user's input
    dataset_path = "./Dataset/"
    task(dataset_path, decision)

    



    






