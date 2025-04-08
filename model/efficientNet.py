"""Accomplishing cassava leaf disease classification via EfficientNet transfer learner.

    This module uses EfficientNet model to accuractly predict the 5 different
    classes of the cassava leaf diseases.

    """

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras.utils import to_categorical, plot_model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, Flatten, BatchNormalization, Input
from keras.optimizers import Adam
from keras.applications import EfficientNetB0
from keras import Input
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.utils import class_weight
from src import utils
import warnings

warnings.filterwarnings("ignore")


def evaluate_model(true_labels, predicted_labels, predict_probs, label_names):
    """Evaluate the EfficientNet model.

    This function evaluates the EfficientNet model and produces classification report and confusion matrix

    Args:
            true_labels
            predicted_labels
            predict_probs
            label_names

    """

    try:

        # Calculates accuracry, precision, recall and f1 scores.
        print(f"Accuracy: {accuracy_score(true_labels, predicted_labels)}")
        print(f"Precision: {precision_score(true_labels, predicted_labels, average='micro')}")
        print(f"Recall: {recall_score(true_labels, predicted_labels, average='micro')}")
        print(f"F1 Score: {f1_score(true_labels, predicted_labels, average='micro')}")

        # Performs classification report
        print("Classification report : ")
        print(classification_report(true_labels, predicted_labels, target_names=label_names))

        # Generates confusion matrix
        matrix = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(10, 7), dpi=200)
        ConfusionMatrixDisplay(matrix, display_labels=label_names).plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
        plt.title("Confusion Matrix for EfficientNet")
        plt.savefig('./figures/Confusion_Matrix_EfficientNet.png', bbox_inches = 'tight')

    except Exception as e:
        print(f"Evaluating the EfficientNet model has failed. Error: {e}")


def class_imbalance_handling(train_labels):
    """Handling class imbalance

    This function performs classes weights, which is useful for the model to "pay more attention" to samples from an under-represented class.

    Args:
            training dataset
    
    Returns:

            Class Weights

    """

    try:

        # Computing class weights
        train_labels = np.array(train_labels)
        disease_class_weights = class_weight.compute_class_weight('balanced',
                                                            classes = np.unique(train_labels),
                                                            y = train_labels)

        # Link each weight to it's corresponding class.
        weights = {0 : disease_class_weights[0], 
                1 : disease_class_weights[1], 
                2 : disease_class_weights[2], 
                3 : disease_class_weights[3], 
                4 : disease_class_weights[4]
                }

        print(f"Class weights for imbalance {weights}")
        return weights
    
    except Exception as e:
        print(f"Class imbalance handling has failed. Error: {e}")


def EfficientNet_model_training(train_dataset, train_labels, val_dataset, val_labels):
    """EfficientNet model training

    This function trains the EfficientNet model and tests it on the dataset. Then, it will evaluate it and produce the accuracy and plot loss.

    Args:
            training and validation datasets and labels
    

    """

    try:
        

        # Categorise the labels into 5 classes
        train_labels_categorical = to_categorical(train_labels, num_classes=5)
        val_labels_categorical = to_categorical(val_labels, num_classes=5)

        # EfficientNet model
        base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224, 224, 3))

        # Freeze the base model (for transfer learning)
        base_model.trainable = False

        
        model = Sequential()
        model.add(Input(shape=(224,224,3)))
        model.add(base_model)
        model.add(GlobalAveragePooling2D())
        model.add(BatchNormalization())
        model.add(Flatten())
        model.add(Dense(1024, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(512, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(5, activation="softmax"))

        # Output the model summary
        print(model.summary())

        # Check if there is any GPU available
        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
                print("✅ GPU is available:", gpus)
        else:
                print("❌ No GPU found. TensorFlow is running on CPU.")

        print(tf.config.experimental.list_physical_devices('GPU'))

        # Plot the EfficientNet model
        plot_model(model, 
                to_file='./figures/EfficientNet_Model_test_47.png', 
                show_shapes=True,
                show_layer_activations=True)

        # Compile the EfficientNet model
        model.compile(loss='categorical_crossentropy',
                optimizer=Adam(0.00001),
                metrics=['accuracy'])
        
        # Handle the class imbalance.
        weights = class_imbalance_handling(train_labels)

        # Fit the EfficientNet model
        history = model.fit(train_dataset, train_labels_categorical, 
                epochs=100,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=8)],
                validation_data=(val_dataset, val_labels_categorical),
                batch_size=32,
                shuffle=True,
                class_weight=weights)
        
        # save the EfficientNet model
        utils.save_model(model, "EfficientNet_Model_test_add_54")

        # plot the accuracy and loss graphs for the EfficientNet model
        utils.plot_accuray_loss(history)

    except Exception as e:
        print(f"Training and saving the EfficientNet model failed. Error: {e}")


def EfficientNet_model_testing(test_dataset, test_lables):
    """EfficientNet model testing

    This function loads the final EfficientNet model and tests it on the test dataset. Then, it will evaluate it and produce the accuracy and plot loss.

    Args:
             test datasets.
    

    """

    try:
            
        # Class labels
        class_labels = ["Cassava Bacterial Blight (CBB)",
                "Cassava Brown Streak Disease (CBSD)",
                "Cassava Green Mottle (CGM)",
                "Cassava Mosaic Disease (CMD)",
                "Healthy"]

        # Load the CNN model
        model = utils.load_model("EfficientNet_Model_test_add_52")

        # Output the model summary
        print(model.summary())

        # Evaluate the model
        test_dataset_prob = model.predict(test_dataset, verbose=0)
        test_predict_labels = np.argmax(test_dataset_prob, axis=-1)
        evaluate_model(test_lables, test_predict_labels, test_dataset_prob, class_labels)

    except Exception as e:
        print(f"Loading and testing the EfficientNet model failed. Error: {e}")
