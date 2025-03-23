"""Accomplishing Task B via Convolutional Neural Networks.

    This module acquires BlooddMNIST data from medmnist library, then it uses the CNN model to accuractly predict the 8 different
    classes of the blood diseases.

    """

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
from keras.utils import to_categorical, plot_model
import tensorflow as tf
from keras.models import Sequential, Model
from keras.layers import GlobalAveragePooling2D, Dropout, Dense, RandomFlip, RandomRotation, RandomZoom, Flatten, BatchNormalization
from keras.optimizers import Adam
from keras.optimizers.schedules import ExponentialDecay
from keras.applications import EfficientNetB0, EfficientNetB3
from keras import regularizers
from keras import Input
from sklearn.metrics import confusion_matrix, classification_report,accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.utils import class_weight
from src import utils
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def image_augmentation():
    """Pre-process check.

    This function checks if the datasets has no missing images.

    Args:
            training, validation and test datasets.

    Returns:
            N/A

    """

    # Convert numpy array to a Tensor
#     x_train_tensor = tf.convert_to_tensor(train_dataset)

    # Apply augmentation
    data_augmentation = Sequential([

        RandomFlip("horizontal"),
        RandomRotation(0.2),
        RandomZoom(0.2),

    ])

#     augmented_train_dataset = data_augmentation(x_train_tensor, training=True)
    return data_augmentation



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

        if(true_labels.ndim==2):
            true_labels = true_labels[:,0]
        if(predicted_labels.ndim==2):
            predicted_labels=predicted_labels[:,0]
        if(predict_probs.ndim==2):
            predict_probs=predict_probs[:,0]

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
        plt.savefig('/figures/Confusion_Matrix_EfficientNet.png', bbox_inches = 'tight')

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
    """CNN model training

    This function trains the CNN model and tests it on the dataset. Then, it will evaluate it and produce the accuracy and plot loss.

    Args:
            training and validation datasets.
    

    """

    try:
        

        # Categorise the labels into 5 classes
        train_labels_categorical = to_categorical(train_labels, num_classes=5)
        val_labels_categorical = to_categorical(val_labels, num_classes=5)

        # CNN model

        #base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(128,128,3))
        #model = base_model.output
        # model = image_augmentation()(model)
        # model = GlobalAveragePooling2D()(model)
        # model = Dropout(0.25)(model)
        # model = Dense(128, activation="relu")(model)
        # model = Dropout(0.5)(model)
        # output_layer = Dense(5, activation="softmax")(model)

        # model = Model(inputs=base_model.input, outputs=output_layer)

        model = Sequential()
        model.add(EfficientNetB0(weights="imagenet", include_top=False, input_shape=(224,224,3)))
        model.add(image_augmentation())
        model.add(GlobalAveragePooling2D())
        #model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(1024,activation='relu', bias_regularizer=regularizers.L2(1e-4), kernel_regularizer=regularizers.L1(1e-4)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(512, activation="relu", bias_regularizer=regularizers.L2(1e-4), kernel_regularizer=regularizers.L1(1e-4)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(256,activation='relu', bias_regularizer=regularizers.L2(1e-4), kernel_regularizer=regularizers.L1(1e-4)))
        model.add(BatchNormalization())
        model.add(Dropout(0.3))
        model.add(Dense(128, activation="relu", bias_regularizer=regularizers.L2(1e-4), kernel_regularizer=regularizers.L1(1e-4)))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        #model.add(Dropout(0.5))
        model.add(Dense(5, activation="softmax"))
        # Output the model summary
        print(model.summary())


        gpus = tf.config.list_physical_devices('GPU')

        if gpus:
                print("✅ GPU is available:", gpus)
        else:
                print("❌ No GPU found. TensorFlow is running on CPU.")

        print(tf.config.experimental.list_physical_devices('GPU'))

        # Plot the CNN model
        plot_model(model, 
                to_file='./figures/EfficientNet_Model_test_29.png', 
                show_shapes=True,
                show_layer_activations=True)

        # Compile the CNN model
        #lr_schedule = ExponentialDecay(5e-3, decay_steps=10000, decay_rate=0.9)
        model.compile(loss='categorical_crossentropy',
                optimizer=Adam(0.0001),
                metrics=['accuracy'])
        
        # Handle the class imbalance.
        weights = class_imbalance_handling(train_labels)

        # Fit the CNN model
        history = model.fit(train_dataset, train_labels_categorical, 
                epochs=20,
                callbacks=[tf.keras.callbacks.EarlyStopping(patience=8)],
                validation_data=(val_dataset, val_labels_categorical),
                batch_size=32,
                shuffle=True,
                class_weight=weights)
        
        # save the CNN model
        utils.save_model(model, "EfficientNet_Model_test_add_29")

        utils.plot_accuray_loss(history)

    except Exception as e:
        print(f"Training and saving the EfficientNet model failed. Error: {e}")


def EfficientNet_model_testing(test_dataset):
    """CNN model testing

    This function loads the final CNN model and tests it on the test dataset. Then, it will evaluate it and produce the accuracy and plot loss.

    Args:
            training, validation and test datasets.
    

    """

    try:
            
        # Class labels
        class_labels = ["Cassava Bacterial Blight (CBB)",
                "Cassava Brown Streak Disease (CBSD)",
                "Cassava Green Mottle (CGM)",
                "Cassava Mosaic Disease (CMD)",
                "Healthy"]

        # Load the CNN model
        model = utils.load_model("CNN_model_taskB_final")

        # Output the model summary
        print(model.summary())

        # Evaluate the model
        test_dataset_prob = model.predict(test_dataset.imgs, verbose=0)
        test_predict_labels = np.argmax(test_dataset_prob, axis=-1)
        evaluate_model(test_dataset.labels, test_predict_labels, test_dataset_prob, class_labels)

    except Exception as e:
        print(f"Loading and testing the EfficientNet model failed. Error: {e}")