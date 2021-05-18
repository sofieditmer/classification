#!/usr/bin/env python
"""
Info: Given the full MNIST data set, this script trains a neural network classifier, prints the evaluation metrics to the terminal  and saves the metrics in the output directory.

Parameters:
    (optional) output_filename: str <name-of-output-file>, default = "nn_classification_metrics.txt"
    (optional) test_size: float <size-of-test-data>, default = 0.2
    (optional) batch_size: int <size-of-batches>, default = 32
    (optional) n_epochs: int <number-of-epochs>, default = 20
    (optional) grid_search: str <perform-grid-search-true-or-false>, default = "False"
    
Usage:
    $ python nn-mnist.py

Output:
    - nn_model_summary.txt: a summary of the model architecture.
    - nn_model_architecture.png: a visual overview of the model architecture.
    - nn_model_loss_accuracy_history.png: a visual representation of the loss/accuracy performance of the model during training. 
    - nn_classification_metrics.txt: Neural network classification metrics.
"""

### DEPENDENCIES ###

# core libraries
import os
import sys
sys.path.append(os.path.join(".."))
from contextlib import redirect_stdout

# numpy
import numpy as np

# matplotlib
import matplotlib.pyplot as plt

# classifier utils
import utils.classifier_utils as clf_util

# sklearn
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn.datasets import fetch_openml
from sklearn.model_selection import GridSearchCV

# tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

# argparse
import argparse

### MAIN FUNCTION ###

def main():
    
    # Initialise ArgumentParser class
    ap = argparse.ArgumentParser()
    
    # Argument 1: Output filename
    ap.add_argument("-o", "--output_filename",
                    type = str,
                    required = False, # not required argument
                    help = "Name of the output file",
                    default = "nn_classification_metrics.txt") # default output filename
    
    # Argument 1: size of the test data
    ap.add_argument("-ts", "--test_size", 
                    type = float,
                    required = False, # not required argument
                    help = "Define the size of the test data", 
                    default = 0.2) # default test size
    
    ap.add_argument("-g", "--grid_search",
                    type = str,
                    required = False, # not required argument
                    help = "Specify whether you want to perform grid search or not",
                    default = "False") # default = do not perform grid search
    
    ap.add_argument("-n", "--n_epochs",
                    type = int,
                    required = False, # not required argument
                    help = "Define how many epochs you wish to train the model for",
                    default = 20) # default number of epochs
    
    ap.add_argument("-b", "--batch_size",
                    type = int,
                    required = False, # not required argument
                    help = "Define how many epochs you wish to train the model for",
                    default = 32) # default batch size
    
    
    # Parse arguments
    args = vars(ap.parse_args())
    
    # Save input parameters
    output_filename = args["output_filename"]
    test_size = args["test_size"]
    grid_search = args["grid_search"]
    n_epochs = args["n_epochs"]
    batch_size = args["batch_size"]
    
    # Create output directory
    if not os.path.exists(os.path.join("..", "output")):
        os.mkdir(os.path.join("..", "output"))
        
    # Start message to user
    print("\n[INFO] Initializing neural network classification...")
    
    # Load data
    print("\n[INFO] Fetching MNIST dataset...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True)
    
    # Instantiating neural network classifier class
    nn_classifier = Nn_classifier(X, y)
    
    # Preprocess data
    print("\n[INFO] Preprocessing the data...")
    X_train_scaled, X_test_scaled, y_train_binarized, y_test_binarized, X_arrays, y_arrays = nn_classifier.preprocess(test_size)
    
    # Perform grid search if the user has specified to do so
    best_params = nn_classifier.perform_gridsearch(X_arrays, y_arrays, grid_search)

    # Define neural network model
    print("\n[INFO] Defining neural network model architecture...")
    model = nn_classifier.define_model(best_params, grid_search)
    
    # Train neural network
    print("\n[INFO] Training the neural network on the training data...")
    history = nn_classifier.train_neural_network(X_train_scaled, X_test_scaled, y_train_binarized, y_test_binarized, model, best_params, grid_search, n_epochs, batch_size)
    
    # Visualize loss/accuracy during training of model
    print("\n[INFO] Computing plot of loss/accuracy of model during training and saving to /output directory...")
    nn_classifier.visualize_loss_accuracy(history, best_params, grid_search, n_epochs)
    
    # Evaluate neural network classifier
    print("\n[INFO] Evaluating the neural network...")
    nn_classification_metrics, predictions = nn_classifier.evaluate_neural_network(model, y_test_binarized, X_test_scaled, batch_size)
    
    # Save classification metrics to output directory
    print(f"\n[INFO] Saving classificaiton metrics as {output_filename} to the output directory...")
    nn_classifier.save_classification_report(nn_classification_metrics, output_filename, model, predictions, y_test_binarized)
    
    # Print classification metrics to terminal
    print(f"\n[INFO] Below are the classification metrics. These can also be found as {output_filename} in the output directory...")
    print(nn_classification_metrics)
    
    # Message to user
    print("\n[INFO] Done! You have now performed the neural network classification. The classification metrics can be found in the output directory.\n")
    
# Creating Neural network classifier class   
class Nn_classifier:
    
    def __init__(self, X, y):
        
        # Receive inputs: Image and labels 
        self.X = X
        self.y = y
        
        
    def preprocess(self, test_size):
        """
        This method preprocesses the MNIST data. The preprocessing includes converting the data into numpy arrays and floats, creating the test-train split, performing min-max regularization, and binarizing the image labels. 
        """
        # Convert data into numpy arrays
        X_arrays = np.array(self.X)
        y_arrays = np.array(self.y)
        
        # Create train and test split
        X_train, X_test, y_train, y_test = train_test_split(X_arrays,
                                                            y_arrays,
                                                            random_state=9,
                                                            test_size=test_size) # specified by the user or default 0.2
        
        # Perform min-max regularization
        X_train_scaled = (X_train - X_train.min())/(X_train.max() - X_train.min()).astype("float")
        X_test_scaled = (X_test - X_test.min())/(X_test.max() - X_test.min()).astype("float")
        
        # Binarize training and test labels
        lb = LabelBinarizer() # initialize binarizer
        y_train_binarized = lb.fit_transform(y_train)
        y_test_binarized = lb.fit_transform(y_test)
    
        return X_train_scaled, X_test_scaled, y_train_binarized, y_test_binarized, X_arrays, y_arrays

    
    def perform_gridsearch(self, X_arrays, y_arrays, grid_search):
        """
        This method performs grid search to estimate the most optimal hyperparameters for the neural network model if the user has specified that they wish to perform grid search. 
        """
        # If the user has indiciated that they wish to perform grid search
        if grid_search == "True":
            
            # User message
            print("\n[INFO] Performing grid search to estimate the best hyperparameters...")
            
            # First we need to define the model. In order to make keras work with scikit-learn, we need to define the model as a function. When we call the function it initializes the model and returns it. 
            def nn_model(optimizer='adam'):
                
                # Initialize a sequential model
                model = Sequential()
                
                # Add input layer to the model which corresponds to the size of the data (28 x 28 images = 784 nodes), and hidden layer with 256 nodes. This layer is "dense" which refers to a fully-connected layer.
                model.add(Dense(256, 
                                input_shape=(784, ), 
                                activation="relu"))
                
                # Add another hidden layer with 128 nodes, and another ReLU activation layer
                model.add(Dense(128, 
                                activation="relu"))
                
                # Add classificaiton layer, 10 classes with softmax activation layer
                model.add(Dense(10, 
                                activation="softmax")) 
                
                # categorical cross-entropy, optimizer defined in function call
                model.compile(loss="categorical_crossentropy", 
                              optimizer=optimizer, # default is adam
                              metrics=["accuracy"])
            
            # return the compiled model
            return model
        
            # Create the model
            model = nn_model()
        
            # Take the defined model and run it through the KerasClassifier. This returns an object that can be used in the sklearn pipeline
            model = KerasClassifier(build_fn=model, # build the model
                                    verbose=0) # we do not want outpur during training
        
            # Define grid search parameters
            optimizers = ['sgd', 'adam']    # optimizer algorithms
            epochs = [5, 10, 15]            # range of epochs to run
            batches = [5, 10]               # batch sizes

            # Create parameter search grid (a Python dictionary) that contains the hyperparameters to iterate over
            param_grid = dict(optimizer=optimizers, 
                              epochs=epochs, 
                              batch_size=batches)
        
            # Run the grid search
            grid = GridSearchCV(estimator=model, 
                                param_grid=param_grid, 
                                n_jobs=-1, # number of CPU cores to use. -1 means use all available cores
                                cv=5, # using 5-fold cross validation
                                scoring='accuracy')
        
            # Fit grid search to *all* data and labels. NB! Cross validation is making random splits. 
            grid_result = grid.fit(X_arrays, y_arrays)
        
            # Print the best results, rounding values to 3 decimal places
            print(f"Best run: {round(grid_result.best_score_, 3)} using {grid_result.best_params_}")
        
            # Save best results to use to train the model
            best_params = grid_result.best_params_
        
            return best_params
        
        # If the user has not specified that they want to perform grid search 
        if grid_search == "False":
            return None
    
         
    def define_model(self, best_params, grid_search):
        """
        This method defines the neural network model architecture using tensorflow.keras. The model architecture is either defined with the estimated hyperparameters from the grid search or with default parameters. 
        """
        # If grid search has been performed 
        if grid_search == "True":
            
            # Initialize a sequential model
            model = Sequential()
        
            # Add input layer to the model which corresponds to the size of the data (28 x 28 images = 784 nodes), and hidden layer with 256 nodes. This layer is "dense" which refers to a fully-connected layer.
            model.add(Dense(256, 
                            input_shape=(784, ), 
                            activation="relu"))
    
            # Add another hidden layer with 128 nodes, and another ReLU activation layer
            model.add(Dense(128, 
                            activation="relu"))
        
            # Add classificaiton layer of 10 classes with softmax as activation function
            model.add(Dense(10, 
                        activation="softmax")) 
        
            # Compile model using categorical cross-entropy
            model.compile(loss="categorical_crossentropy", 
                      optimizer=best_params['optimizer'], # taking the most optimal optimizer as estimated by grid search
                      metrics=["accuracy"])
        
            # Model summary
            model_summary = model.summary()
        
            # Save model summary to output directory
            output_path_summary = os.path.join("..", "output", "nn_model_summary.txt")
        
            with open(output_path_summary, 'w') as f:
                with redirect_stdout(f):
                    model.summary()
    
            # Visualize model architecture and save to output directory
            output_path_model = os.path.join("..", "output", "nn_model_architecture.png")  
            plot = plot_model(model,
                              to_file = output_path_model,
                              show_shapes=True,
                              show_layer_names=True)

            return model
        
        
        # If grid search was not performed
        if grid_search == "False":
            
            # Initialize a sequential model and use standard hyperparameters
            model = Sequential()
        
            # Add input layer to the model which corresponds to the size of the data (28 x 28 images = 784 nodes), and hidden layer with 256 nodes. This layer is "dense" which refers to a fully-connected layer.
            model.add(Dense(256,
                            input_shape=(784,), 
                            activation="relu"))
            
            # Add a second hidden layer of 128 nodes, and another activation layer
            model.add(Dense(128, 
                            activation="relu"))
        
            # Add classificaiton layer of 10 classes with softmax as activation function
            model.add(Dense(10,
                            activation="softmax")) 
        
            # Compile model using categorical cross-entropy
            model.compile(loss="categorical_crossentropy", 
                      optimizer="adam", # using adam as the default optimizer
                      metrics=["accuracy"])
        
            # Save model summary to output directory
            output_path_summary = os.path.join("..", "output", "nn_model_summary.txt")
        
            with open(output_path_summary, 'w') as f:
                with redirect_stdout(f):
                    model.summary() # model summary
    
            # Visualize model architecture and save to output directory
            output_path_model = os.path.join("..", "output", "nn_model_architecture.png")  
            plot = plot_model(model,
                              to_file = output_path_model,
                              show_shapes=True,
                              show_layer_names=True)

            return model
        
        
    def train_neural_network(self, X_train_scaled, X_test_scaled, y_train_binarized, y_test_binarized, model, best_params, grid_search, n_epochs, batch_size):
        """
        This method trains the neural network on the training data either using the best hyperparameters estimated by the grid search or using default hyperparameters depending on what the user has specified. 
        """
        # If the user has chosen to perform grid search
        if grid_search == "True":
            
            # Fit model to training data
            model_history = model.fit(X_train_scaled, y_train_binarized, 
                                      validation_data=(X_test_scaled, y_test_binarized), 
                                      epochs=best_params['epochs'], # taking the best optimizer as estimated by the grid search

                                      batch_size=best_params['batch_size'], # taking the most optimal number of batches as estimated by grid search
                                      verbose=1) # show progress bars 
            
            return model_history
    
        # If grid search has not been performed
        if grid_search == "False":
            
            # Fit model to training data
            model_history = model.fit(X_train_scaled, y_train_binarized, 
                                      batch_size = batch_size, # using 32 as the default batch size
                                      validation_data=(X_test_scaled, y_test_binarized), 
                                      epochs=n_epochs, # using 20 epochs as the default number
                                      verbose=1) # show progress bars
            
            return model_history
            

    def visualize_loss_accuracy(self, model_history, best_params, grid_search, n_epochs):
        """
        This method visualizes the loss and accuracy of the model during training using matplotlib. This method was developed for use in class and was adopted for this assignment. 
        """ 
        # If the user has performed grid search the visualization should adapt to this (the number of epochs showed in the plot will depend on the grid search results)
        if grid_search == "True":
            
            # Visualize model performance
            plt.style.use("fivethirtyeight")
            plt.figure()
            plt.plot(np.arange(0, best_params['epochs']), model_history.history["loss"], label="train_loss")
            plt.plot(np.arange(0, best_params['epochs']), model_history.history["val_loss"], label="val_loss")
            plt.plot(np.arange(0, best_params['epochs']), model_history.history["accuracy"], label="train_acc")
            plt.plot(np.arange(0, best_params['epochs']), model_history.history["val_accuracy"], label="val_acc")
            plt.title("Loss and Accuracy")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join("..", "output", "nn_model_loss_accuracy_history.png"))
            
        # If the user has not performed grid search 
        if grid_search == "False":
            plt.style.use("fivethirtyeight")
            plt.figure()
            plt.plot(np.arange(0, n_epochs), model_history.history["loss"], label="train_loss")
            plt.plot(np.arange(0, n_epochs), model_history.history["val_loss"], label="val_loss")
            plt.plot(np.arange(0, n_epochs), model_history.history["accuracy"], label="train_acc")
            plt.plot(np.arange(0, n_epochs), model_history.history["val_accuracy"], label="val_acc")
            plt.title("Loss and Accuracy")
            plt.xlabel("Epoch #")
            plt.ylabel("Loss/Accuracy")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join("..", "output", "nn_model_loss_accuracy_history.png"))
            
            
    def evaluate_neural_network(self, model, y_test_binarized, X_test_scaled, batch_size):
        """
        This method evaluates the performance of the neural network classifier.
        """
        # Compute predictions
        predictions = model.predict(X_test_scaled, batch_size=batch_size)
        
        # Create label names
        label_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        
        # Classification report
        nn_classification_metrics = classification_report(y_test_binarized.argmax(axis=1),
                                                          predictions.argmax(axis=1),
                                                          target_names=label_names)
        
        # Print classification report to terminal
        print(nn_classification_metrics)
        
        return nn_classification_metrics, predictions
    
    
    def save_classification_report(self, nn_classification_metrics, output_filename, model, predictions, y_test_binarized):
        """
        This method saves the classification metrics of the neural network to the output directory. 
        """
        # Define output path
        out_path = os.path.join("..", "output", output_filename)
        
        # Save classification metrics to output directory
        with open(out_path, "w") as f:
            f.write(f"Below are the classification metrics for the neural network classifier:\n \n{nn_classification_metrics}")
            
# Define behaviour when called from command line
if __name__=="__main__":
    main()