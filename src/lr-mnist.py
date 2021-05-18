#!/usr/bin/env python
"""
Info: Given the full MNIST dataset, this script trains a Logistic Regression Classifier, prints evaluation metrics to the terminal and saves the metrics in output folder.

Parameters:
    (optional) output_filename: str <name-of-output-file>, default = "lr_classification_metrics.txt"
    (optional) test_size: float <size-of-test-data>, default = 0.2
    (optional) unseen_img: str <path-to-unseen-image>, default = None
    
Usage:
    $ python lr-mnist.py

Output:
    - lr_classification_metrics.txt: Logistic regression classification metrics.
    - confusion_matrix.png: Normalized confusion matrix. This provides an overview of how well the classifier performed. 
"""

### DEPENDENCIES ###

# core libraries
import os
import sys
sys.path.append(os.path.join(".."))

# numpy, openCV, and matplotlib, pandas
import numpy as np
import cv2
import matplotlib.pyplot as plt
import pandas as pd

# classifier utils
import utils.classifier_utils as clf_util

# sci-kit learn
from sklearn import metrics
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

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
                    help = "Name of the output file that will contain the logistic regression classification metrics",
                    default = "lr_classification_metrics.txt") # default output filename

    # Argument 2: Size of test dataset
    ap.add_argument("-ts", "--test_size", 
                    type = float,
                    required = False, # not required argument
                    help = "Define the size of the test dataset", 
                    default = 0.2) # default size of test-split (i.e. 20% of the data goes to testing)
    
    # Argument 4: Unseen image
    ap.add_argument("-u", "--unseen_img", 
                    type = str,
                    required = False, # not required argument
                    help = "Specify path to an unseen image that you want the trained logistic regression model to predict the value of",
                    default = None) # If the user does not specify a path to an unseen image, this argument will be None

    # Parse arguments
    args = vars(ap.parse_args())
    
    # Save input parameters
    output_filename = args["output_filename"]
    test_size = args["test_size"]
    unseen_image = args["unseen_img"]
    
    # Create output directory if it does not already exist
    if not os.path.exists(os.path.join("..", "output")):
        os.mkdir(os.path.join("..", "output"))
    
    # Load data
    print("\n[INFO] Fetching the full MNIST dataset...")
    X, y = fetch_openml("mnist_784", version=1, return_X_y=True) # X = images, y = labels (adhering to the standard way of labelling data and labels in machine learning)
    
    # Start message to user
    print("\n[INFO] Initializing logistic regression classification...")
    
    # Instantiating the Lr_classifier class
    lr_classifier = Lr_classifier(X, y)
    
    # Preprocess data
    print("\n[INFO] Preprocessing the MNIST dataset...")
    X_train_scaled, X_test_scaled, y_train, y_test = lr_classifier.preprocess(test_size)
    
    # Perform grid search to estimate the best hyperparameters
    print("\n[INFO] Peforming grid search to estimate the best hyperparameters for the logistic regression classifier...")
    best_params = lr_classifier.perform_gridsearch(X_test_scaled, y_test)
    
    # Train the logistic regression classifier
    print("\n[INFO] Training the logistic regression classifier...")
    clf = lr_classifier.train_lr_classifier(X_train_scaled, y_train, best_params)
    
    # Evaluate the logistic regression classifier
    print("\n[INFO] Evaluating the logistic regression classifier...")
    lr_classification_metrics, y_predictions = lr_classifier.evaluate_lr_classifier(clf, X_test_scaled, y_test)
    
    # Save outputs to output directory
    print(f"\n[INFO] Saving classification metrics to output/ directory as {output_filename}...")
    lr_classifier.save_output(output_filename, lr_classification_metrics, y_predictions, y_test, clf)
    
    # Print classification metrics to terminal
    print(f"\n[INFO] Below are the classification metrics. These can also be found as {output_filename} in the output directory\n")
    print(lr_classification_metrics)
    
    # Test the model if the user has specified an unseen image
    lr_classifier.test_lr_classifier(clf, unseen_image)
    
    # User message
    print("\n[INFO] Done! You have now performed the logistic regression classification. Classification metrics have been saved in output directory.\n")
    
# Creating Logistic regression classifier class
class Lr_classifier:
    
    # Initialize class
    def __init__(self, X, y):
        
        # Define input: Image and labels 
        self.X = X # images
        self.y = y # labels
        
        
    def preprocess(self, test_size):
        """
        This method preprocesses the MNIST data. The preprocessing includes converting the data into numpy arrays, creating the test-train split, and performing min-max regularization.
        """
        # Convert images (X) and labels (y) to numpy arrays
        X_arrays = np.array(self.X)
        y_arrays = np.array(self.y)
        
        # Create train and test split
        X_train, X_test, y_train, y_test = train_test_split(X_arrays, # images
                                                            y_arrays, # labels
                                                            random_state=9,
                                                            test_size=test_size) # specified by the user or default 0.2
        
        # Min-Max regularization - scaling the training and testing images. Min-max regularization is a smart way to normalize the data instead of dividing everything by 255. Min-max regularization normalizes each data point, which gives us a more compressed and regular dataset to work with. 
        X_train_scaled = (X_train - X_train.min())/(X_train.max() - X_train.min()).astype("float")
        X_test_scaled = (X_test - X_test.min())/(X_test.max() - X_test.min()).astype("float")
        
        return X_train_scaled, X_test_scaled, y_train, y_test 
    
    
    def perform_gridsearch(self, X_train_scaled, y_train):
        """
        This method performs grid search, i.e. iterates over possible hyperparameters for the logistic regression model in order to find the most optimal values. The hyperparameters that I have chosen to iterate over are penalty, C, and solver algorithm. 
        """
        # Initialize pipeline consisting of the "classifier" which which is made up of the logistic regression classification function
        pipe = Pipeline([('classifier', LogisticRegression())])

        # Set tunable parameters for grid search
        penalties = ['l1', 'l2']         # regularization parameters
        C = [1.0, 0.1, 0.01]             # regularization strengths
        solvers = ['liblinear', 'saga']  # solvers
        tol = [0.1, 0.01, 0.001]         # tolerance values

        # Create parameter grid (a Python dictionary) that contains the hyperparameters 
        parameters = dict(classifier__penalty = penalties,
                          classifier__C = C,
                          classifier__solver = solvers,
                          classifier__tol = tol)

        # Choose which metrics on which we want to optimise
        scores = ['precision', 'recall', 'f1']
        
        # For each of the metrics find the optimal hyperparameter values
        for score in scores:
            
            # Initialise Gridsearch with predefined parameters
            clf = GridSearchCV(pipe, 
                               parameters, 
                               scoring= f"{score}_weighted",
                               cv=10) # using 10-fold cross-validation
            
            # Fit grid search model to data
            clf.fit(X_train_scaled, y_train)

            # Print the best paremeters to terminal 
            print(f"Best parameters found on training data for {score}: \n")
            print(clf.best_params_)
            
            # Save best parameters
            best_params = clf.best_params_
            
        return best_params
            
    
    def train_lr_classifier(self, X_train_scaled, y_train, best_params):
        """
        This method trains the logistic regression classifier on the scaled training data with hyperparameters estimated by the grid search. 
        """
        # Train the logistic regression classifier on the scaled data
        clf = LogisticRegression(C=best_params['classifier__C'], # taking the most optimal regularization strength as estimated by grid search
                                 penalty=best_params['classifier__penalty'], # taking the best penalty method estimated by grid search
                                 tol=best_params['classifier__tol'], # taking the best tolerance value estimated by grid search
                                 solver=best_params['classifier__solver'], # algorithm to use for optimization. The saga algorithm is fast and well-suited for large datasets. We take the best solver method estimated by grid search
                                 multi_class='multinomial').fit(X_train_scaled, y_train) # when using 'multinomial' the loss minimized is the multinomial loss fit across the entire probability distribution
        
        return clf
    
    
    def evaluate_lr_classifier(self, clf, X_test_scaled, y_test):
        """
        This method evaluates the logistic regression classifier on the validation data. 
        """
        # Extract predictions
        y_predictions = clf.predict(X_test_scaled)
    
        # Compute the classification metrics
        lr_classification_metrics = metrics.classification_report(y_test, y_predictions)
        
        return lr_classification_metrics, y_predictions
    
    
    def save_output(self, output_filename, lr_classification_metrics, y_predictions, y_test, clf):
        """
        This method saves the classification metrics for the logistic regression classifier as a txt-file in the output directory.
        """
        # Define the output path
        out_path = os.path.join("..", "output", output_filename)
        
        # Save classification metrics to output directory
        with open(out_path, "w") as output_file:
            output_file.write(f"Below are the classification metrics for the logistic regression classifier: {clf} \n \n{lr_classification_metrics}")
      
        # Create confusion matrix to visualize performance of the model
        confusion_matrix = clf_util.plot_cm(y_test, y_predictions, normalized=True)
        
        # Save confusion matrix as .png to output folder
        plt.savefig(os.path.join("..", "output", "lr_confusion_matrix.png"), dpi = 300, bbox_inches = "tight")
        
        
    def test_lr_classifier(self, clf, unseen_image):
        """
        This method takes an unseen image provided by the user and tests if the logistic regression classifier is able to classify it correctly. 
        """
        # If the user has specified an unseen image
        if unseen_image != None:
            
            # Read image
            image = cv2.imread(unseen_image)
            
            # Define classes
            classes = sorted(set(self.y))
            
            # Convert image to greyscale
            grey_img = cv2.bitwise_not(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            
            # Compress image to 28x28 which is the same dimensions as the images in the MNIST dataset
            compressed = cv2.resize(grey_img, # which image to resize
                                    (28, 28), # dimensions to resize to
                                    interpolation=cv2.INTER_AREA) # resizing method. INTER_AREA is a resizing technique that resamples using pixel area relation - this prevents distorting the images when they are resized.
           
            # Extract the model's predictions for the unseen image
            print(f"\n[INFO] Prediction for unseen image, {unseen_image}:\n")
            clf_util.predict_unseen(compressed, clf, classes)
            

# Define behaviour when called from command line
if __name__=="__main__":
    main()