#!/usr/bin/env python
"""
This script stores functions for image classification with sklearn. The code was developed for use in class and has been adopted for this project.
"""

# Dependencies
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Function for plotting confusion matrix
def plot_cm(y_test, y_pred, normalized:bool):
    """
    Plot confusion matrix
    """
    if normalized == False:
        cm = pd.crosstab(y_test, y_pred, 
                            rownames=['Actual'], colnames=['Predicted'])
        p = plt.figure(figsize=(10,10));
        p = sns.heatmap(cm, annot=True, fmt="d", cbar=False)
    elif normalized == True:
        cm = pd.crosstab(y_test, y_pred, 
                               rownames=['Actual'], colnames=['Predicted'], normalize='index')
        p = plt.figure(figsize=(10,10));
        p = sns.heatmap(cm, annot=True, fmt=".2f", cbar=False)
        
        
# Function for predicting an unseen image        
def predict_unseen(image, model, classes):
    """
    Predict the category of unseen data, show probabilities 
    
    image: unseen data
    model: trained model
    classes: list of possible classes
    """
    # Reshape array
    test_probs = model.predict_proba(image.reshape(1,784))
    # plot prediction
    sns.barplot(x=classes, y=test_probs.squeeze());
    plt.ylabel("Probability");
    plt.xlabel("Class")
    
    #predictied label
    idx_cls = np.argmax(test_probs)
    print(f"I think that this is class {classes[idx_cls]}")
    
    return None

    
if __name__=="__main__":
    pass