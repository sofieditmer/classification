# Assignment 4: Classification Benchmarks

### Description of task: Classifier benchmarks using Logistic Regression and a Neural Network <br>
This assignment was assigned by the course instructor as “Assignment 4 – Classification Benchmarks”. The purpose of this assignment was to demonstrate our knowledge of how to train classification models on the MNIST dataset using machine learning and neural networks that can be used as statistical benchmarks. The MNIST is a large dataset which consists of 70,000 handwritten digits and 10 unique classes and is commonly used for image processing and machine learning purposes. Two separate python scripts were to be created; ```lr-mnist.py``` that should take the full MNIST dataset, train a logistic regression classifier, print the evaluation metrics to the terminal and save them to an output directory, and ```nn-mnist.py``` that should take the full MNIST dataset, train a neural network classifier, print the evaluation metrics to the terminal, and save them in an output directory. 
Apart from what was directly specified in the assignment description, I also chose to implement a method that performs grid search to estimate the most optimal hyperparameters to use when training the models.  Furthermore, I implemented a method that takes an unseen image, processes it, and uses the logistic classifier to predict its class. 

### Content and Repository Structure <br>

The repository follows the overall structure below. The python scripts, ```lr-mnist.py``` and ```nn-mnist.py```, are located in the ```src``` folder. The outputs produced when running the scripts can be found within the ```output``` folder. In the ```utils``` folder a utility script for classification is stored.

| Folder | Description|
|--------|:-----------|
| ```data``` | A folder containing an example of an unseen image for the logistic classifier to predict if one wishes to test it. 
| ```src``` | A folder containing the python scripts for the particular assignment.
| ```output``` | A folder containing the outputs produced when running the python scripts within the src folder.
| ```utils``` | A folder containing utility scripts that store functions that are used within the main python script.
| ```requirements.txt```| A file containing the dependencies necessary to run the python script.
| ```create_edgedetection_venv.sh```| A bash-file that creates a virtual environment in which the necessary dependencies listed in the ```requirements.txt``` are installed. This script should be run from the command line.
| ```LICENSE``` | A file declaring the license type of the repository.


### Usage and Technicalities <br>
If the user wishes to engage with the code and reproduce the obtained results, this section includes the necessary instructions to do so. It is important to remark that all the code that has been produced has only been tested in Linux and MacOS. Hence, for the sake of convenience, I recommend using a similar environment to avoid potential problems. <br>

If the user wishes to engage with the code and reproduce the obtained results, this section includes the necessary instructions to do so. First, the user will have to create their own version of the repository by cloning it from GitHub. This is done by executing the following from the command line: 

```
$ git clone https://github.com/sofieditmer/classification.git
```

Once the user has cloned the repository, a virtual environment must be set up in which the relevant dependencies can be installed. To set up the virtual environment and install the relevant dependencies, a bash-script is provided, which automatically creates a virtual environment and installs the necessary dependencies listed in the ```requirements.txt``` file. To run the bash-script that sets up the virtual environment and installs the relevant dependencies, the user must execute the following from the command line. 

```
$ cd classification
$ bash create_classification_venv.sh 
```

Once the virtual environment has been set up and the relevant dependencies listed in the ```requirements.txt``` have been installed within it, the user is now able to run the two scripts, ```lr-mnist.py``` and ```nn-mnist.py```, provided in the ```src``` folder directly from the command line. The user has the option of specifying additional arguments, however, this is not required to run the script. In order to run the script, the user must first activate the virtual environment in which the script can be run. Activating the virtual environment is done as follows.

```
$ source classification_venv/bin/activate
```

Once the virtual environment has been activated, the user is now able to run the lr-mnist.py as well as the nn-mnist.py within it.

```
(classification_venv) $ cd src

(classification_venv) $ python lr-mnist.py

(classification_venv) $ python nn-mnist.py
```

For the ```lr-mnist.py``` script, the user is able to modify the following parameters, however, as mentioned this is not compulsory:

```
-o, --output, str <name-of-output-file>, default = "lr_classification_metrics.txt"
-ts, --test_size: float <size-of-test-dataset>, default = 0.2
-u, --unseen_img: str <path-to-unseen-image>, default = None
```

For the ```nn-mnist.py``` script, the user is able to modify the following parameters, but again, this is not compulsory:

```
-o, --output, str <name-of-output-file>, default = "nn_classification_metrics.txt"
-ts, --test_size: float <size-of-test-dataset>, default = 0.2
-b, --batch_size: int <size-of-batches>, default = 32
-n, --n_epochs: int <number-of-epochs>, default = 20
-g, --grid_search: str <perform-grid-search-true-or-false>, default = “False”
```

The abovementioned parameters allow the user to adjust the pipeline, if necessary, but because default parameters have been set, it makes the script run without explicitly specifying these arguments.  


### Output <br>
When running the ```lr-mnist.py``` you will get two outputs saved in the specified output directory:
1. ```lr_classification_metrics.txt``` Logistic regression classification metrics.
2. ```confusion_matrix.png``` Normalized confusion matrix. This provides an overview of how well the classifier performed. 

When running the ```nn-mnist.py``` you will get four outputs saved in the specified output directory:
1. ```nn_model_summary.txt``` Summary of the model architecture.
2. ```nn_model_architecture.png``` Visual overview of the model architecture.
3. ```nn_model_loss_accuracy_history.png``` Visual representation of the loss/accuracy performance of the model during training. 
4. ```nn_classification_metrics.txt``` Neural network classification metrics.

### Discussion of Results
The initial logistic regression baseline classifier obtained a weighted average accuracy of 92% on the MNIST dataset (see [Classification Report](https://github.com/sofieditmer/classification/blob/main/output/lr_classification_metrics.txt)).
The logistic regression classifier was trained with C-value of 0.1, representing its learning strength, the 'saga' algorithm as the optimization algorithm used, a tolerance value of 0.1, and the l2 as the regularization method. These hyperparameter values have been estimated as providing the best results by grid search. The classification report reveals that the F1-scores are high for all classes with the lowest accuracy score, i.e., 88%, obtained for class 5. Given that the lowest F1-score is 88%, it clearly demonstrates that the logistic regression classifier performs very well. <br>
To test the logistic regression classifier, I also implemented a method that takes any image, performs some preprocessing steps, and uses the classifier to predict its class. The classifier was able to predict handwritten digits with 90-100% accuracy on several attempts. <br>
By comparison, the neural network classifier obtained a weighted accuracy of 98% on the MNIST dataset (see [Classification Report](https://github.com/sofieditmer/classification/blob/main/output/nn_classification_metrics.txt)).  The neural network classifier was trained for 20 epochs with a batch size of 32. The classification report for the neural network classifier reveals that the F1-scores are extremely high, 97%-99%. Hence, the neural network classifier outperforms the benchmarks of the logistic regression classifier. <br>
When assessing the loss and accuracy learning curves of the model, it is evident that the model quickly reaches high performance on both the training and validation dataset (see [Loss/Accuracy Plot](https://github.com/sofieditmer/classification/blob/main/output/nn_model_loss_accuracy_history.png)). This is manifested by the training and validation accuracy curves reaching nearly 100% accuracy immediately after the first epoch, suggesting that the model quickly learns the training data and is not able to learn more after only a few epochs. Hence, the model performs very well on the specified task, but perhaps too well. Similarly, the training and validation loss start off low, suggesting that the model very quickly and effectively learns the task. While the training loss decreases in the beginning suggesting that the model is learning, it quickly reaches a loss of 0 suggesting that there is nothing left for the model to learn about the data. The validation loss curve also starts off with a very small loss and continues at this rate. To sum up, the model seems to quickly learn the task and reach a high accuracy for both the training data and validation data. Hence, it might not be necessary to train the model for 20 epochs. 
The superior performance of the model might also be attributable to the nature of the data. The MNIST dataset is a toy-dataset that contains uniform images with only 1 color channel, and does not contain much real-world complexity and noise, which means that the model is able to quickly learn from the data. 

### License <br>
This project is licensed under the MIT License - see the [LICENSE](https://github.com/sofieditmer/classification/blob/main/LICENSE) file for details.

### Contact Details <br>
If you have any questions feel free to contact me on [201805308@post.au.dk](201805308@post.au.dk)

