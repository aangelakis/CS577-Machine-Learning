import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import os

PLOT_RESULTS = True


def train_NBC(X, X_dtype, Y, L, D_categorical):
    """
    Method to train a Naive Bayes Classifier (NBC) using the training data.
    Inputs:
        X: A IxM matrix of categorical variables. Rows correspond to samples and columns to variables.
        X_dtype: A string describing the data type of X, which could be either 'categorical' or 'continuous'.
        Y: A Ix1 vector. Y is the class variable to predict.
        L: A scalar. L is the parameter referred to in the MAP estimates equation (for L=0, you get the MLE estimates). L >= 0.
        D_categorical: A 1xM vector. Each element D(m) contains the number of possible different values that the categorical variable m can have. This vector is ignored if X_dtype is 'continuous'.
    Outputs:
        model: This model should contain all the parameters required by the NBC to classify new samples.
    """
    # Number of samples
    I = X.shape[0]
    
    # Number of features
    M = X.shape[1]
    
    # Number of classes
    possible_classes = np.unique(Y)
    samples_in_classes = [np.sum(Y == c) for c in possible_classes]
    
    C = len(possible_classes)
    prior_probabilities = np.zeros(C)
    
    # Calculate the prior probabilities
    for c in range(C):
        prior_probabilities[c] = (samples_in_classes[c] + L) / (I + L*C)
        
    # If the data is categorical
    if X_dtype == 'categorical':
        # Calculate the frequency tables
        frequency_tables = {}
        
        # For each feature
        for i in range(M):
            D = D_categorical[0, i]
            frequency_table = np.zeros((D, C))
            
            possible_values = np.unique(X[:, i])
            
            # For each possible value of the feature
            for d, value in enumerate(possible_values):
                indices = np.where(X[:, i] == value)[0]
                
                # For each class
                for c, class_label in enumerate(possible_classes):
                    frequency_table[d, c] = np.sum(Y[indices] == class_label)
            
            frequency_tables[i] = frequency_table

        # Calculate the likelihood tables
        likelihood_tables = {}
        
        # For each feature
        for i in range(M):
            D = D_categorical[0, i]
            likelihood_table = np.zeros((D, C))
            frequency_table = frequency_tables[i]
            
            # For each possible value of the feature
            for d in range(D):
                for c in range(C):
                    likelihood_table[d, c] = (frequency_table[d, c] + L) / (samples_in_classes[c] + L*D)
            
            likelihood_tables[i] = likelihood_table
                
        # Store the model (likelihood tables and prior probabilities)
        model = {
            'likelihood_tables': likelihood_tables, 
            'prior_probabilities': prior_probabilities
        }
    # If the data is continuous
    else:
        # Calculate the mean and standard deviation for each feature and class
        mean = np.zeros((M, C))
        std = np.zeros((M, C))
        
        # For each feature
        for m in range(M):
            # For each class
            for c in range(C):
                indices = np.where(Y == possible_classes[c])[0]
                mean[m, c] = np.mean(X[indices, m])
                std[m, c] = np.std(X[indices, m])

        # Store the model (mean, std and prior probabilities)
        model = {
            'mean': mean,
            'std': std,
            'prior_probabilities': prior_probabilities
        }
        
    return model


def predict_NBC(model, X, X_dtype):
    """
    Method to predict the class of new samples using a previously trained Naive Bayes Classifier (NBC).
    Inputs:
        model: A model previously trained using train_NBC.
        X: A JxM matrix of variables. Rows correspond to samples and columns to variables.
        X_dtype: A string describing the data type of X, which could be either 'categorical' or 'continuous'.
    Ouput:
        predictions: A Jx1 vector. It contains the predicted class for each of the input samples.
    """
    
    # Number of samples
    J = X.shape[0]
    
    # Number of features
    M = X.shape[1]
    
    # Prior probabilities
    prior_probabilities = model['prior_probabilities']
    
    # Number of classes
    C = len(prior_probabilities)
    
    # Initialize the predictions
    predictions = np.zeros(J)

    # If the data is categorical
    if X_dtype == 'categorical':    
        likelihood_tables = model['likelihood_tables']
        # For each sample
        for i in range(J):
            probs = np.ones(C)
            # For each class
            for c in range(C):
                probs[c] = prior_probabilities[c]
                # For each feature
                for m in range(M):
                    likelihood_table = likelihood_tables[m]
                    probs[c] *= likelihood_table[X[i, m], c]

            # Assign the class with the highest probability
            predictions[i] = np.argmax(probs)
    # If the data is continuous
    else:
        mean = model['mean']
        std = model['std']     
        # For each sample
        for i in range(J):
            probs = np.ones(C)
            # For each class
            for c in range(C):
                probs[c] = prior_probabilities[c]
                # For each feature
                for m in range(M):
                    probs[c] *= gaussian_pdf(X[i, m], mean[m, c], std[m, c])
            
            # Assign the class with the highest probability
            predictions[i] = np.argmax(probs)
        
    return predictions
    
    
def gaussian_pdf(x, mean, std):
    """
    Method to calculate the probability density function of a Gaussian distribution.
    Inputs:
        x: A scalar. The point at which to calculate the probability density.
        mean: A scalar. The mean of the Gaussian distribution.
        std: A scalar. The standard deviation of the Gaussian distribution.
    Output:
        pdf: A scalar. The probability density at point x.
    """
    pdf = (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std)**2)
    
    return pdf    
    
    
if __name__ == '__main__':
    # Parse the arguments
    parser = argparse.ArgumentParser(description='Naive Bayes Classifier')
    parser.add_argument('--type', type=str, default='categorical', help='Data type of X. It could be either categorical or continuous', choices=['categorical', 'continuous'])
    args = parser.parse_args()
    args_dict = vars(args)
    X_dtype = args_dict['type']
    
    
    if X_dtype == 'categorical':
        # Load the categorical data
        X = pd.read_csv('Assignment2_Data/DatasetA_X_categorical.csv', header=None).to_numpy()
        Y = pd.read_csv('Assignment2_Data/DatasetA_Y.csv', header=None).to_numpy()
        D = pd.read_csv('Assignment2_Data/DatasetA_D_categorical.csv', header=None).to_numpy()
    else:
        # Load the continuous data
        X = pd.read_csv('Assignment2_Data/DatasetB_X_continuous.csv', header=None).to_numpy()
        Y = pd.read_csv('Assignment2_Data/DatasetB_Y.csv', header=None).to_numpy()
        D = None
    
    # Hyperparameter L
    Ls = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    
    runs = 500
    mean_accuracy = np.zeros(len(Ls))

    for i, L in enumerate(Ls):
        accuracy = np.zeros(runs)
        for run in tqdm(range(runs)):
            # Split the data into training and testing
            X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, shuffle=True)
                        
            # print('X_train shape:', X_train.shape,', X_test shape:', X_test.shape,', y_train shape:', y_train.shape,', y_test shape:', y_test.shape)  
                
            # Train the model
            model = train_NBC(X_train, X_dtype, y_train, L, D)
                
            # Predict the classes
            predictions = predict_NBC(model, X_test, X_dtype)
            
            # Calculate the accuracy
            accuracy[run] = accuracy_score(y_test, predictions)
            
        mean_accuracy[i] = np.mean(accuracy)
        print(f'Mean Accuracy over 100 runs for L={L}: {mean_accuracy[i]}')
    
    # Plot the results
    if PLOT_RESULTS:
        if not os.path.isdir('figures'):
            os.mkdir('figures')
            
        plt.figure(figsize=(10, 6))
        plt.plot(Ls, mean_accuracy, marker='o')
        plt.xlabel('L')
        plt.ylabel('Mean Accuracy')
        plt.grid(True)
        plt.savefig(f'figures/naive_bayes_classifier_{X_dtype}.eps', format='eps')
        plt.savefig(f'figures/naive_bayes_classifier_{X_dtype}.png', format='png')
        plt.show()