from trivial_classifier import trivial_train, trivial_predict
from naive_bayes_classifier import train_NBC, predict_NBC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import os


def one_hot(X):
    """
    Method to one-hot encode the data.
    Inputs: 
        X: A IxM matrix of categorical variables. Rows correspond to samples and columns to variables.
    Outputs:
        one_hot_X: A IxN matrix of one-hot encoded variables. Rows correspond to samples and columns to variables.
    """
    encoder = OneHotEncoder(sparse=False)
    one_hot_X = encoder.fit_transform(X)
    return one_hot_X


def data_loader(X_dtype: str):
    """
    Method to load the data.
    Inputs:
        X_dtype: A string describing the data type of X, which could be either 'categorical', 'continuous' or 'mixed'.
    Outputs:
        X: A IxM matrix of categorical variables. Rows correspond to samples and columns to variables.
        Y: A Ix1 vector. Y is the class variable to predict.
    """
    if X_dtype == 'categorical':
        # Load the categorical data
        X = pd.read_csv('Assignment3_Data/Dataset3.2_A_X.csv', header=None).to_numpy().astype(int)
        Y = pd.read_csv('Assignment3_Data/Dataset3.2_A_Y.csv', header=None).to_numpy().astype(int)
    elif X_dtype == 'continuous':
        # Load the continuous data
        X = pd.read_csv('Assignment3_Data/Dataset3.2_B_X.csv', header=None, sep=';').to_numpy()
        Y = pd.read_csv('Assignment3_Data/Dataset3.2_B_Y.csv', header=None).to_numpy().astype(int)
    else:
        # Load the mixed data
        X = pd.read_csv('Assignment3_Data/Dataset3.2_C_X.csv', header=None).to_numpy()
        Y = pd.read_csv('Assignment3_Data/Dataset3.2_C_Y.csv', header=None).to_numpy().astype(int)
    
    return X, Y    


if __name__ == "__main__":
    if not os.path.exists('figures/'):
        os.makedirs('figures/')

    for X_dtype in ['categorical', 'continuous', 'mixed']:
        X, Y = data_loader(X_dtype)
        D_categorical = None
        
        if X_dtype == 'categorical':
            D_categorical = np.array([len(np.unique(X[:, i])) for i in range(X.shape[1])]).reshape(1, -1)
            X = one_hot(X)
            
        logistic_model = LogisticRegression(random_state=42)
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42, shuffle=True)    
        y_train, y_test = y_train.ravel(), y_test.ravel()

        K = [50, 60, 70, 80, 80, 100]
        runs = 100
        
        average_baseline_accuracies = []
        average_nbc_accuracies = []
        average_logistic_accuracies = []
        
        for k in tqdm(K, desc='Percentage of Training Data'):
            baseline_accuracies = []
            nbc_accuracies = []
            logistic_accuracies = []
            
            for run in range(runs):
                percentage = k / 100
                X_train_k, y_train_k = X_train[:int(percentage * len(y_train))], y_train[:int(percentage * len(y_train))]
                
                # Train the models
                baseline_model = trivial_train(X_train_k, y_train_k)
                nbc_model = train_NBC(X_train_k, X_dtype, y_train_k, 1, D_categorical)
                logistic_model.fit(X_train_k, y_train_k)

                # Predict the classes
                baseline_predictions = trivial_predict(baseline_model, X_test)
                nbc_predictions = predict_NBC(nbc_model, X_test, X_dtype, D_categorical)
                logistic_predictions = logistic_model.predict(X_test)

                # Calculate the accuracies
                baseline_accuracies.append(accuracy_score(y_test, baseline_predictions))
                nbc_accuracies.append(accuracy_score(y_test, nbc_predictions))
                logistic_accuracies.append(accuracy_score(y_test, logistic_predictions))
            
            # Calculate the average accuracies
            average_baseline_accuracies.append(np.mean(baseline_accuracies))
            average_nbc_accuracies.append(np.mean(nbc_accuracies))
            average_logistic_accuracies.append(np.mean(logistic_accuracies))
            
        # Plot the accuracies
        plt.figure()
        plt.plot(K, average_baseline_accuracies, label='Baseline', marker='o')
        plt.plot(K, average_nbc_accuracies, label='Naive Bayes', marker='o')
        plt.plot(K, average_logistic_accuracies, label='Logistic Regression', marker='o')
        plt.xlabel('Percentage of Training Data')
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy vs Percentage of Training Data for {X_dtype} over {runs} runs')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'figures/{X_dtype}_accuracy_vs_k.png')
        plt.savefig(f'figures/{X_dtype}_accuracy_vs_k.eps')
        plt.show()
        
        