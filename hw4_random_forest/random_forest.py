import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.utils import resample
import matplotlib.pyplot as plt
from scipy.stats import mode


def data_loader():
    """
    This function loads the data from the csv file and transforms it into numpy arrays.
    
    Parameters
    ----------
    None
    
    Returns
    -------
    X : numpy array
        This is the input data.
    y : numpy array
        These are the labels for the input data.
    """
    data = pd.read_csv('Dataset5_XY.csv', header=0, sep=',').to_numpy()
    X = data[:, 0:-1]  # input data
    Y = data[:, -1].astype(int)  # labels
    
    print(f'X shape: {X.shape}, y shape: {Y.shape}')
    
    return X, Y


def TrainRF(X, Y, n_trees, min_samples_leaf=1, max_features="sqrt", bootstrap=True):
    """
    Trains a Random Forest model using Decision Trees.

    Parameters
    ----------
    X : numpy array
        The input data.
    Y : numpy array
        The labels for the input data.
    n_trees : int
        The number of trees in the forest.
    min_samples_leaf : int
        The minimum number of samples required to be at a leaf node.
    max_features : str or int
        The number of features to consider when looking for the best split. If
        "sqrt", then the number of features is the square root of the number of
        features in the data. If int, then the number of features is the
        specified value.
    bootstrap : bool
        Whether to use bootstrapping to sample the data for each tree.

    Returns
    -------
    random_forest : dict
        A dictionary where each key is a unique identifier for each tree and the
        value is the DecisionTreeClassifier object that represents the tree.
    """
    random_forest = {}

    # Iterate through the number of trees
    for i in range(n_trees):
        # Initialize a new Decision Tree for each iteration
        random_forest[i] = DecisionTreeClassifier(
            criterion='entropy',  # The criterion used to measure the quality of a split
            min_samples_leaf=min_samples_leaf,  # The minimum number of samples required to be at a leaf node
            max_features=max_features,  # The number of features to consider when looking for the best split
            random_state=None  # The seed used to shuffle the data
        )

        # Create a bootstrap sample of the data
        if bootstrap:
            # Use resample to create a bootstrap sample of the data
            X_bs, Y_bs = resample(X, Y, replace=True)
        else:
            # If not bootstrapping, then just use the original data
            X_bs, Y_bs = X, Y

        # Train the tree on the bootstrap sample
        random_forest[i].fit(X_bs, Y_bs)
    
    return random_forest


def PredictRF(model, X):
    """
    Predict the class labels for the input data given a Random Forest model.

    Parameters
    ----------
    model : dict
        A dictionary where each key is a unique identifier for each tree and the
        value is the DecisionTreeClassifier object that represents the tree.
    X : numpy array
        The input data.

    Returns
    -------
    predictions : numpy array
        A matrix where each row is the predictions from each Decision Tree and
        each column is the predictions for each sample in the input data.
    """
    # Create a matrix to store the predictions from each Decision Tree
    predictions = np.zeros((len(model), len(X)))
    
    # Iterate through each tree in the forest
    for i in range(len(model)):
        # Get the predictions from the tree
        predictions[i] = model[i].predict(X)
    
    return predictions


def accuracy_and_plotting(predictions, Y_test, n_trees, min_samples_leaf, bonus=False, bins=20):
    """
    Calculates and plots accuracies of individual decision trees and the overall random forest model.

    Parameters
    ----------
    predictions : numpy array
        A matrix where each row is the predictions from each Decision Tree and
        each column is the predictions for each sample in the input data.
    Y_test : numpy array
        The true labels for the test data.
    n_trees : int
        The number of trees in the forest.
    min_samples_leaf : int
        The minimum number of samples required to be at a leaf node.
    bonus : bool
        If True, the plot is saved with a different name.
    bins : int
        The number of bins in the histogram.

    Returns
    -------
    None
    """
    # List to store accuracies of individual decision trees
    DT_accuracies = []
    
    # Calculate accuracy for each decision tree
    for i in range(n_trees):
        DT_accuracy = accuracy_score(Y_test, predictions[i])
        DT_accuracies.append(DT_accuracy)
        
    # Calculate the majority vote across all decision trees
    majority_vote = mode(predictions, axis=0, keepdims=True).mode[0]
    RF_predictions = majority_vote
    
    # Calculate the accuracy of the random forest based on majority vote
    RF_accuracy = accuracy_score(Y_test, RF_predictions)
    
    # Plot histogram of decision tree accuracies
    plt.figure(figsize=(10, 6))
    plt.hist(DT_accuracies, bins=bins, alpha=0.7, label='Decision Tree Accuracies')
    
    # Plot mean accuracy of decision trees
    plt.axvline(np.mean(DT_accuracies), color='k', linestyle='dashed', linewidth=2, label='Mean Accuracy of Decision Trees')
    
    # Plot accuracy of the random forest
    plt.axvline(RF_accuracy, color='r', linestyle='dashed', linewidth=2, label='Random Forest Accuracy')
    
    # Set plot labels and legend
    plt.xlabel('Accuracy')
    plt.ylabel('Count')
    plt.legend()
    
    if bonus:
        plt.savefig(f'Accuracy_Distribution_min_samples_leaf_{min_samples_leaf}_bonus_{bonus}.png')
    else:
        plt.savefig(f'Accuracy_Distribution_min_samples_leaf_{min_samples_leaf}.png')
    
    print(f'Accuracy of Random Forest: {RF_accuracy}, with min_samples_leaf = {min_samples_leaf}')
    

def bonus(X_train, Y_train, X_test, Y_test):
    """
    This function is the bonus part of the assignment. 
        
    Parameters
    ----------
    X_train : numpy array
        The training data.
    Y_train : numpy array
        The labels for the training data.
    X_test : numpy array
        The test data.
    Y_test : numpy array
        The labels for the test data.
    """
    # Train the custom Random Forest model
    random_forest = TrainRF(X_train, Y_train, n_trees=1000, min_samples_leaf=1, max_features=None, bootstrap=False)
    
    # Make predictions with the custom model
    predictions = PredictRF(random_forest, X_test)
    
    # Calculate and plot accuracies
    accuracy_and_plotting(predictions, Y_test, n_trees=1000, min_samples_leaf=1, bonus=True, bins=25)
    

def main():
    """
    This function contains the main code for the program. It trains a Random Forest model
    using a custom implementation and compares it with scikit-learn's implementation.
    Additionally, it retrains with different parameters and executes a bonus function.
    """
    # Load data from a CSV file
    X, Y = data_loader()
    
    # Set initial parameters for the Random Forest
    n_trees = 1000
    min_samples_leaf = 1
    
    # Split the data into training and testing sets
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=42)
    
    # Train the custom Random Forest model
    model = TrainRF(X_train, Y_train, n_trees, min_samples_leaf)
        
    # Make predictions with the custom model
    predictions = PredictRF(model, X_test)
    
    # Calculate and plot accuracies for the custom model
    accuracy_and_plotting(predictions, Y_test, n_trees, min_samples_leaf)
    
    # Train and evaluate scikit-learn's Random Forest model
    random_forest_sklearn = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=min_samples_leaf)
    random_forest_sklearn.fit(X_train, Y_train)
    RF_accuracy_sklearn = accuracy_score(Y_test, random_forest_sklearn.predict(X_test))
    
    # Print accuracy for scikit-learn model
    print(f'Accuracy of Random Forest using scikit-learn: {RF_accuracy_sklearn}, with min_samples_leaf = {min_samples_leaf}')
    
    # Update minimum samples per leaf and retrain the custom model
    min_samples_leaf = 10
    model = TrainRF(X_train, Y_train, n_trees, min_samples_leaf)
    
    # Make predictions with the updated custom model
    predictions = PredictRF(model, X_test)
    
    # Calculate and plot accuracies for the updated custom model
    accuracy_and_plotting(predictions, Y_test, n_trees, min_samples_leaf)
        
    # Train and evaluate scikit-learn's Random Forest model with updated min_samples_leaf
    random_forest_sklearn = RandomForestClassifier(n_estimators=n_trees, min_samples_leaf=min_samples_leaf)
    random_forest_sklearn.fit(X_train, Y_train)
    RF_accuracy_sklearn = accuracy_score(Y_test, random_forest_sklearn.predict(X_test))
    
    # Print accuracy for the updated scikit-learn model
    print(f'Accuracy of Random Forest using scikit-learn: {RF_accuracy_sklearn}, with min_samples_leaf = {min_samples_leaf}')

    # Execute bonus function with the training and testing data
    bonus(X_train, Y_train, X_test, Y_test)


if __name__ == '__main__':
    main()