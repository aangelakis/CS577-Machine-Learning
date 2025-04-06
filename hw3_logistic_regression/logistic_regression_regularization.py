from sklearn.linear_model import LogisticRegression
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train_models(X, Y):
    """
    Train logistic regression models with different penalty values.
    Inputs:
        X: A IxM matrix of categorical variables. Rows correspond to samples and columns to variables.
        Y: A Ix1 vector. Y is the class variable to
    Outputs:
        models: A dictionary containing the trained models. The keys are the names of the models and the values are the models
    """
    
    # Using the liblinear solver because our dataset is small and liblinear is faster for small datasets
    # Smaller values of C correspond to stronger regularization
    models = {
        "No penalty": LogisticRegression(penalty='none'),
        "L1 penalty, " + r'$\lambda$=0.5': LogisticRegression(penalty='l1', C=1/0.5, solver='liblinear'),
        "L1 penalty, " + r'$\lambda$=10': LogisticRegression(penalty='l1', C=1/10, solver='liblinear'),
        "L1 penalty, " + r'$\lambda$=100': LogisticRegression(penalty='l1', C=1/100, solver='liblinear')
    }
    # Train each model
    for name, model in models.items():
        model.fit(X, Y)
        models[name] = model

    return models


def evaluate_models(models, X, Y):
    """
    Evaluate the models.
    Inputs:
        models: A dictionary containing the trained models. The keys are the names of the models and the values are the models
        X: A IxM matrix of categorical variables. Rows correspond to samples and columns to variables.
        Y: A Ix1 vector. Y is the class variable to
    Outputs:
        accuracies: A dictionary containing the accuracy of each model. The keys are the names of the models and the values are the accuracies
    """
    accuracies = {}
    for name, model in models.items():
        accuracies[name] = accuracy_score(Y, model.predict(X))
    return accuracies


def plot_weights(models):
    """
    Plot the weights of each model.
    Inputs:
        models: A dictionary containing the trained models. The keys are the names of the models and the values are the models
    """
    plt.figure(figsize=(10, 6))

    # Plot each model's weights
    for label, model in models.items():
        plt.plot(model.coef_.flatten(), label=label, marker='o')

    plt.legend()
    plt.xlabel('Weights')
    plt.ylabel('Weights\' Values')
    plt.title('Weights of Logistic Regression Models with Various Penalties')
    plt.grid(True)
    plt.savefig('figures/weights.png')
    plt.savefig('figures/weights.eps')
    plt.show()


if __name__ == '__main__':
    # Load the data
    X = pd.read_csv('Assignment3_Data/Dataset3.3_X.csv', header=None).to_numpy()
    Y = pd.read_csv('Assignment3_Data/Dataset3.3_Y.csv', header=None).to_numpy().flatten().astype(int)
    print(X.shape, Y.shape)
    
    if not os.path.exists('figures/'):
        os.makedirs('figures/')
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=42, shuffle=True)
    
        
    # Train the models
    models = train_models(X_train, y_train)
    
    # Evaluate the models
    accuracies = evaluate_models(models, X_test, y_test)
    print('Accuracies:', accuracies)
    
    # Plot the weights
    plot_weights(models)    