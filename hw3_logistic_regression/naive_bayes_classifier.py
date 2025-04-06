import numpy as np


PLOT_RESULTS = True


def train_NBC(X, X_dtype, Y, L, D_categorical=None):
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
        
    # If the data are categorical
    if X_dtype == 'categorical':
        # If the data are one-hot encoded
        if is_one_hot_encoded(X, D_categorical):
            likelihood_tables = np.zeros((M, C))

            # Calculate likelihoods for each one-hot feature and each class
            for m in range(M):
                for c in range(C):
                    class_indices = np.where(Y == possible_classes[c])[0]
                    likelihood_tables[m, c] = (np.sum(X[class_indices, m]) + L) / (samples_in_classes[c] + 2 * L)

            # Store the model (likelihood tables and prior probabilities)
            model = {
                'likelihood_tables': likelihood_tables,
                'prior_probabilities': prior_probabilities
            }
        else:
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
    # If the data are continuous
    elif X_dtype == 'continuous':
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
    # If the data are mixed
    else:
        # Calculate the mean and standard deviation for each feature and class
        mean = np.zeros((M, C))
        std = np.zeros((M, C))
        
        # Calculate the likelihood tables
        likelihood_tables = {}

        feature_types = infer_feature_types(X)

        # For each feature
        for m in range(M):
            if feature_types[m] == 'categorical':
                D = D_categorical[0, m] if D_categorical is not None else len(np.unique(X[:, m]))
                likelihood_table = np.zeros((D, C))
                frequency_table = np.zeros((D, C))
                
                # For each possible value of the feature
                for d in range(D):
                    indices = np.where(X[:, m] == D)[0]
                    # For each class
                    for c in range(C):
                        frequency_table[d, c] = np.sum(Y[indices] == c)
                        likelihood_table[d, c] = (frequency_table[d, c] + L) / (samples_in_classes[c] + L*D)

                likelihood_tables[m] = likelihood_table
            else:
                # For each class
                for c in range(C):
                    indices = np.where(Y == possible_classes[c])[0]
                    mean[m, c] = np.mean(X[indices, m])
                    std[m, c] = np.std(X[indices, m])
            
        # Store the model (mean, std, likelihood tables and prior probabilities)
        model = {
            'mean': mean,
            'std': std,
            'likelihood_tables': likelihood_tables,
            'prior_probabilities': prior_probabilities,
            'feature_types': feature_types
        }    
                
    return model


def predict_NBC(model, X, X_dtype, D_categorical=None):
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

    # If the data are categorical
    if X_dtype == 'categorical':
        if is_one_hot_encoded(X, D_categorical):
            likelihood_tables = model['likelihood_tables']
        
            # For each sample
            for i in range(J):
                probs = np.ones(C)
                # For each class
                for c in range(C):
                    probs[c] = prior_probabilities[c]
                    # For each feature
                    for m in range(X.shape[1]):
                        # Apply likelihoods based on one-hot encoded data
                        if X[i, m] == 1:
                            probs[c] *= likelihood_tables[m, c]
                        else:
                            probs[c] *= (1 - likelihood_tables[m, c])

                # Assign the class with the highest probability
                predictions[i] = np.argmax(probs)
        else:
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
    # If the data are continuous
    elif X_dtype == 'continuous':
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
    # If the data are mixed
    else:
        likelihood_tables = model['likelihood_tables']
        mean = model['mean']
        std = model['std']
        feature_types = model['feature_types']
        
        # For each sample
        for i in range(J):
            probs = np.ones(C)
            # For each class
            for c in range(C):
                probs[c] = prior_probabilities[c]
                # For each feature
                for m in range(M):
                    if feature_types[m] == 'categorical':
                        likelihood_table = likelihood_tables[m]
                        probs[c] *= likelihood_table[int(X[i, m]), c]
                    else:
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
    

def is_one_hot_encoded(X, D_categorical):
    """
    Check if the given matrix X is one-hot encoded, based on the expected number of columns
    for each original categorical feature as specified by D_categorical.
    Inputs:
        X: A IxM matrix of variables. Rows correspond to samples and columns to variables.
        D_categorical: A 1xM vector. Each element D(m) contains the number of possible different values that the categorical variable m can have.
    Output:
        is_one_hot: A boolean. True if the matrix is one-hot encoded, False otherwise.
    """
    start_idx = 0

    # For each categorical feature
    for num_categories in D_categorical[0]:
        # Select the subset of columns for this feature
        end_idx = start_idx + num_categories
        feature_columns = X[:, start_idx:end_idx]
        
        # Check that each value in this set of columns is either 0 or 1
        if not np.all((feature_columns == 0) | (feature_columns == 1)):
            return False
        
        # Check that each row in this feature group has exactly one '1' and the rest '0's
        if not np.all(feature_columns.sum(axis=1) == 1):
            return False
        
        # Move to the next group of columns
        start_idx = end_idx

    return True
   
   
def infer_feature_types(X, unique_threshold=15):
    """
    Infers the data type of each feature column in X.
    Categorical features are identified based on a limited number of unique values.
    Inputs:
        X (np.array): Feature matrix.
        unique_threshold (int): Maximum number of unique values for a feature to be considered categorical.
    Outputs:
        feature_types (list): List indicating 'categorical' or 'continuous' for each feature.
    """
    feature_types = []
    
    for m in range(X.shape[1]):
        unique_values = np.unique(X[:, m])
        
        # If the number of unique values is below a threshold, consider it categorical
        if len(unique_values) <= unique_threshold:
            feature_types.append('categorical')
        else:
            feature_types.append('continuous')
    
    return feature_types
