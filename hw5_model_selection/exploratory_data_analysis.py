import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

DATA_PREPROCESSING = False
CATEGORICAL_THRESHOLD = 12


def find_categorical_features(X):
    """
    This function takes in a Pandas DataFrame and 
    classifies the columns as categorical or continuous.
    
    Parameters
    ----------
    X : Pandas DataFrame
        The DataFrame to be classified.
    
    Returns
    -------
    categorical_features : dict
        A dictionary of categorical features and the number
        of unique values in each feature.
    continuous_features : dict
        A dictionary of continuous features and the number
        of unique values in each feature.
    """
    categorical_features = {}
    continuous_features = {}
    for column in X.columns:
        unique_values = X[column].nunique()
        if unique_values <= CATEGORICAL_THRESHOLD:
            categorical_features[column] = unique_values
        else:
            continuous_features[column] = unique_values
    return categorical_features, continuous_features


def plot_histograms(X, title=None):
    """
    Plots histograms for each column in the DataFrame X.

    Parameters
    ----------
    X : Pandas DataFrame
        The DataFrame containing data to plot.
    title : str, optional
        The title for each subplot (default is None).
    """
    plt.figure(figsize=(10, 10))  # Set the size of the figure
    for i, column in enumerate(X.columns):
        # Create a subplot for each column
        plt.subplot(3, 2, i+1)
        # Plot a histogram of the current column
        plt.hist(X[column], bins=15, color='blue', alpha=0.7)
        # Set the title of the subplot
        plt.title('Feature ' + str(column))
    plt.tight_layout()  # Adjust subplots to fit into figure area.
    plt.suptitle(title)  # Set the title of the entire figure
    # plt.show()  # Display the plot
    plt.savefig(f'{title}.png')
    
    
def main():
    """
    Main entry point for the script. Loads the dataset, computes statistics
    and plots histograms for most correlated features, features with the highest
    std/mean ratio and features with the highest dominance ratio.
    """
    dataset = pd.read_csv('Dataset5.A_XY.csv', header=None)   
    print(dataset.describe()) 
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]
    print(X.shape, y.shape)
        
    categorical_features, continuous_features = find_categorical_features(X)
    
    print(f'Categorical features: {categorical_features}, Continuous features: {continuous_features}')
    print(f'Number of categorical features: {len(categorical_features)}, Number of continuous features: {len(continuous_features)}')
    
    # get the 5 most correlated features
    corr = X.corrwith(y)
    most_corr = corr.abs().sort_values(ascending=False)[0:5]
    print('Correlation with target:\n', most_corr)    
    
    # get the 5 features with the highest std/mean (coefficient of variation) ratio (low CV -> more stable, high CV -> less stable or noisier)
    std_mean_ratio = X[continuous_features.keys()].std() / X[continuous_features.keys()].mean()
    most_std_mean = std_mean_ratio.sort_values(ascending=False)[0:5]
    print('Std/Mean ratio:\n', most_std_mean)
    
    # get the 5 features with the highest dominance ratio (close to 1 > 0.7 significant imbalance)
    categorical_df = X[categorical_features.keys()]
    dominance_ratios = categorical_df.apply(lambda col: col.value_counts(normalize=True).iloc[0]).sort_values(ascending=False)[0:5]
    print('Dominance ratio:\n', dominance_ratios)
    
    # plot histograms
    # plot_histograms(X[most_corr.index], title='Most correlated features')
    # plot_histograms(X[most_std_mean.index], title='Highest coefficient of variation ratio')
    # plot_histograms(X[dominance_ratios.index], title='Highest dominance ratio')
    
    indexes = [most_corr.index[0], most_corr.index[1], most_std_mean.index[0], most_std_mean.index[-1], dominance_ratios.index[0], dominance_ratios.index[1]]
    print(X[indexes].describe())
    plot_histograms(X[indexes], title='Selected features')  


if __name__ == '__main__':
    main()