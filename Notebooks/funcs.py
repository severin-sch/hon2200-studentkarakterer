import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_training(train_loss, val_loss=None):
    """
    Makes a plot of the training and validation loss of a model
    :param train_loss:
    :param val_loss:
    :return: plot of
    """
    fig, ax = plt.subplots()
    ax.plot(train_loss)
    if val_loss is not None:
        ax.plot(val_loss)
    ax.legend(['train', 'validation'], loc='upper right')
    ax.set_ylabel('loss')
    ax.set_xlabel('epoch')
    return fig


def correlation_matrix(data, labels=None):
    """
    Makes a covariance plot of the data
    :param data: pandas dataframe or nd-array
    :param labels: set names for variables
    :return: fig of covariance
    """
    if type(data) == "pandas.core.frame.DataFrame":
        labels = [data.columns]
        data = data.values
    print(data.shape)
    print(type(data))
    plt.figure()
    print(data.T.cov.shape)
    sns.heatmap(data.cov, xticklabels=labels, yticklabels=labels)
    plt.title("Covariance matrix")