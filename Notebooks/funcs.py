import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split

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
    
    
def plot_corr(corr, n, title="Correlation with G3_mat and G3_por", fig_length=15, y_max=0.36, y_min=-0.36):
    """Plots a bar chart with correlation between the last two variables and the rest"""
    lablocs = np.arange(n)
    width = 0.20
    labels = corr.iloc[-2:,:-2].columns

    fig, ax = plt.subplots(figsize=(fig_length,5), dpi=100)
    ax.bar(lablocs-width/2, corr.iloc[-2,:-2],width, label='G3_mat')
    ax.bar(lablocs+width/2, corr.iloc[-1,:-2],width, label='G3_por')

    ax.set_ylabel('Correlation')
    ax.set_title(title)
    ax.set_xticks(lablocs)
    ax.set_xticklabels(labels, rotation=70)
    ax.legend()

    plt.ylim(y_min, y_max)
    fig.tight_layout()
    plt.show()
    
def data_handling(df_por, df_mat):
    """Does all datahandling outlined in 001_ss_datautforsker.ipynb
    
    returns: dataframe with numeric values
    """
    
    df= df_mat.merge(df_por, on=["school","sex","age","address","famsize","Pstatus","Medu",
                                     "Fedu","Mjob","Fjob","reason","nursery","internet"])
    df = df[[c for c in df if c not in ['G1_x', 'G2_x', 'G3_x', 'G1_y', 'G2_y', 'G3_y']] 
       + ['G1_x', 'G2_x', 'G3_x', 'G1_y', 'G2_y', 'G3_y']]
    df.rename(columns={'G1_x': 'G1_mat', 'G2_x': 'G2_mat', 'G3_x': 'G3_mat', 'G1_y': 'G1_por', 'G2_y': 'G2_por', 'G3_y': 'G3_por'}, inplace=True)
    drop = []
    rename = []
    for name in df.columns:
        if name[-2:] == "_x":
            drop.append(name)
        elif name[-2:] == "_y":
             df.rename(columns={name:name[:-2]}, inplace=True)
    df.drop(columns=drop, inplace=True)
    df_numeric = df.copy()
    to_one_hot = ["Mjob", "Fjob", "reason", "guardian"]
    
    for col in to_one_hot:
        df_numeric[col] = df_numeric[col].astype("category")
        categories = df_numeric[col].cat.categories
        df_numeric = pd.get_dummies(df_numeric, columns=[col], prefix=[col], prefix_sep='_')
        
    replace_map = {"sex": {"F":1, "M":0}, "school": {"GP":1, "MS":0}, "address":{"U":1, "R":0}, "famsize":{"GT3":1, "LE3":0},
              "schoolsup":{"yes":1, "no":0}, "famsup":{"yes":1, "no":0}, "paid":{"yes":1, "no":0},
              "activities":{"yes":1, "no":0}, "nursery":{"yes":1, "no":0},"higher":{"yes":1, "no":0},"internet":{"yes":1, "no":0},
              "romantic":{"yes":1, "no":0}, "Pstatus": {"A":1, "T":0}}
    df_numeric.replace(replace_map, inplace=True)
    df_numeric = df_numeric[[c for c in df_numeric if c not in ['G1_mat', 'G2_mat', 'G3_mat', 'G1_por', 'G2_por', 'G3_por']] 
       + ['G1_mat', 'G2_mat', 'G3_mat', 'G1_por', 'G2_por', 'G3_por']]
    
    return df_numeric


def train_test_split_M1(passed_failed = False, random_state=np.random.randint(1)):
    """Splits the data for the M1 model
    
    Y_pred must be sorted with the xxx_sort_ind before compared to Y_test."""
    
    df_mat = pd.read_csv("../Data/student-mat.csv", delimiter=";")
    df_por = pd.read_csv("../Data/student-por.csv", delimiter=";")
    df = data_handling(df_por, df_mat)
    
    Y = df[['G3_mat', 'G3_por']]
    X = df[list(df.columns[:45])+['G1_por', 'G2_por']]
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=random_state)
    
    Y_train_mat = Y_train.values[:,0]
    Y_train_por = Y_train.values[:,1]
    mat_sort_ind = Y_test.G3_mat.values.argsort()
    Y_test_mat = Y_test.G3_mat.values[mat_sort_ind]
    por_sort_ind = Y_test.G3_por.values.argsort()
    Y_test_por = Y_test.G3_por.values[por_sort_ind]
    
    if passed_failed:
        Y_train_mat = convert_y_passed_failed(Y_train_mat)
        Y_train_por = convert_y_passed_failed(Y_train_por)
        Y_test_mat = convert_y_passed_failed(Y_test_mat)
        Y_test_por = convert_y_passed_failed(Y_test_por)
    
    return X_train, X_test, Y_train_mat, Y_train_por, Y_test_mat, Y_test_por, mat_sort_ind, por_sort_ind


def convert_y_passed_failed(Y):
    return np.where(Y>=10, 1, 0)


def train_test_split_M2(passed_failed = False, random_state = np.random.randint(1)):
    """Splits the data for the M2 model
    
    Y_pred must be sorted with the xxx_sort_ind before compared to Y_test."""
    
    df_mat = pd.read_csv("../Data/student-mat.csv", delimiter=";")
    df_por = pd.read_csv("../Data/student-por.csv", delimiter=";")
    df = data_handling(df_por, df_mat)
    
    Y_mat = df.G3_mat
    Y_por = df.G3_por
    X_por = df[['G1_por', 'G2_por']]
    X_mat = df[['G1_mat', 'G2_mat']]
    X_p_train, X_p_test, Y_p_train, Y_p_test = train_test_split(X_por,Y_por, random_state = random_state)
    X_m_train, X_m_test, Y_m_train, Y_m_test = train_test_split(X_mat,Y_mat, random_state = random_state)
    
    
    mat_sort_ind = Y_m_test.values.argsort()
    Y_m_test = Y_m_test.values[mat_sort_ind]
    
    por_sort_ind = Y_p_test.values.argsort()
    Y_p_test = Y_p_test.values[por_sort_ind]
    
    if passed_failed:
        Y_m_train = convert_y_passed_failed(Y_m_train)
        Y_p_train = convert_y_passed_failed(Y_p_train)
        Y_m_test = convert_y_passed_failed(Y_m_test)
        Y_p_test = convert_y_passed_failed(Y_p_test)
    
    return [[X_m_train, X_m_test, Y_m_train, Y_m_test, mat_sort_ind], [X_p_train, X_p_test, Y_p_train, Y_p_test, por_sort_ind]]

def train_test_split_M3(columns, passed_failed=False, random_state = np.random.randint(1)):
    """Splits the dataset into training and testing and only uses the chosen columns."""
    
    df_mat = pd.read_csv("../Data/student-mat.csv", delimiter=";")
    df_por = pd.read_csv("../Data/student-por.csv", delimiter=";")
    df = data_handling(df_por, df_mat)

    Y_mat = df.G3_mat
    Y_por = df.G3_por
    X_por = df[columns]
    X_mat = df[columns]
    X_p_train, X_p_test, Y_p_train, Y_p_test = train_test_split(X_por,Y_por, random_state = random_state)
    X_m_train, X_m_test, Y_m_train, Y_m_test = train_test_split(X_mat,Y_mat, random_state = random_state)
    
    mat_sort_ind = Y_m_test.values.argsort()
    Y_m_test = Y_m_test.values[mat_sort_ind]
    
    por_sort_ind = Y_p_test.values.argsort()
    Y_p_test = Y_p_test.values[por_sort_ind]
    
    if passed_failed:
        Y_m_train = convert_y_passed_failed(Y_m_train)
        Y_p_train = convert_y_passed_failed(Y_p_train)
        Y_m_test = convert_y_passed_failed(Y_m_test)
        Y_p_test = convert_y_passed_failed(Y_p_test)
    
    return [[X_m_train, X_m_test, Y_m_train, Y_m_test, mat_sort_ind], [X_p_train, X_p_test, Y_p_train, Y_p_test, por_sort_ind]]
    

def get_X_line(df, sort_inds, idx):
    return df.loc[sort_inds[idx],:]
    
def sort_grades(G3, G2, G1):
    """Sorts G2 and G1 based on the grades of G[i+1]
    
    inputs are a sorted G3 with normal G2 and G1"""
    for i in range(0,21):
        idxs = np.where(G3==i)[0]
        if idxs.shape[0]>0:
            G2_loc = G2[idxs]
            G2_sort = G2_loc.argsort()
            G2[idxs] = G2_loc[G2_sort]
            G1[idxs] = G1[G2_sort+idxs[0]]
            for j in range(0,21):
                idxs_2 = np.where(G2[idxs]==j)[0]
                if idxs_2.shape[0]>0:
                    G1_loc = G1[idxs_2+idxs[0]]
                    G1_sort = G1_loc.argsort()
                    G1[idxs_2+idxs[0]] = G1_loc[G1_sort]
    return G3, G2, G1

