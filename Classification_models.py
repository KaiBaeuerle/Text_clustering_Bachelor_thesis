from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
from sklearn.cluster import KMeans
import pandas as pd
from sklearn.metrics import f1_score
import math


def random_forest_clas(data, labels):
    """
    Does a random forest classification with the given labeled data
    :param data: a df, which consists the to be classified data
    :param labels: labels, which represents the true values
    """
    # splits the data in a training and a test set
    df = pd.DataFrame(data.toarray())
    df["result_kmeans"] = labels
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
    train, test = df[df['is_train'] == True], df[df['is_train'] == False]
    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))
    features = df.columns[:5876]
    y = train["result_kmeans"]
    # Creates a model and trains it with the data
    clf = RandomForestClassifier(n_jobs=2, random_state=0)
    clf.fit(train[features], y)
    # Apply the trained Classifier to the test data
    results = clf.predict(test[features])
    # create confusion matrix
    ct = pd.crosstab(test["result_kmeans"], results, rownames=['Actual Cluster'], colnames=['Predicted Cluster'])
    print(ct)
    print("F-Measure: " + str(round(f1_score(test["result_kmeans"], results, average='weighted'), 3)))
    test["pred"] = results
    print("Entrpoy: " + str(round(entropy(df=test), 3)))


def neural_netwrok_clas(data, labels):
    """
    Creates a small neural network and classifies with the given labeled data
    :param data: data: a df, which consists the to be classified data
    :param labels: labels, which represents the true values
    """
    # splits the data in a training and a test set
    df = pd.DataFrame(data.toarray())
    df["result_kmeans"] = labels
    df['is_train'] = np.random.uniform(0, 1, len(df)) <= .75
    train, test = df[df['is_train'] == True], df[df['is_train'] == False]
    print('Number of observations in the training data:', len(train))
    print('Number of observations in the test data:', len(test))
    features = df.columns[:5876]
    y = train["result_kmeans"]
    # Creates a model and trains it with the data
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(train[features], y)
    # Apply the trained Classifier to the test data
    results = clf.predict(test[features])
    # create confusion matrix
    ct = pd.crosstab(test["result_kmeans"], results, rownames=['Actual Cluster'], colnames=['Predicted Cluster'])
    print(ct)
    print("F-Measure: " + str(round(f1_score(test["result_kmeans"], results, average='weighted'), 3)))
    test["pred"] = results
    print("Entrpoy: " + str(round(entropy(df=test), 3)))


# Method for calculating the entropy of the classification methods
def entropy(df):
    """
    Method for calculating the entropy of the classification methods
    :param df: an df, which consists the data and the labels
    :return: returns the entropy
    """
    H_of_clus = 0
    n_all = len(df)
    a_set = set(df["result_kmeans"])
    for cluster_amount in a_set:
        cluster = df[df["result_kmeans"] == cluster_amount]
        n = len(cluster)
        b_set = set(cluster["pred"])
        H_of_clas = 0
        for class_amount in b_set:
            classer = cluster[cluster["pred"] == class_amount]
            w_c = len(classer)
            H_of_clas = H_of_clas + (w_c / n) * math.log2(w_c / n)
        H_of_clus = H_of_clus + (H_of_clas * (n / n_all))
    return -H_of_clus
