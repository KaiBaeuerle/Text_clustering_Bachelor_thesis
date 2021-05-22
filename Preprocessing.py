# Imports
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np
import re
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

# import the data
csvData = open("Dec2015_ISA.csv")
com_data = csvData.read()

# create an object of PorterStemmer
porter = PorterStemmer()


def data_in_massages_tfidf(com_data, stemming):
    """
    this methods creates a corpus in tfidf format with the massages
    :param com_data:
    :param stemming:
    :return:
    """
    com_data = com_data.split("\n")

    for p in range(len(com_data)):
        spalten = com_data[p].split(";")
        com_data.append(spalten)

    # Change chars, which doesn't exist or basically are wrong
    com_data = change_wrong_chars(com_data)

    com_data = np.array(com_data, dtype=object)
    documents = com_data[0:1260]
    # Feature extraction
    if stemming:
        for doc in range(len(documents)):
            documents[doc] = stemMassages(documents[doc])
        vectorizer = TfidfVectorizer(stop_words='english')
    else:
        vectorizer = TfidfVectorizer()
    data = vectorizer.fit_transform(documents)
    return data, vectorizer


def data_in_massages_tf(com_data, stemming):
    """
    this methods creates a corpus in tf format with the massages
    :param com_data:
    :param stemming:
    :return:
    """
    com_data = com_data.split("\n")

    for p in range(len(com_data)):
        spalten = com_data[p].split(";")
        com_data.append(spalten)

    # Change chars, which doesn't exist or basically are wrong
    com_data = change_wrong_chars(com_data)

    com_data = np.array(com_data, dtype=object)
    documents = com_data[0:1260]

    # Feature extraction
    if stemming:
        for doc in range(len(documents)):
            documents[doc] = stemMassages(documents[doc])
        vectorizer = TfidfVectorizer(stop_words='english', use_idf=False)
    else:
        vectorizer = TfidfVectorizer(stop_words='english', use_idf=False)
    data = vectorizer.fit_transform(documents)
    return data, vectorizer


def data_in_sentences_tf(com_data, stemming):
    """
    this methods creates a corpus in tf format with the sentences
    :param com_data:
    :param stemming:
    :return:
    """
    com_data = sent_tokenize(com_data)
    com_data = change_wrong_chars(com_data)
    for l in com_data:
        if len(l) < 2:
            com_data.remove(l)
    com_data = np.array(com_data)
    # Show Porter Stemming
    if stemming:
        for doc in range(len(com_data)):
            com_data[doc] = stemMassages(com_data[doc])
        vectorizer = TfidfVectorizer(stop_words='english', use_idf=False)
    else:
        vectorizer = TfidfVectorizer(stop_words='english', use_idf=False)
    data = vectorizer.fit_transform(com_data)
    return data, vectorizer


def data_in_sentences_tfidf(com_data, stemming):
    """
    this methods creates a corpus in tfidf format with the sentences
    :param com_data:
    :param stemming:
    :return:
    """
    com_data = sent_tokenize(com_data)
    com_data = change_wrong_chars(com_data)
    for l in com_data:
        if len(l) < 2:
            com_data.remove(l)
    com_data = np.array(com_data)
    if stemming:
        for doc in range(len(com_data)):
            com_data[doc] = stemMassages(com_data[doc])
        vectorizer = TfidfVectorizer(stop_words='english')
    else:
        vectorizer = TfidfVectorizer()
    data = vectorizer.fit_transform(com_data)
    return data, vectorizer


# This function splits the input text in sentences
def sent_tokenize(text):
    """
    this methods split the data in sentences
    :param text:
    :return:
    """
    sentences = re.split(r"[.!?]", text)
    sentences = [sent.strip(" ") for sent in sentences]
    return sentences


# This fuction replces unwanted chars with the right ones in the input corpus
def change_wrong_chars(com_data):
    """
    this methods changes unwanted chars and replaces wit the wanted
    :param com_data: the data with wrong chars
    :return: returns the changed data
    """
    for x in range(0, len(com_data)):
        com_data[x] = com_data[x].replace("\x9a", "oe")
        com_data[x] = com_data[x].replace("\xac", "")
        com_data[x] = com_data[x].replace("\x92", "i")
        com_data[x] = com_data[x].replace("\x93", "i")
        com_data[x] = com_data[x].replace("\xec", "i")
        com_data[x] = com_data[x].replace("\xe3", "'")
        com_data[x] = com_data[x].replace("\xd0", "-")
        com_data[x] = com_data[x].replace("\x8f", "e")
        com_data[x] = com_data[x].replace("\x88", "a")
        com_data[x] = com_data[x].replace("\xd2", "")
        com_data[x] = com_data[x].replace("\xd3", "")
        com_data[x] = com_data[x].replace("\x9f", "u")
        com_data[x] = com_data[x].replace("\xa7", "ss")
        com_data[x] = com_data[x].replace("\xd4", "")
        com_data[x] = com_data[x].replace("\xd5", "")
        com_data[x] = com_data[x].replace("\xab", "")
        com_data[x] = com_data[x].replace("\xc9", " ")
        com_data[x] = com_data[x].replace("-", " ")
        com_data[x] = com_data[x].lower()
        return com_data


def dem_reduction_pca(data, amount_components, title_name):
    """
    this method reduced the data in terms of features
    :param data: the data which should be reduced
    :param amount_components: in how much components the data should be reduced
    :param title_name: this is a part of the plot title
    :return: returns the reduced data
    """
    pca = PCA()
    pca.fit_transform(data.toarray())
    pca_variance = pca.explained_variance_

    # Create figure
    plt.figure(figsize=(8, 6))
    plt.plot(pca_variance, color= 'c')
    plt.title("Described Variance " + title_name + " with reduction to: %i" % amount_components)
    plt.axvline(x=amount_components, color="red", linestyle="--")
    plt.ylabel('Variance ratio')
    plt.xlabel('Principal components')
    plt.show()

    # reduces the data
    pca2 = PCA(n_components=amount_components)
    pca2.fit(data.toarray())
    x_3d = pca2.transform(data.toarray())
    return x_3d


def plot_data(data, title_info, color="b"):
    """
    this methods plots the data
    :param data: data to be plotted
    :param title_info: this is the title of the plot
    :param color: the colors of the data points
    """
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(data.toarray())
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=color)
    plt.title("Data " + title_info)
    plt.xlabel("Feature space for the 1st feature")
    plt.ylabel("Feature space for the 2nd feature")
    plt.show()


def stemSentence(sentence):
    """
    this method stems the sentences
    :param sentence: sentences to be stemmed
    :return: the stemmed sentences
    """
    token_words = word_tokenize(sentence)
    stem_sentence = []
    for word in token_words:
        stem_sentence.append(porter.stem(word))
        stem_sentence.append(" ")
    return "".join(stem_sentence)


def stemMassages(messages):
    """
    this massage stems masssges
    :param messages: massages to be stemmed
    :return: the stemmed massages
    """
    token_sent = sent_tokenize(messages)
    stem_massages = []
    for sent in token_sent:
        stem_massages.append(stemSentence(sent))
    return "".join(stem_massages)
