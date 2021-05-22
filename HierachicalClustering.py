# imports
from sklearn.cluster import AgglomerativeClustering
from DunnIndex import *
import matplotlib
from sklearn.metrics.pairwise import euclidean_distances
import scipy.cluster.hierarchy as sch
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from scipy.stats import rankdata

matplotlib.use('TkAgg')
from matplotlib import pyplot as plt


def ward_clustering(data, k, is_reduced=False):
    """
    This method does a ward clustering with plots as an output
    :param data: this variable includes the corpus, which should be clustered
    :param k: this is the cut off value
    :param is_reduced: this boolean indicates whether data is reduced or not
    """
    # Checking threshold
    pca = PCA(n_components=2)
    if is_reduced:
        reduced_features = pca.fit_transform(data)
    else:
        reduced_features = pca.fit_transform(data.toarray())
    plt.figure(figsize=(12, 6))
    dendogramm = sch.dendrogram(sch.linkage(reduced_features, method="ward"))
    plt.show()

    # Clusering
    h_clus = AgglomerativeClustering(n_clusters=k, affinity="euclidean", linkage="ward")
    h_clus.fit(reduced_features)
    labels = h_clus.labels_
    plt.scatter(reduced_features[labels == 0, 0], reduced_features[labels == 0, 1], s=50, marker='o', color='red')
    plt.scatter(reduced_features[labels == 1, 0], reduced_features[labels == 1, 1], s=50, marker='o', color='blue')
    plt.scatter(reduced_features[labels == 2, 0], reduced_features[labels == 2, 1], s=50, marker='o', color='green')
    plt.scatter(reduced_features[labels == 3, 0], reduced_features[labels == 3, 1], s=50, marker='o', color='purple')
    plt.scatter(reduced_features[labels == 4, 0], reduced_features[labels == 4, 1], s=50, marker='o', color='orange')
    plt.title("Hierarchial Clustering (Ward) with Cut-off: %i" % k)
    plt.show()
    # calculates the validation indices
    if is_reduced:
        print("Silhouette Index: ", round(silhouette_score(data, labels=labels), 3))
        print("Calinski Harabaz Index: ", round(calinski_harabasz_score(data, labels=labels), 1))
        print("Davies Bouldin Index: ", round(davies_bouldin_score(data, labels=labels), 3))
        print("Dunn Index: ", round(dunn(labels=labels, distances=euclidean_distances(data)), 3))
    else:
        print("Silhouette Index: ", round(silhouette_score(data, labels=labels), 3))
        print("Calinski Harabaz Index: ", round(calinski_harabasz_score(data.toarray(), labels=labels), 1))
        print("Davies Bouldin Index: ", round(davies_bouldin_score(data.toarray(), labels=labels), 3))
        print("Dunn Index: ", round(dunn(labels=labels, distances=euclidean_distances(data.toarray())), 3))




def single_linkage(data, k):
    """
    This method does a single linkage clustering with plots as an output
    :param data: this variable includes the corpus, which should be clustered
    :param k: this is the cut off value
    """
    # Checking threshold
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(data.toarray())
    plt.figure(figsize=(12, 6))
    dendogramm = sch.dendrogram(sch.linkage(reduced_features, method="single"))
    plt.show()

    # Clusering
    h_clus = AgglomerativeClustering(n_clusters=k, affinity="euclidean", linkage="single")
    h_clus.fit(reduced_features)
    labels = h_clus.labels_
    plt.scatter(reduced_features[labels == 0, 0], reduced_features[labels == 0, 1], s=50, marker='o', color='red')
    plt.scatter(reduced_features[labels == 1, 0], reduced_features[labels == 1, 1], s=50, marker='o', color='blue')
    plt.scatter(reduced_features[labels == 2, 0], reduced_features[labels == 2, 1], s=50, marker='o', color='green')
    plt.scatter(reduced_features[labels == 3, 0], reduced_features[labels == 3, 1], s=50, marker='o', color='purple')
    plt.scatter(reduced_features[labels == 4, 0], reduced_features[labels == 4, 1], s=50, marker='o', color='orange')
    plt.title("Hierarchial Clustering (single) with Cut-off: %i" % k)
    plt.show()

    # calculates the validation indices
    print("Silhouette Index: ", round(silhouette_score(data, labels=labels), 3))
    print("Calinski Harabaz Index: ", round(calinski_harabasz_score(data.toarray(), labels=labels), 1))
    print("Davies Bouldin Index: ", round(davies_bouldin_score(data.toarray(), labels=labels), 3))
    print("Dunn Index: ", round(dunn(labels=labels, distances=euclidean_distances(data.toarray())), 3))


def complete_linkage(data, k):
    """
    This method does a complete linkage clustering with plots as an output
    :param data: this variable includes the corpus, which should be clustered
    :param k: this is the cut off value
    """
    # Checking threshold
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(data.toarray())
    plt.figure(figsize=(12, 6))
    dendogramm = sch.dendrogram(sch.linkage(reduced_features, method="complete"))
    plt.show()

    # Clusering
    h_clus = AgglomerativeClustering(n_clusters=k, affinity="euclidean", linkage="complete")
    h_clus.fit(reduced_features)
    labels = h_clus.labels_
    plt.scatter(reduced_features[labels == 0, 0], reduced_features[labels == 0, 1], s=50, marker='o', color='red')
    plt.scatter(reduced_features[labels == 1, 0], reduced_features[labels == 1, 1], s=50, marker='o', color='blue')
    plt.scatter(reduced_features[labels == 2, 0], reduced_features[labels == 2, 1], s=50, marker='o', color='green')
    plt.scatter(reduced_features[labels == 3, 0], reduced_features[labels == 3, 1], s=50, marker='o', color='purple')
    plt.scatter(reduced_features[labels == 4, 0], reduced_features[labels == 4, 1], s=50, marker='o', color='orange')
    plt.title("Hierarchial Clustering (complete) with Cut-off: %i" % k)
    plt.show()

    # calculates the validation indices
    print("Silhouette Index: ", round(silhouette_score(data, labels=labels), 3))
    print("Calinski Harabaz Index: ", round(calinski_harabasz_score(data.toarray(), labels=labels), 1))
    print("Davies Bouldin Index: ", round(davies_bouldin_score(data.toarray(), labels=labels), 3))
    print("Dunn Index: ", round(dunn(labels=labels, distances=euclidean_distances(data.toarray())), 3))


def check_k(data, algo, max_k):
    """
    This method check the best combination of cut off calue with the different combination methods
    :param data: this variable includes the corpus, which should be clustered
    :param algo: this variable indicates the clustering method
    :param max_k: this variable includes the maximum cut off value which should be tested
    """
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(data.toarray())
    range_n_clusters = range(2, max_k, 1)
    save_sil = []
    save_db = []
    save_ch = []
    save_d = []

    # creates outcome for different values
    for n_clusters in range_n_clusters:
        h_clus = AgglomerativeClustering(n_clusters=n_clusters, affinity="euclidean", linkage=algo)
        h_clus.fit(reduced_features)
        labels = h_clus.labels_
        save_sil.append(silhouette_score(data, labels=labels))
        save_db.append(davies_bouldin_score(data.toarray(), labels=labels))
        save_d.append(dunn(labels=labels, distances=euclidean_distances(data.toarray())))
        save_ch.append(calinski_harabasz_score(data.toarray(), labels=labels))
    ranking = np.array([rankdata(save_sil), rankdata([-1 * i for i in save_db]), rankdata(save_d), rankdata(save_ch)])
    ranking = np.array([rankdata(save_sil), rankdata([-1 * i for i in save_db]), rankdata(save_d), rankdata(save_ch),
                        rankdata([ranking[:, 0].sum(), ranking[:, 1].sum(), ranking[:, 2].sum(), ranking[:, 3].sum(),
                                  ranking[:, 4].sum(), ranking[:, 5].sum(), ranking[:, 6].sum(),
                                  ranking[:, 7].sum()])])

    # plots all the insights
    plt.figure(12)
    plt.plot(range_n_clusters, save_sil)
    plt.plot(range_n_clusters, save_db)
    plt.plot(range_n_clusters, save_d)
    plt.plot(range_n_clusters, save_ch)
    plt.ylabel("Index")
    plt.xlabel("# of Clusters")
    plt.title("Searching right k")
    plt.legend(["Sil", "db", "d", "ch"],
               loc="upper right")
    plt.figure(13)
    plt.plot(range_n_clusters, ranking[0])
    plt.plot(range_n_clusters, ranking[1])
    plt.plot(range_n_clusters, ranking[2])
    plt.plot(range_n_clusters, ranking[3])
    plt.plot(range_n_clusters, ranking[4], ':')
    plt.ylabel("Ranking")
    plt.xlabel("# of Clusters")
    plt.title("Searching right k in Ranking")
    plt.legend(["Sil", "db", "d", "ch", "ave"],
               loc="upper right")
    plt.show()

