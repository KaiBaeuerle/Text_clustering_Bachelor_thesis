# imports
import matplotlib
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, calinski_harabasz_score
from sklearn import metrics
from DunnIndex import *
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import hdbscan

matplotlib.use('TkAgg')


def dbscan_clustering_check(data, k, max_eps):
    """
    This method, tries to check which combination of eps and k generates the best validation indices. It generates plots
    to better visualize the best combination.
    :param data: an corpus in different formats
    :param k: a maximum of min_cluster-values, which should  be tested
    :param max_eps: maximum of aps-value, which should be tested
    """
    # creating empty arrays to store the values for validation
    save_values = []  # to check best result
    save_min = []  # to check best result
    save_sil = []
    save_sil_per_k = []
    save_db = []
    save_db_per_k = []
    save_ch = []
    save_ch_per_k = []
    save_dunn = []
    save_dunn_per_k = []
    range_min = range(2, k, 1)
    range_eps = range(5, max_eps, 5)
    save_cluster_amount = []
    for min_k in range_min:
        save_sil_for_k = []
        save_db_for_k = []
        save_ch_for_k = []
        save_dunn_for_k = []
        # tries out different eps values for every min_cluster value
        for eps in range_eps:
            eps_yct = float(eps) / 100
            save_values.append(eps_yct)
            save_min.append(min_k)
            db = DBSCAN(eps=eps_yct, min_samples=min_k).fit(data)
            core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
            core_samples_mask[db.core_sample_indices_] = True
            labels = db.labels_

            # Number of clusters in labels, ignoring noise if present.
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

            save_cluster_amount.append(n_clusters_)
            #  it has to get checked whether the amount of cluster is 0, because in this case its impossible to
            #  calculate the validation indices. In case there are 0 clusters the validation indices are -1
            if n_clusters_ != 0:
                save_sil.append(metrics.silhouette_score(data, labels))
                save_db.append(davies_bouldin_score(data.toarray(), labels))
                save_ch.append(calinski_harabasz_score(data.toarray(), labels))
                save_dunn.append(dunn(labels=labels, distances=euclidean_distances(data.toarray())))
                save_sil_for_k.append(metrics.silhouette_score(data, labels))
                save_db_for_k.append(davies_bouldin_score(data.toarray(), labels))
                save_ch_for_k.append(calinski_harabasz_score(data.toarray(), labels))
                save_dunn_for_k.append(dunn(labels=labels, distances=euclidean_distances(data.toarray())))
            else:
                save_sil.append(-1)
                save_sil_for_k.append(-1)
                save_db.append(100)
                save_db_for_k.append(100)
                save_ch.append(-1)
                save_ch_for_k.append(-1)
                save_dunn.append(-1)
                save_dunn_for_k.append(-1)
        if save_sil_for_k:
            save_sil_per_k.append(max(save_sil_for_k))
        if save_db_for_k:
            save_db_per_k.append(min(save_db_for_k))  # min
        if save_ch_for_k:
            save_ch_per_k.append(max(save_ch_for_k))
        if save_dunn_for_k:
            save_dunn_per_k.append(max(save_dunn_for_k))
    print("Sil-Max: ", max(save_sil))
    print("DB-Min: ", min(save_db))
    print("CH-Max: ", max(save_ch))
    print("Dunn-Max: ", max(save_dunn))
    # The plots getting created
    plt.figure(3)
    plt.plot(save_sil)
    plt.title("Sil Comparison from DBSCAN")
    plt.xlabel("min_size values")
    plt.ylabel("Sil-Index")
    plt.figure(6)
    plt.plot(save_db)
    plt.title("DB Comparison from DBSCAN")
    plt.xlabel("min_size values")
    plt.ylabel("DB-Index")
    plt.figure(8)
    plt.plot(save_ch)
    plt.title("CH Comparison from DBSCAN")
    plt.xlabel("min_size values")
    plt.ylabel("CH-Index")
    plt.figure(10)
    plt.plot(save_dunn)
    plt.title("Dunn Comparison from DBSCAN")
    plt.xlabel("min_size values")
    plt.ylabel("Dunn-Index")
    plt.figure(4)
    plt.plot(save_cluster_amount)
    plt.title("# of Clusteres in DBSCAN")
    plt.xlabel("min_size values")
    plt.ylabel("amout of clusters")
    plt.figure(5)
    plt.plot(range_min, save_sil_per_k)
    plt.title("Sil per k in DBSCAN")
    plt.xlabel("k values")
    plt.ylabel("Sil value")
    plt.figure(7)
    plt.plot(range_min, save_db_per_k)
    plt.title("DB per k in DBSCAN")
    plt.xlabel("k values")
    plt.ylabel("DB value")
    plt.figure(9)
    plt.plot(range_min, save_ch_per_k)
    plt.title("CH per k in DBSCAN")
    plt.xlabel("k values")
    plt.ylabel("CH value")
    plt.figure(11)
    plt.plot(range_min, save_dunn_per_k)
    plt.title("Dunn per k in DBSCAN")
    plt.xlabel("k values")
    plt.ylabel("Dunn value")

    # finding good eps
    neigh = NearestNeighbors(n_neighbors=2)
    nbrs = neigh.fit(data)
    distances, indices = nbrs.kneighbors(data)

    distances = np.sort(distances, axis=0)
    distances = distances[:, 1]
    plt.figure(2)
    plt.title("KNN Plt bei 3")
    plt.plot(distances)


def dbscan_clustering(data, min_samles, eps, legend):
    """
    This method clusters one corpus with the dbscan cluster method and provides plots
    :param data: this variable includes the corpus, which should be clustered
    :param min_samles: this variable sets the min_sample size per clsuter for the dbscan clustering
    :param eps: this variable sets the eps value for the dbscan clustering
    :param legend: the variable changes the legend of the created plots
    """
    # With right Values
    db = DBSCAN(eps=eps, min_samples=min_samles).fit(data)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # Shows the information about the clsutering
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    if n_clusters_ != 0:
        print("Silhouette Coefficient: %0.3f"
              % round(metrics.silhouette_score(data, labels), 3))
    print("Calinski Harabaz Index: ", round(calinski_harabasz_score(data.toarray(), labels=labels), 1))
    print("Davies Bouldin Index: ", round(davies_bouldin_score(data.toarray(), labels=labels), 3))
    print("Dunn Index: ", round(dunn(labels=labels, distances=euclidean_distances(data)), 3))
    # Plot result
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data.toarray())

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)

        xy = data_pca[class_member_mask & core_samples_mask]
        plt.figure(6)
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=14)
        xy = data_pca[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title("DB-Clustering result with min samples of %d" % min_samles + " and eps of %f" % round(eps, 1))
    plt.legend(legend, loc="upper right")

    plt.show()


def hdbscan_check(data, k):
    """
    This method, tries to check which k generates the best validation indices. It generates plots
    to better visualize which k is the best to cluster this corpus.
    :param data: this variable includes the corpus, which should be clustered
    :param k: this variable defines the maximum min cluster size
    """
    range_hdbscan_size = range(2, k, 1)
    save_sil_hdbscan = []
    save_db_hdbscan = []
    save_ch_hdbscan = []
    save_dunn_hdbscan = []
    for hdbscan_size in range_hdbscan_size:
        hdb_clusters = hdbscan.HDBSCAN(min_cluster_size=hdbscan_size)
        hdb_clusters.fit(data)
        labels = hdb_clusters.labels_
        core_samples_mask = np.zeros_like(hdb_clusters.labels_, dtype=bool)
        # core_samples_mask[hdb_clusters.core_sample_indices_] = True

        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise_ = list(labels).count(-1)

        if n_clusters_ != 0:
            # print("Silhouette Coefficient: %0.3f"
            #       % metrics.silhouette_score(data, labels))
            save_sil_hdbscan.append(metrics.silhouette_score(data, labels))
            save_db_hdbscan.append(metrics.davies_bouldin_score(data.toarray(), labels))
            save_ch_hdbscan.append(metrics.calinski_harabasz_score(data.toarray(), labels))
            save_dunn_hdbscan.append(dunn(labels=labels, distances=euclidean_distances(data.toarray())))
        else:
            save_sil_hdbscan.append(-1)
            save_db_hdbscan.append(-1)
            save_ch_hdbscan.append(-1)
            save_dunn_hdbscan.append(-1)

    print("Sil-Max: ", max(save_sil_hdbscan))
    print("DB-Min: ", min(save_db_hdbscan))
    print("CH-Max: ", max(save_ch_hdbscan))
    print("Dunn-Max: ", max(save_dunn_hdbscan))
    plt.figure(8)
    plt.plot(range_hdbscan_size, save_sil_hdbscan)
    plt.title("Sil Comparison from HDBSCAN")
    plt.xlabel("min cluster size")
    plt.ylabel("Sil-Index")
    plt.figure(9)
    plt.plot(range_hdbscan_size, save_db_hdbscan)
    plt.title("DB Comparison form HDBSCAN")
    plt.xlabel("min cluster size")
    plt.ylabel("DB-Index")
    plt.figure(10)
    plt.plot(range_hdbscan_size, save_ch_hdbscan)
    plt.title("CH Comparison form HDBSCAN")
    plt.xlabel("min cluster size")
    plt.ylabel("CH-Index")
    plt.figure(11)
    plt.plot(range_hdbscan_size, save_dunn_hdbscan)
    plt.title("Dunn Comparison form HDBSCAN")
    plt.xlabel("min cluster size")
    plt.ylabel("Dunn-Index")


def hdbscan_clustering(data, min_clust, legend):
    """

    :param data: this variable includes the corpus, which should be clustered
    :param min_clust: this variable includes the value for min cluster size
    :param legend: this variable includes the legend of the plot
    """
    # Right Value
    hdb_clusters = hdbscan.HDBSCAN(min_cluster_size=min_clust)
    hdb_clusters.fit(data)
    core_samples_mask = np.zeros_like(hdb_clusters.labels_, dtype=bool)
    labels = hdb_clusters.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)

    # output of the information about the result
    print('Estimated number of clusters: %d' % n_clusters_)
    print('Estimated number of noise points: %d' % n_noise_)
    if n_clusters_ != 0:
        print("Silhouette Coefficient: %0.3f"
              % round(metrics.silhouette_score(data, labels), 3))

    print("Calinski Harabaz Index: ", round(calinski_harabasz_score(data.toarray(), labels=labels), 1))
    print("Davies Bouldin Index: ", round(davies_bouldin_score(data.toarray(), labels=labels), 3))
    print("Dunn Index: ", round(dunn(labels=labels, distances=euclidean_distances(data)), 3))

    # Plot result
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data.toarray())

    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each)
              for each in np.linspace(0, 1, len(unique_labels))]
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        plt.figure(12)

        xy = data_pca[class_member_mask & ~core_samples_mask]
        plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col),
                 markeredgecolor='k', markersize=6)

    plt.title("HDB-Clustering result with min cluster size of %d" % min_clust)
    plt.legend(legend, loc="upper right")
    plt.cm

    plt.show()
