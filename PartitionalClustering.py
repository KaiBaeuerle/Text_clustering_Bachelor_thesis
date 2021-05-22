# imports
from sklearn.cluster import KMeans
import matplotlib
import matplotlib.cm as cm
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples, davies_bouldin_score
from sklearn.metrics import calinski_harabasz_score
from DunnIndex import *
from sklearn.metrics.pairwise import euclidean_distances

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# This function does an kmeans-clustering, with either k-means++ or random assigned inital-points
def kmeans_Clustering(data, vectorizer, k, init="k-means++"):
    """
    This method does a kmeans clustering with validaiton
    :param data: the data to be clustered
    :param vectorizer: the vec of the data
    :param k: the k value, thus how much cluster should be created
    :param init: the init parameter
    """
    # Clustering
    true_k = k # defining k with the given paramter k
    # creating a 
    model = KMeans(n_clusters=true_k, init=init, max_iter=100, n_init=1)
    model.fit(data)
    centroids = model.cluster_centers_

    # Visualization
    pca = PCA(n_components=2)
    reduced_features = pca.fit_transform(data.toarray())
    reduced_cluster_centers = pca.transform(model.cluster_centers_)
    plt.scatter(reduced_features[:, 0], reduced_features[:, 1], c=model.predict(data))
    plt.scatter(reduced_cluster_centers[:, 0], reduced_cluster_centers[:, 1], marker='x', s=150, c='b')

    # Evaluation
    sil_index = silhouette_score(data, labels=model.predict(data))
    print("Silhouette Index: " + str(sil_index))
    db_index = davies_bouldin_score(data.toarray(), labels=model.predict(data))
    print("Davies Bouldin Index: ", db_index)
    ch_index = calinski_harabasz_score(data.toarray(), labels=model.predict(data))
    print("Calinski Harabaz Index: ", ch_index)
    d_index = dunn(labels=model.predict(data), distances=euclidean_distances(data.toarray()))
    print("Dunn Index: ", d_index)

    # Top terms per Cluster
    print("Top terms per cluster:")
    order_centroids = model.cluster_centers_.argsort()[:, ::-1]
    terms = vectorizer.get_feature_names()
    for i in range(true_k):
        print("Cluster %d:" % i),
        for ind in order_centroids[i, :20]:
            print(' %s' % terms[ind]),
        print


def single_elbow_plot(data, max_k, is_random, text):
    """
    This method creates a elbow plot
    :param data: the data for the elbow plot
    :param max_k: the max k value, thus the method or kmenas++ should be computed
    :param text: the text for the title of the plot
    """
    range_n_clusters = range(2, max_k, 1)  # Amount of Clusters
    distortions = []
    # comuptes a lot of cluster results
    for n_clusters in range_n_clusters:
        if is_random:
            clusterer = KMeans(n_clusters=n_clusters, random_state=10, init="random")
        else:
            clusterer = KMeans(n_clusters=n_clusters, random_state=10, init="k-means++")
        clusterer.fit_predict(data)
        distortions.append(clusterer.inertia_)
    # generates the elbow plot
    plt.figure("Elbow " + text)
    plt.plot(range_n_clusters, distortions)
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k with ' + text)


def big_visualization_kmeans(data, max_k, is_massage, is_random, name, is_reduced=False):
    """
    this method does a big comparison between a lot of different k values
    :param data: the data for the elbow plot
    :param max_k: the max k value, thus the method should be computed
    :param is_massage: indicates whether the corpus is in massage format or in sentences
    :param is_random: indicates whether the the random method should be computed
    :param name: is a part of the title of the plot
    :param is_reduced: indicates whether the corpus is in massage format or in sentences
    """
    add_fig_count = 0
    if is_massage:
        add_fig_count = 30
    # Big Visualization
    range_n_clusters = range(2, max_k, 1)  # Amount of Clusters
    save_sil = []
    save_db = []
    save_ch = []
    save_d = []
    distortions = []
    pca = PCA(n_components=2)
    if is_reduced:
        reduced_features = pca.fit_transform(data)
    else:
        reduced_features = pca.fit_transform(data.toarray())
    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(reduced_features) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        if is_random:
            clusterer = KMeans(n_clusters=n_clusters, random_state=10, init="random")
        else:
            clusterer = KMeans(n_clusters=n_clusters, random_state=10, init="k-means++")
        cluster_labels = clusterer.fit_predict(data)

        # Data for the Elbow graph
        distortions.append(clusterer.inertia_)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(data, cluster_labels)  # euclidean Distance
        if is_reduced:
            db_index = davies_bouldin_score(data, cluster_labels)  # euclidean Distance <-- No others possible
            ch_index = calinski_harabasz_score(data, cluster_labels)
            if is_massage:
                d_index = dunn(labels=cluster_labels,
                               distances=euclidean_distances(data))  # euclidean Distance
        else:
            db_index = davies_bouldin_score(data.toarray(), cluster_labels)  # euclidean Distance <-- No others possible
            ch_index = calinski_harabasz_score(data.toarray(), cluster_labels)
            if is_massage:
                d_index = dunn(labels=cluster_labels, distances=euclidean_distances(data.toarray()))  # euclidean Distance
        save_sil.append(silhouette_avg)
        save_db.append(db_index)
        save_ch.append(ch_index)
        if is_massage:
            save_d.append(d_index)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", round(silhouette_avg, 4))
        print("For n_clusters =", n_clusters,
              "The davies bouldin is :", round(db_index, 2))
        print("For n_clusters =", n_clusters,
              "The calinski harbaz index is :", round(ch_index, 1))
        if is_massage:
            print("For n_clusters =", n_clusters,
                  "The Dunn index is :", round(d_index, 2))

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(data, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(reduced_features[:, 0], reduced_features[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors, edgecolor='k')

        # Labeling the clusters
        centers = pca.transform(clusterer.cluster_centers_)

        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1], marker='o',
                    c="white", alpha=1, s=200, edgecolor='k')

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1,
                        s=50, edgecolor='k')

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on " + name +
                      " with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')
    # plotting the data
    plt.figure(len(range_n_clusters) + 2 + add_fig_count)
    plt.plot(range_n_clusters, save_sil)
    plt.ylabel("Sillhouette Index")
    plt.xlabel("# of Clusters")
    plt.title("Silhouette Index Comparison")

    plt.figure(len(range_n_clusters) + 3 + add_fig_count)
    plt.plot(range_n_clusters, save_db)
    plt.ylabel("Davies Bouldin Index")
    plt.xlabel("# of Clusters")
    plt.title("Davies Bouldin Index Comparison")

    plt.figure(len(range_n_clusters) + 4 + add_fig_count)
    plt.plot(range_n_clusters, save_ch)
    plt.ylabel("Calinski-Harabasz Index")
    plt.xlabel("# of Clusters")
    plt.title("Calinski-Harabasz Index Comparison")

    plt.figure(len(range_n_clusters) + 5 + add_fig_count)
    plt.plot(range_n_clusters, distortions)
    plt.xlabel('k')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')

    if is_massage:
        plt.figure(len(range_n_clusters) + 6 + add_fig_count)
        plt.plot(range_n_clusters, save_d)
        plt.xlabel('# of Clusters')
        plt.ylabel('Dunn Index')
        plt.title('Dunn Index Comparison')
