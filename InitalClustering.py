# This class runs all the methods from the other class and compares them

from PartitionalClustering import *
from Preprocessing import *
from HierachicalClustering import *
from DbClustering import *
from Classification_models import *

print("####################  Preprocessing  ######################")
# Create Data with stemming
data_stemmed, vectorizer_stemmed = data_in_massages_tfidf(com_data, True)
data_stemmed_tf, vectorizer_stemmed_tf = data_in_massages_tf(com_data, True)

# Create Data without stemming
data, vectorizer = data_in_massages_tfidf(com_data, False)
data_tf, vectorizer_tf = data_in_massages_tf(com_data, False)

# Creating Sentences with stemming
sentences_stemmed, vec_stemmed_tfidf = data_in_sentences_tfidf(com_data, True)
sentences_stemmed_tf, vec_stemmed_tf = data_in_sentences_tf(com_data, True)

# Create Sentences without stemming
sentences, vec_tfidf = data_in_sentences_tfidf(com_data, False)
sentences_tf, vec_tf = data_in_sentences_tf(com_data, False)

# Dimensionality Reduction
reduced_pca = dem_reduction_pca(data=data, amount_components=50, title_name="TFIDF")
reduced_pca_stemmed = dem_reduction_pca(data=data_stemmed, amount_components=60, title_name="TFIDF stemmed")
reduced_pca_tf = dem_reduction_pca(data=data_tf, amount_components=40, title_name="TF")
reduced_pca_stemmed_tf = dem_reduction_pca(data=data_stemmed_tf, amount_components=35, title_name="TF stemmed")

reduced_pca_sentences = dem_reduction_pca(data=sentences, amount_components=150, title_name="TFIDF - Sentences")
reduced_pca_sentences_stemmed = dem_reduction_pca(data=sentences_stemmed, amount_components=175,
                                                  title_name="TFIDF stemmed - Sentences")
reduced_pca_sentences_stemmed_tf = dem_reduction_pca(data=sentences_stemmed_tf, amount_components=100,
                                                     title_name="TF - Sentences")
reduced_pca_sentences_tf = dem_reduction_pca(data=sentences_tf, amount_components=80,
                                             title_name="TF stemmed - Sentences")
print("####################  finished  ######################")


def big_comparison_sentences_kmeans(k, is_random, is_reduced=False):
    """
    This methods compares different k values for kmeans with different data sets in sentences typ
    :param k: what is tha maximum of k
    :param is_random: variable, which indicates whether the comparison should be with the random method
    :param is_reduced: variable, which indicates whether the comparison should be with the reduced data
    """
    k_max_sentences = k  # number of ks
    if is_reduced:
        print("Stemmed TFIDF Sentences ")
        big_visualization_kmeans(reduced_pca_sentences_stemmed, k_max_sentences, False, is_random=is_random,
                                 name="Stemmed TFIDF Sentences & Dim red", is_reduced=is_reduced)
        print("TFIDF Sentences")
        big_visualization_kmeans(reduced_pca_sentences, k_max_sentences, False, is_random=is_random,
                                 name="TFIDF Sentences & Dim red", is_reduced=is_reduced)
        print("Stemmed TF Sentences")
        big_visualization_kmeans(reduced_pca_sentences_stemmed_tf, k_max_sentences, False, is_random=is_random,
                                 name="Stemmed TF Sentences & Dim red", is_reduced=is_reduced)
        print("TF Sentences")
        big_visualization_kmeans(reduced_pca_sentences_tf, k_max_sentences, False, is_random=is_random,
                                 name="TF Sentences & Dim red", is_reduced=is_reduced)
    else:
        print("Stemmed TFIDF Sentences")
        big_visualization_kmeans(sentences_stemmed, k_max_sentences, False, is_random=is_random,
                                 name="Stemmed TFIDF Sentences")
        print("TFIDF Sentences")
        big_visualization_kmeans(sentences, k_max_sentences, False, is_random=is_random, name="TFIDF Sentences")
        print("Stemmed TF Sentences")
        big_visualization_kmeans(sentences_stemmed_tf, k_max_sentences, False, is_random=is_random,
                                 name="Stemmed TF Sentences")
        print("TF Sentences")
        big_visualization_kmeans(sentences_tf, k_max_sentences, False, is_random=is_random, name="TF Sentences")

    plt.figure(k_max_sentences + 0)
    plt.legend(["TF-IDF with Stemming", "TF-IDF without Stemming", "TF with Stemming", "TF without Stemming"],
               loc="upper right")
    plt.figure(k_max_sentences + 1)
    plt.legend(["TF-IDF with Stemming", "TF-IDF without Stemming", "TF with Stemming", "TF without Stemming"],
               loc="upper right")
    plt.figure(k_max_sentences + 2)
    plt.legend(["TF-IDF with Stemming", "TF-IDF without Stemming", "TF with Stemming", "TF without Stemming"],
               loc="upper right")
    plt.figure(k_max_sentences + 3)
    plt.legend(["TF-IDF with Stemming", "TF-IDF without Stemming", "TF with Stemming", "TF without Stemming"],
               loc="upper right")
    single_elbow_plot(data=sentences_stemmed, max_k=k, is_random=is_random, text="Stemmed TFIDF Sentences")
    single_elbow_plot(data=sentences, max_k=k, is_random=is_random, text="TFIDF Sentences")
    single_elbow_plot(data=sentences_stemmed_tf, max_k=k, is_random=is_random, text="Stemmed TF Sentences")
    single_elbow_plot(data=sentences_tf, max_k=k, is_random=is_random, text="TF Sentences")


def big_comparison_masssages_kmeans(k, is_random, is_reduced=False):
    """
    This methods compares different k values for kmeans with different data sets in massage type
    :param k: what is tha maximum of k
    :param is_random: variable, which indicates whether the comparison should be with the random method
    :param is_reduced: variable, which indicates whether the comparison should be with the reduced data
    """
    k_max = k  # number of ks
    # Compare Data with and without stemming
    if is_reduced:
        big_visualization_kmeans(reduced_pca_stemmed, k_max, True, is_random=is_random, name="Stemmed TFIDF & Dim red",
                                 is_reduced=is_reduced)
        print("TFIDF")
        big_visualization_kmeans(reduced_pca, k_max, True, is_random=is_random, name="TFIDF & Dim red",
                                 is_reduced=is_reduced)
        print("Stemmed TF")
        big_visualization_kmeans(reduced_pca_stemmed_tf, k_max, True, is_random=is_random, name="Stemmed TF & Dim red",
                                 is_reduced=is_reduced)
        print("TF")
        big_visualization_kmeans(reduced_pca_tf, k_max, True, is_random=is_random, name="TF & Dim red",
                                 is_reduced=is_reduced)
    else:
        print("Stemmed TFIDF")
        big_visualization_kmeans(data_stemmed, k_max, True, is_random=is_random, name="Stemmed TFIDF",
                                 is_reduced=is_reduced)
        print("TFIDF")
        big_visualization_kmeans(data, k_max, True, is_random=is_random, name="TFIDF", is_reduced=is_reduced)
        print("Stemmed TF")
        big_visualization_kmeans(data_stemmed_tf, k_max, True, is_random=is_random, name="Stemmed TF",
                                 is_reduced=is_reduced)
        print("TF")
        big_visualization_kmeans(data_tf, k_max, True, is_random=is_random, name="TF", is_reduced=is_reduced)
    plt.figure(k_max + 0 + 30)
    plt.legend(["TF-IDF with Stemming", "TF-IDF without Stemming", "TF with Stemming", "TF without Stemming"],
               loc="upper right")
    plt.figure(k_max + 1 + 30)
    plt.legend(["TF-IDF with Stemming", "TF-IDF without Stemming", "TF with Stemming", "TF without Stemming"],
               loc="upper right")
    plt.figure(k_max + 2 + 30)
    plt.legend(["TF-IDF with Stemming", "TF-IDF without Stemming", "TF with Stemming", "TF without Stemming"],
               loc="upper right")
    plt.figure(k_max + 3 + 30)
    plt.legend(["TF-IDF with Stemming", "TF-IDF without Stemming", "TF with Stemming", "TF without Stemming"],
               loc="upper right")
    plt.figure(k_max + 4 + 30)
    plt.legend(["TF-IDF with Stemming", "TF-IDF without Stemming", "TF with Stemming", "TF without Stemming"],
               loc="upper right")
    single_elbow_plot(data=data_stemmed, max_k=k, is_random=is_random, text="Stemmed TFIDF")
    single_elbow_plot(data=data, max_k=k, is_random=is_random, text="TFIDF")
    single_elbow_plot(data=data_stemmed_tf, max_k=k, is_random=is_random, text="Stemmed TF")
    single_elbow_plot(data=data_tf, max_k=k, is_random=is_random, text="TF")


# Plotting the data
plot_data(data, "TFIDF not stemmed")
plot_data(data_stemmed, "TFIDF stemmed")
plot_data(data_tf, "TF not stemmed")
plot_data(data_stemmed_tf, "TF stemmed")
plot_data(sentences, "TFIDF not stemmed - Sentences", color="c")
plot_data(sentences_stemmed, "TFIDF stemmed - Sentences", color="c")
plot_data(sentences_tf, "TF not stemmed - Sentences", color="c")
plot_data(sentences_stemmed_tf, "TF stemmed - Sentences", color="c")


#Density-based Clustering
print("####################  Density based Clustering  ######################")
print("################  HDBSCAN  ###################")
print("## Stemmed TFIDF")
hdbscan_check(data_stemmed, 5)
print("## TFIDF")
hdbscan_check(data, 5)
print("## Stemmed TF")
hdbscan_check(data_stemmed_tf, 5)
print("## TF")
hdbscan_check(data_tf, 5)
plt.figure(8)
plt.legend(["TF-IDF with Stemming", "TF-IDF without Stemming", "TF with Stemming", "TF without Stemming"],
           loc="best")
plt.figure(9)
plt.legend(["TF-IDF with Stemming", "TF-IDF without Stemming", "TF with Stemming", "TF without Stemming"],
           loc="best")
plt.figure(10)
plt.legend(["TF-IDF with Stemming", "TF-IDF without Stemming", "TF with Stemming", "TF without Stemming"],
           loc="best")
plt.figure(11)
plt.legend(["TF-IDF with Stemming", "TF-IDF without Stemming", "TF with Stemming", "TF without Stemming"],
           loc="best")
plt.show()
hdbscan_clustering(data, min_clust=3, legend=[1, 2, 3, "Noise"])
hdbscan_clustering(data_tf, min_clust=4, legend=[1, 2, 3, "Noise"])

print("################  DBSCAN  ###################")
print("## Stemmed TFIDF")
dbscan_clustering_check(data_stemmed, 10, 105)
print("## TFIDF")
dbscan_clustering_check(data, 10, 105)
print("## Stemmed TF")
dbscan_clustering_check(data_stemmed_tf, 10, 105)
print("## TF")
dbscan_clustering_check(data_tf, 10, 105)
plt.figure(2)
plt.legend(["TF-IDF with Stemming", "TF-IDF without Stemming", "TF with Stemming", "TF without Stemming"],
           loc="best")
plt.figure(5)
plt.legend(["TF-IDF with Stemming", "TF-IDF without Stemming", "TF with Stemming", "TF without Stemming"],
           loc="best")
plt.figure(7)
plt.legend(["TF-IDF with Stemming", "TF-IDF without Stemming", "TF with Stemming", "TF without Stemming"],
           loc="best")
plt.figure(9)
plt.legend(["TF-IDF with Stemming", "TF-IDF without Stemming", "TF with Stemming", "TF without Stemming"],
           loc="best")
plt.figure(11)
plt.legend(["TF-IDF with Stemming", "TF-IDF without Stemming", "TF with Stemming", "TF without Stemming"],
           loc="best")
plt.show()
dbscan_clustering(data=data_stemmed, min_samles=4, eps=1,
                  legend=["Core 1", 1, "Core 2", 2, "Core 3", 3, "Core 4", 4, "Core 5", 5, "Noise", "Noise"])
dbscan_clustering(data=data_stemmed_tf, min_samles=4, eps=1, legend=["Core 1", 1, "Noise", "Noise"])

print("####################  Hierarchical Clustering  ######################")
print("################  Complete Linkage  ###################")
print("## Stemmed TFIDF")
check_k(data_stemmed, "complete", max_k=10)
complete_linkage(data_stemmed, k=4)
print("## Stemmed TF")
check_k(data_stemmed_tf, "complete", max_k=10)
complete_linkage(data_stemmed_tf, k=3)
print("## TFIDF")
check_k(data, "complete", max_k=10)
complete_linkage(data, k=3)
print("## TF")
check_k(data_tf, "complete", max_k=10)
complete_linkage(data_tf, k=3)

print("################  Single Linkage  ###################")
print("## Stemmed TFIDF")
check_k(data_stemmed, "single", max_k=10)
single_linkage(data_stemmed, k=2)
print("## Stemmed TF")
check_k(data_stemmed_tf, "single", max_k=10)
single_linkage(data_stemmed_tf, k=2)
single_linkage(data_stemmed_tf, k=6)
print("## TFIDF")
check_k(data, "single", max_k=10)
single_linkage(data, k=4)
print("## TF")
check_k(data_tf, "single", max_k=10)
single_linkage(data_tf, k=3)
single_linkage(data_tf, k=5)

print("################  Ward  ###################")
print("## Stemmed TFIDF")
check_k(data_stemmed, "ward", max_k=10)
ward_clustering(data_stemmed, k=2)
print("## Stemmed TF")
check_k(data_stemmed_tf, "ward", max_k=10)
ward_clustering(data_stemmed_tf, k=2)
ward_clustering(data_stemmed_tf, k=3)
print("## TFIDF")
check_k(data, "ward", max_k=10)
ward_clustering(data, k=2)
ward_clustering(data, k=3)
ward_clustering(data, k=4)
print("## TF")
check_k(data_tf, "ward", max_k=10)
ward_clustering(data_tf, k=2)
ward_clustering(data_tf, k=3)

print("####################  Pational Clustering  ######################")
print("################  K-means with kmeans++ ###################")
big_comparison_sentences_kmeans(5, is_random=False)
plt.show()
big_comparison_sentences_kmeans(5, is_random=False, is_reduced=True)
plt.show()
big_comparison_masssages_kmeans(10, is_random=False)
plt.show()
big_comparison_masssages_kmeans(10, is_random=False, is_reduced=True)
plt.show()




print("################  K-means with random ###################")
big_comparison_sentences_kmeans(5, is_random=True)
plt.show()
big_comparison_sentences_kmeans(5, is_random=True, is_reduced=True)
plt.show()
big_comparison_masssages_kmeans(10, is_random=True)
plt.show()
big_comparison_masssages_kmeans(10, is_random=True, is_reduced=True)
plt.show()


print("####################  classification  ######################")
print("################  Random forest classifier ###################")
clusterer = KMeans(n_clusters=2, random_state=10, init="k-means++").fit(data_stemmed_tf)
random_forest_clas(data_stemmed_tf, clusterer.labels_)
clusterer = KMeans(n_clusters=2, random_state=10, init="random").fit(data_stemmed_tf)
random_forest_clas(data_stemmed_tf, clusterer.predict(data_stemmed_tf))
clusterer = hdbscan.HDBSCAN(min_cluster_size=3).fit(data)
random_forest_clas(data, clusterer.labels_)
clusterer = DBSCAN(eps=1, min_samples=5).fit(data_stemmed)
random_forest_clas(data_stemmed, clusterer.labels_)
clusterer = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward").fit(data_tf.toarray())
random_forest_clas(data_tf, clusterer.labels_)
clusterer = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="single").fit(data_tf.toarray())
random_forest_clas(data_tf, clusterer.labels_)
clusterer = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="complete").fit(data_tf.toarray())
random_forest_clas(data_tf, clusterer.labels_)

print("################  MLP classifier ###################")
clusterer = KMeans(n_clusters=2, random_state=10, init="k-means++").fit(data_stemmed_tf)
neural_netwrok_clas(data_stemmed_tf, clusterer.labels_)
clusterer = KMeans(n_clusters=2, random_state=10, init="random").fit(data_stemmed_tf)
neural_network_clas(data_stemmed_tf, clusterer.predict(data_stemmed_tf))
clusterer = hdbscan.HDBSCAN(min_cluster_size=3).fit(data)
neural_netwrok_clas(data, clusterer.labels_)
clusterer = DBSCAN(eps=1, min_samples=5).fit(data_stemmed)
neural_network_clas(data_stemmed, clusterer.labels_)
clusterer = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="ward").fit(data_tf.toarray())
neural_network_clas(data_tf, clusterer.labels_)
clusterer = AgglomerativeClustering(n_clusters=5, affinity="euclidean", linkage="single").fit(data_tf.toarray())
neural_network_clas(data_tf, clusterer.labels_)
clusterer = AgglomerativeClustering(n_clusters=3, affinity="euclidean", linkage="complete").fit(data_tf.toarray())
neural_netwrok_clas(data_tf, clusterer.labels_)


print("####################  Top Words  ######################")
kmeans_Clustering(data_tf, vectorizer_tf, k=2)
kmeans_Clustering(data_stemmed_tf, vectorizer_stemmed_tf, k=2)
kmeans_Clustering(data_tf, vectorizer_tf, k=7)
kmeans_Clustering(data_tf, vectorizer_tf, k=5)


# Data for the thesis
clusterer = KMeans(n_clusters=4, random_state=10, init="k-means++")
labels = clusterer.fit_predict(data)
print(labels[0:40])
print(labels[1150:1259])
