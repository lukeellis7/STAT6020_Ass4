#evaluation.py
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.decomposition import PCA

import numpy as np

def elbow_method(scaled_data):
    """Plot the Elbow Method for a range of clusters."""
    wcss = []
    for n in range(1, 11):  # Test for 1 to 10 clusters
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(scaled_data)
        wcss.append(kmeans.inertia_)  #sum of squares distances to the closest centroid

    #Plot Elbow
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), wcss, marker='o')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS (Within-cluster Sum of Squares)')
    plt.grid(True)
    plt.show()

def silhouette_analysis(scaled_data):
    """Plots the silhouette score for a range of clusters."""
    silhouette_scores = []
    for n in range(2, 11):
        kmeans = KMeans(n_clusters=n, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_data)
        score = silhouette_score(scaled_data, cluster_labels)
        silhouette_scores.append(score)

    #plotted silhouette scores
    plt.figure(figsize=(8, 5))
    plt.plot(range(2, 11), silhouette_scores, marker='o')
    plt.title('Silhouette Scores for Different Numbers of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.grid(True)
    plt.show()



#GAP STATISTIC MANUALLY IMPLEMENTED
def compute_gap_statistic(scaled_data, n_refs=10, max_clusters=10):
    """
       Compute the gap statistic for a range of clusters.

       Args:
       - scaled_data: The data for clustering.
       - n_refs: Number of random reference datasets.
       - max_clusters: Maximum number of clusters to test.

       Returns:
       - gaps: The gap values for each number of clusters.
       - optimal_k: The optimal number of clusters based on the gap statistic.
       """
    shape = scaled_data.shape
    gaps = np.zeros(max_clusters)
    s_k = np.zeros(max_clusters)
    ref_disps = np.zeros((n_refs, max_clusters))

    #gen random reference datasets
    for i in range(n_refs):
        random_reference = np.random.random_sample(size=shape)

        #fit KMeans to the random reference dataset
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(random_reference)
            ref_disp, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, random_reference)
            ref_disps[i, k - 1] = np.sum(ref_disp)

    #calc actual WCSS (within-cluster sum of squares) for original data
    for k in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=k)
        kmeans.fit(scaled_data)
        disp, _ = pairwise_distances_argmin_min(kmeans.cluster_centers_, scaled_data)
        s_k[k - 1] = np.sum(disp)

    #get gap statistic
    for k in range(1, max_clusters + 1):
        ref_mean = np.mean(ref_disps[:, k - 1])
        gaps[k - 1] = np.log(ref_mean) - np.log(s_k[k - 1])

    #optimal number of clusters (maximum gap)
    optimal_k = np.argmax(gaps) + 1
    return gaps, optimal_k


#plot the Gap Statistic
def plot_gap_statistic(gaps, max_clusters=10):
    """Plot the Gap Statistic."""
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, max_clusters + 1), gaps, marker='o')
    plt.title('Gap Statistic for Optimal Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Gap Statistic')
    plt.grid(True)
    plt.show()


def gap_statistic(scaled_data):
    """Calculate and plot the Gap Statistic manually."""
    gaps, optimal_k = compute_gap_statistic(scaled_data, n_refs=10, max_clusters=10)

    print(f'Optimal number of clusters according to Gap Statistic: {optimal_k}')

    #plot Gap Statistic
    plot_gap_statistic(gaps, max_clusters=10)



def print_all_countries_in_clusters(data, clusters, n_clusters):
    """
    Print all countries in each cluster for a given number of clusters.
    Assumes 'country' column exists in the 'data' DataFrame.
    """
    print(f"\nCountries in each cluster for {n_clusters} clusters:")
    for cluster_num in range(n_clusters):
        print(f"\nCluster {cluster_num + 1}:")
        cluster_countries = data.loc[clusters == cluster_num, 'country']
        print(", ".join(cluster_countries))






def plot_cumulative_variance(scaled_data, n_components=10):
    """Plots cumulative explained variance for the PCA components."""
    #PCA on the scaled data
    pca = PCA(n_components=n_components)
    pca.fit(scaled_data)

    #cumulative explained variance
    cumulative_variance = pca.explained_variance_ratio_.cumsum()

    #plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_components + 1), cumulative_variance, marker='o', linestyle='-')
    plt.title('Cumulative Explained Variance by Number of Principal Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)
    plt.tight_layout()
    plt.show()