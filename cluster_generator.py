#clustering.py
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

def perform_kmeans(scaled_data, n_clusters=4):
    """Perform KMeans clustering and return cluster labels."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    return clusters

def perform_pca(scaled_data, n_components=2):
    """Perform PCA and return reduced data."""
    pca = PCA(n_components=n_components)
    pca_data = pca.fit_transform(scaled_data)
    return pca_data, pca

def add_cluster_to_data(data, clusters):
    """Add cluster labels to the original data."""
    data['Cluster'] = clusters
    return data

