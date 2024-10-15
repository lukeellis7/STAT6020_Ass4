import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
data = pd.read_csv('../wbcc_bc.csv')
print(data.shape)
print(data.head())
print(data.isnull().sum())

# Impute missing values using KNN Imputer on numeric columns only
imputer = KNNImputer(n_neighbors=5)
numeric_data = data.iloc[:, 2:]  # Selecting only numeric columns for imputation
numeric_data = imputer.fit_transform(numeric_data)

# Scale the imputed numeric data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)

# Perform K-Means clustering on the scaled numeric data
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(scaled_data)

# Add clusters back to the original DataFrame (only the non-numeric parts and clusters)
data['Cluster'] = clusters

# Display the data with clusters, including 'iso3c' and 'country' columns for reference
pd.set_option('display.max_rows', None)  # Show all rows
#print(country_clusters.to_string(index=False))  # Print the entire DataFrame without row numbers

# Reset the display option to its default to avoid affecting future outputs
print(data[['iso3c', 'country', 'Cluster']])
pd.reset_option('display.max_rows')


# Visualization: Scatter plot of the first two features in original high-dimensional space
plt.figure(figsize=(8, 5))
sns.scatterplot(x=scaled_data[:, 0], y=scaled_data[:, 1], hue=clusters, palette='viridis', s=50)
plt.title('Clustering on the First Two Features')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.legend(title='Cluster')
plt.show()


# Perform PCA to reduce dimensions to 2
pca = PCA(n_components=2)
pca_data = pca.fit_transform(scaled_data)

# Plot explained variance
plt.figure(figsize=(6, 4))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()

# Visualization: PCA-Reduced Clustering
plt.figure(figsize=(8, 5))
sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=clusters, palette='viridis', s=50)
plt.title('Clustering Visualized with PCA-Reduced Data')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend(title='Cluster')
plt.show()



# 3D Clustering Visualization using the First Three Features
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(scaled_data[:, 0], scaled_data[:, 1], scaled_data[:, 2], c=clusters, cmap='viridis', s=50)
plt.title('Clustering on the First Three Features')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_zlabel('Feature 3')
plt.legend(*sc.legend_elements(), title="Cluster")
plt.show()

# Perform PCA to reduce dimensions to 3
pca_3d = PCA(n_components=3)
pca_3d_data = pca_3d.fit_transform(scaled_data)

# 3D Visualization of PCA-Reduced Clustering
fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot(111, projection='3d')
sc = ax.scatter(pca_3d_data[:, 0], pca_3d_data[:, 1], pca_3d_data[:, 2], c=clusters, cmap='viridis', s=50)
plt.title('Clustering Visualized with PCA-Reduced Data (3D)')
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
plt.legend(*sc.legend_elements(), title="Cluster")
plt.show()





