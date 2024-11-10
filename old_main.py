import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import folium
#from mpl_toolkits.mplot3d import Axes3D

# Load the dataset
data = pd.read_csv('../wbcc_bc.csv')
print(data.shape)
print(data.head())

#Preprocessing
# Summary statistics
print(data.describe())


# Load a sample dataset with lat/lon data or join your data with country lat/lon data
map_data = data[['iso3c', 'country', 'EN.ATM.CO2E.PC']]  # CO2 emissions example

#create map base
world_map = folium.Map(location=[20, 0], zoom_start=2)

# Use the external GeoJSON file for country boundaries
geojson_url = 'https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json'


#add a choropleth map
folium.Choropleth(
    geo_data=geojson_url,  #find the map online
    name="CO2 Emissions",
    data=map_data,
    columns=['iso3c', 'EN.ATM.CO2E.PC'],  #country codes and CO2 values
    key_on="feature.id",  #'iso3c' country code
    fill_color="YlGnBu",
    fill_opacity=0.7,
    line_opacity=0.2,
    legend_name="CO2 Emissions per Capita (Metric Tons)"
).add_to(world_map)

#create a html file for map
world_map.save("world_co2_map.html")

# Missing values
print(data.isnull().sum())

numeric_data = data.iloc[:, 2:]  #Select only number columns, skip first two.
# Correlation heatmap
plt.figure(figsize=(12, 10))

#Filter for strong correlations
corr_matrix = numeric_data.corr()
mask = np.abs(corr_matrix) < 0.5
filtered_corr = corr_matrix.mask(mask)

#Plot filtered heatmap
sns.heatmap(filtered_corr, annot=True, cmap='coolwarm', cbar=True)
plt.title('Filtered Correlation Heatmap (|correlation| > 0.5)')
plt.show()


# Dictionary to map variable names to their corresponding topics for the heatmap
variable_mapping = {
    'AG.LND.AGRI.ZS': 'Agricultural and Rural Development', # Agricultural Land
    'BX.KLT.DINV.WD.GD.ZS': 'Economy & Growth', #Done - Foreign direct investment
    'SE.PRM.CMPT.ZS': 'Education', #Done - Primary Completion Rate
    'EG.USE.COMM.GD.PP.KD': 'Energy & Mining', #Done - energy use per capita kg oil equiv per $1000 GDP
    'ER.LND.PTLD.ZS': 'Environment', #Done - terrestrial protection areas
    'SH.DYN.MORT': 'Health', #Done - Mortality Rate Under 5
    'IS.ROD.PAVE.ZS': 'Infrastructure', #Done - % total roads paved
    'SI.POV.DDAY': 'Poverty', # Done - Poverty headcount ratio at $1.90 per day
    'IC.BUS.EASE.XQ': 'Private Sector', # Done - Ease of doing business index
    'IQ.CPA.PUBS.XQ': 'Financial and Public Sectors', #Done - CPIA public sector management clusters
    'SP.URB.TOTL.IN.ZS': 'Urban Development', #Done - Urban population as percentage of total population
    'EN.ATM.CO2E.KT': 'Climate Change' #Done - C02 emissions in kilotons
}

# List of selected standardized variables (1 per topic)
selected_columns = list(variable_mapping.keys())  # Extract variable names from the dictionary

# Check which columns are actually present in the dataset
missing_columns = [col for col in selected_columns if col not in data.columns]
present_columns = [col for col in selected_columns if col in numeric_data.columns]


# Print missing and present columns
print(f"Present columns: {present_columns}")
print(f"Missing columns: {missing_columns}")

# Subset the data to include only the columns that are actually available
selected_data = numeric_data[present_columns]

# Compute the correlation matrix for the available selected variables
correlation_matrix = selected_data.corr()

# Rename columns and index of the correlation matrix to reflect the topics
correlation_matrix.columns = [variable_mapping.get(col, col) for col in correlation_matrix.columns]
correlation_matrix.index = [variable_mapping.get(idx, idx) for idx in correlation_matrix.index]

# Plot the correlation matrix as a heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix for Selected Indicators by Topic')
plt.tight_layout()
plt.show()

# Visualize distribution of a key variable (e.g., greenhouse gas emissions)
plt.hist(numeric_data['EG.USE.PCAP.KG.OE'].dropna(), bins=30)
plt.title('Distribution of Electricity Usage Per Capita (kg of oil equivalent')
#plt.show()

#Get missing values using KNN Imputer on numeric columns only.
imputer = KNNImputer(n_neighbors=5)
#numeric_data = data.iloc[:, 2:]  #Select only number columns, skip first two.
numeric_data = imputer.fit_transform(numeric_data)

#scale the imputed numeric data (77/79 cols).
scaler = StandardScaler()
scaled_data = scaler.fit_transform(numeric_data)


# Convert the scaled_data (NumPy array) back into a DataFrame using the original numeric_data's column names
scaled_data_df = pd.DataFrame(scaled_data, columns=data.columns[2:])  # Numeric data starts from the 3rd column

# Create a new variable for the selected columns in the scaled data correlation matrix
scaled_present_columns = [col for col in variable_mapping.keys() if col in scaled_data_df.columns]

# Now, select only the columns from scaled_present_columns (the 12 selected columns)
scaled_selected_data = scaled_data_df[scaled_present_columns]

# Compute the correlation matrix for the selected variables (scaled data)
scaled_correlation_matrix = scaled_selected_data.corr()

# Rename columns and index of the correlation matrix to reflect the topics using variable_mapping
scaled_correlation_matrix.columns = [variable_mapping.get(col, col) for col in scaled_correlation_matrix.columns]
scaled_correlation_matrix.index = [variable_mapping.get(idx, idx) for idx in scaled_correlation_matrix.index]

# Plot the correlation matrix as a heatmap for the scaled data
plt.figure(figsize=(12, 8))
sns.heatmap(scaled_correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix for Scaled Data (Selected Indicators by Topic)')
plt.tight_layout()
plt.show()



























# Perform K-Means clustering on the scaled numeric data
kmeans = KMeans(n_clusters=4, random_state=42)
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