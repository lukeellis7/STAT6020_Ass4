import matplotlib.pyplot as plt
import seaborn as sns
import folium
#from mpl_toolkits.mplot3d import Axes3D


def plot_correlation_heatmap(correlation_matrix, variable_mapping, title='Correlation Matrix'):
    """Plot the heatmap for a given correlation matrix, using topic names from the variable_mapping."""
    #rename columns and index of correlation matrix to topics
    new_columns = [variable_mapping.get(col, col) for col in correlation_matrix.columns]
    new_index = [variable_mapping.get(idx, idx) for idx in correlation_matrix.index]

    #update correlation matrix with new labels
    correlation_matrix.columns = new_columns
    correlation_matrix.index = new_index

    #plot heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_pca(pca_data, clusters, title='PCA-Reduced Data (2D)'):
    """Plot the 2D PCA-Reduced Clustering"""
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=pca_data[:, 0], y=pca_data[:, 1], hue=clusters, palette='viridis', s=50)
    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.legend(title='Cluster')
    plt.tight_layout()
    plt.show()


def plot_pca_3d(pca_data, clusters, title='PCA-Reduced Data (3D)', elev=30, azim=120):
    """Plot the 3D PCA-Reduced Clustering with adjustable viewing angles."""
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c=clusters, cmap='viridis', s=50)

    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')

    plt.title(title)
    plt.legend(*sc.legend_elements(), title="Cluster")

    #view angles
    ax.view_init(elev=elev, azim=azim)  # Adjust elevation and azimuth to rotate the plot

    plt.show()


def plot_pca_labeled(pca_data, clusters, countries, title="PCA with Country Labels"):
    """Plot 2D PCA with country labels."""
    plt.figure(figsize=(12, 8))
    sc = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis', s=100)

    for i, country in enumerate(countries):
        plt.text(pca_data[i, 0], pca_data[i, 1], country, fontsize=8)

    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(sc)
    plt.show()


def plot_pca_3d_labeled(pca_data, clusters, countries, title="3D PCA with Country Labels", elev=30, azim=120):
    """Plot 3D PCA with country labels."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c=clusters, cmap='viridis', s=100)

    for i, country in enumerate(countries):
        ax.text(pca_data[i, 0], pca_data[i, 1], pca_data[i, 2], country, fontsize=8)

    ax.set_title(title)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    fig.colorbar(sc, ax=ax)

    #view angles
    ax.view_init(elev=elev, azim=azim)

    plt.show()


def plot_pca_labeled_specific(pca_data, clusters, countries, title="PCA with Specific Country Labels",
                              specific_countries=None):
    """Plot 2D PCA with specific country labels."""
    plt.figure(figsize=(12, 8))
    sc = plt.scatter(pca_data[:, 0], pca_data[:, 1], c=clusters, cmap='viridis', s=100)

    for i, country in enumerate(countries):
        if specific_countries and country in specific_countries:
            plt.text(pca_data[i, 0], pca_data[i, 1], country, fontsize=8)

    plt.title(title)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.colorbar(sc)
    plt.show()


def plot_pca_3d_labeled_specific(pca_data, clusters, countries, title="3D PCA with Specific Country Labels",
                                 specific_countries=None, elev=30, azim=120):
    """Plot 3D PCA with specific country labels."""
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    sc = ax.scatter(pca_data[:, 0], pca_data[:, 1], pca_data[:, 2], c=clusters, cmap='viridis', s=100)

    for i, country in enumerate(countries):
        if specific_countries and country in specific_countries:
            ax.text(pca_data[i, 0], pca_data[i, 1], pca_data[i, 2], country, fontsize=8)

    ax.set_title(title)
    ax.set_xlabel('Principal Component 1')
    ax.set_ylabel('Principal Component 2')
    ax.set_zlabel('Principal Component 3')
    fig.colorbar(sc, ax=ax)

    #view angles
    ax.view_init(elev=elev, azim=azim)

    plt.show()


def plot_world_map(data, geojson_url, map_column, legend_name="Map Legend"):
    """Plot the world map using Folium and a specified column from the dataset."""
    #base map create
    world_map = folium.Map(location=[20, 0], zoom_start=2)

    #choropleth layer to the map
    folium.Choropleth(
        geo_data=geojson_url,  # GeoJSON for country boundaries
        name="Choropleth Map",
        data=data,
        columns=['iso3c', map_column],  # Country code and the column to visualize
        key_on="feature.id",  # Match 'iso3c' with GeoJSON
        fill_color="YlGnBu",
        fill_opacity=0.7,
        line_opacity=0.2,
        legend_name=legend_name
    ).add_to(world_map)

    #save map to HTML
    map_filename = f"world_map_{map_column}.html"
    world_map.save(map_filename)
    print(f"World map saved as: {map_filename}")

    return world_map
