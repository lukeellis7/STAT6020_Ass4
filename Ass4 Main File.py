#ass4_main.py
import pandas as pd
from data_loader import load_data, preprocess_data
from cluster_generator import perform_kmeans, perform_pca
from cluster_evaluations import elbow_method, silhouette_analysis, gap_statistic, print_all_countries_in_clusters, \
    plot_cumulative_variance
from visualisations import plot_correlation_heatmap, plot_pca, plot_pca_3d, plot_world_map, plot_pca_labeled, \
    plot_pca_3d_labeled, plot_pca_labeled_specific, plot_pca_3d_labeled_specific

#mapping
variable_mapping = {
    'AG.LND.AGRI.ZS': 'Agricultural and Rural Development',
    'BX.KLT.DINV.WD.GD.ZS': 'Economy & Growth',
    'SE.PRM.CMPT.ZS': 'Education',
    'EG.USE.COMM.GD.PP.KD': 'Energy & Mining',
    'ER.LND.PTLD.ZS': 'Environment',
    'SH.DYN.MORT': 'Health',
    'IS.ROD.PAVE.ZS': 'Infrastructure',
    'SI.POV.DDAY': 'Poverty',
    'IC.BUS.EASE.XQ': 'Private Sector',
    'IQ.CPA.PUBS.XQ': 'Financial and Public Sectors',
    'SP.URB.TOTL.IN.ZS': 'Urban Development',
    'EN.ATM.CO2E.KT': 'Climate Change'
}

#load and preprocess
data = load_data('../wbcc_bc.csv')
scaled_data, data = preprocess_data(data)

#get a data subset for correlation
selected_columns = list(variable_mapping.keys())
present_columns = [col for col in selected_columns if col in data.columns[2:]]
selected_data = data[present_columns]

#corre matrix for unscaled data
correlation_matrix = selected_data.corr()
plot_correlation_heatmap(correlation_matrix, variable_mapping, title='Correlation Matrix for Selected Indicators')

#get column match right for scaled df
scaled_data_df = pd.DataFrame(scaled_data, columns=data.columns[2:79])  # Ensure 77 columns
scaled_selected_data = scaled_data_df[present_columns]

#corre matrix for scaled data
scaled_correlation_matrix = scaled_selected_data.corr()
plot_correlation_heatmap(scaled_correlation_matrix, variable_mapping, title='Correlation Matrix for Scaled Data')

#plot variance cumulative
plot_cumulative_variance(scaled_data, n_components=10)

#cluster and PCA visualization loop for 2, 3, 4, 5, and 6, 7 clusters
for n_clusters in [2, 3, 4, 5, 6, 7]:
    print(f"Running clustering and PCA for {n_clusters} clusters...")

    #KMeans clustering with n_clusters
    clusters = perform_kmeans(scaled_data, n_clusters=n_clusters)
    print_all_countries_in_clusters(data, clusters, n_clusters=n_clusters)

    #2D PCA and plot
    pca_data_2d, _ = perform_pca(scaled_data)
    plot_pca(pca_data_2d, clusters, title=f'2D PCA-Reduced Data with {n_clusters} Clusters')
    plot_pca_labeled(pca_data_2d, clusters, data['country'], title=f'2D PCA with Labels ({n_clusters} Clusters)')

    #3D PCA and plot
    pca_data_3d, _ = perform_pca(scaled_data, n_components=3)
    #Rotate3D plot for better visualization
    plot_pca_3d(pca_data_3d, clusters, title=f'3D PCA-Reduced Data with {n_clusters} Clusters', elev=35, azim=45)
    plot_pca_3d(pca_data_3d, clusters, title=f'3D PCA-Reduced Data with {n_clusters} Clusters', elev=60, azim=210)
    plot_pca_3d(pca_data_3d, clusters, title=f'3D PCA-Reduced Data with {n_clusters} Clusters', elev=25, azim=130)


    #Plots with labels
    plot_pca_3d_labeled(pca_data_3d, clusters, data['country'], title=f'3D PCA with Labels ({n_clusters} Clusters)',
                        elev=35, azim=45)
    #Plots with certain labels
    specific_countries = [
        'United States', 'China', 'India', 'Australia', 'Germany', 'Chad', 'Botswana', 'Fiji', 'Tonga', 'Russian Federation',
        'Norway', 'Thailand', 'South Africa', 'Nigeria', 'Kenya', 'Egypt', 'Ethiopia', 'Saudi Arabia', 'Qatar',
        'Japan', 'Bangladesh', 'Indonesia', 'South Korea', 'France', 'United Kingdom', 'Poland', 'Sweden',
        'Canada', 'Mexico', 'Guatemala', 'Nicaragua', 'Brazil', 'Argentina', 'Chile', 'Colombia', 'New Zealand',
        'Israel', 'United Arab Emirates, Tuvalu, Kiribati, Maldives, Turks and Caicos Islands'
    ]
    #Cheeky specific labelled plots
    plot_pca_labeled_specific(pca_data_2d, clusters, data['country'], specific_countries=specific_countries)
    plot_pca_3d_labeled_specific(pca_data_3d, clusters, data['country'], specific_countries=specific_countries, elev=30,
                                 azim=120)

#folium map plots
geojson_url = 'https://raw.githubusercontent.com/johan/world.geo.json/master/countries.geo.json'
plot_world_map(data, geojson_url, 'EN.ATM.CO2E.PC', legend_name="CLIMATE CHANGE: CO2 Emissions per Capita (Metric Tons)")
plot_world_map(data, geojson_url, 'AG.LND.AGRI.ZS', legend_name="AGRICULTURAL AND RURAL DEVELOPMENT: Agricultural land (% of land area)")
plot_world_map(data, geojson_url, 'BX.KLT.DINV.WD.GD.ZS', legend_name="ECONOMY AND GROWTH: Foreign direct investment, net inflows (% of GDP)")
plot_world_map(data, geojson_url, 'SE.PRM.CMPT.ZS', legend_name="EDUCATION: Primary completion rate, total (% of relevant age group)")
plot_world_map(data, geojson_url, 'EG.USE.COMM.GD.PP.KD', legend_name="ENERGY AND MINING: Energy use (kg of oil equivalent) per $1,000 GDP (constant 2017 PPP)")
plot_world_map(data, geojson_url, 'ER.LND.PTLD.ZS', legend_name="ENVIRONMENT: Terrestrial protected areas (% of total land area)")
plot_world_map(data, geojson_url, 'IQ.CPA.PUBS.XQ', legend_name="FINANCIAL SECTOR/PUBLIC SECTOR: CPIA public sector management and institutions cluster average")
plot_world_map(data, geojson_url, 'SH.DYN.MORT', legend_name="HEALTH: Mortality rate, under-5 (per 1,000 live births)")
plot_world_map(data, geojson_url, 'IS.ROD.PAVE.ZS', legend_name="INFRASTRUCTURE: Roads, paved (% of total roads)")
plot_world_map(data, geojson_url, 'SI.POV.DDAY', legend_name="POVERTY: Poverty headcount ratio at $1.90 a day (2011 PPP) (% of population)")
plot_world_map(data, geojson_url, 'IC.BUS.EASE.XQ', legend_name="PRIVATE SECTOR: Ease of doing business index (1=most business-friendly regulations)")
plot_world_map(data, geojson_url, 'SP.URB.TOTL.IN.ZS', legend_name="URBAN DEVELOPMENT: Urban population (% of total population)")
plot_world_map(data, geojson_url, 'EN.ATM.CO2E.KT', legend_name="CLIMATE CHANGE: CO2 emissions (kt)")

#clustering evaluations for scaled data
elbow_method(scaled_data)
silhouette_analysis(scaled_data)
gap_statistic(scaled_data)
