#data_loader.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer


def load_data(file_path):
    """Load the dataset from the specified file path."""
    data = pd.read_csv(file_path)
    return data


def preprocess_data(data):
    """Impute missing vals and scale numeric data."""
    numeric_data = data.iloc[:, 2:]  # Select only number columns, skip first two
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(numeric_data)

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(imputed_data)

    return scaled_data, data  # Return scaled numeric data and original data (with 'iso3c' and 'country')

