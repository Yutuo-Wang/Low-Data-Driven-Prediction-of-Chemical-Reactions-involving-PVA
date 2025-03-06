import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(data):
    """
    Data preprocessing function: handles missing values and standardizes numerical columns.
    """
    data.columns = data.columns.str.strip().str.lower()
    data = data.dropna(subset=['reactant smiles', 'product smiles'])
    if len(data) == 0:
        raise ValueError("No valid reactant or product SMILES data found.")
    
    # Handle possible multiple values in strings, split and take the first numeric value
    data['temperature'] = data['temperature'].apply(lambda x: float(x.split(';')[0]) if isinstance(x, str) else x)
    data['time'] = data['time'].apply(lambda x: float(x.split(';')[0]) if isinstance(x, str) else x)
    
    # Standardize numerical columns
    scaler = StandardScaler()
    data[['temperature', 'time']] = scaler.fit_transform(data[['temperature', 'time']])
    
    return data
