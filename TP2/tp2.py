import numpy as np
import pandas as pd

def load_data(file_path : str) -> pd.DataFrame :
    df = pd.read_csv(file_path, sep = ',', header = None)
    return df




def normalize_data(data : pd.DataFrame) : 
    mean = np.mean(data, axis = 0)
    std = np.std(data, axis = 0)
    Z_score = (data - mean) / std
    return Z_score

def split_train_test(data : pd.DataFrame, train_ratio : float) -> pd.DataFrame :
    data = data.sample(frac = 1, random_state=42)
    train_size = int(data.shape[0] * train_ratio)
    test_size = data.shape[0] - train_size
    train_chunk = data.iloc[:train_size]
    test_chunk = data.iloc[train_size:]
    X_train = train_chunk.iloc[:, :-1]
    y_train = train_chunk.iloc[:, -1]
    X_test = test_chunk.iloc[:, :-1]
    y_test = test_chunk.iloc[:, -1]
    return X_train, y_train, X_test, y_test
    
    
    
    
    
    
if __name__=='__main__':
    file_path = 'data/insurance.csv'
    data = load_data(file_path)
    print(data)