import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List


DEBUG: bool = True
NUMBER_OF_EPOCHS = 100
LEARNING_RATE = 0.01


def load_data(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=",", header=None)
    return df


def normalize_data(data: pd.DataFrame):
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    Z_score = (data - mean) / std
    return Z_score


def split_train_test(data: pd.DataFrame, train_ratio: float) -> pd.DataFrame:
    data = data.sample(frac=1, random_state=42)
    train_size = int(data.shape[0] * train_ratio)
    test_size = data.shape[0] - train_size
    train_chunk = data.iloc[:train_size]
    test_chunk = data.iloc[train_size:]
    X_train = train_chunk.iloc[:, :-1]
    y_train = train_chunk.iloc[:, -1]
    X_test = test_chunk.iloc[:, :-1]
    y_test = test_chunk.iloc[:, -1]
    return X_train, y_train, X_test, y_test


# def visualize_data(X_train, y_train, X_test, y_test) -> None :
#     plt.scatter(X_train , y_train, color = 'navyblue', label = 'Train data')
#     plt.scatter(X_test, y_test, color = 'magenta', label = 'Test data')
#     plt.xlabel('')


def initialize_parameters(n_feature) -> List[float]:
    thetas = np.random.rand(n_feature + 1)
    return thetas


def model_predict(X: pd.DataFrame, thetas: List[float]) -> pd.DataFrame:
    return np.dot(X, thetas[1:]) + thetas[0]


def cost_function(y: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    return np.mean((y - y_pred) ** 2) / 2


def get_gradient(X: pd.DataFrame, y: pd.DataFrame, y_pred: pd.DataFrame) -> List[float]:
    error = (model_predict(X, thetas) - y) / X.shape[0]
    return np.dot(X.T, error)


# def gradient_descent()


if __name__ == "__main__":
    file_path = "data/insurance.csv"
    data = load_data(file_path)
    if DEBUG:
        print(data.shape)
        X_train, y_train, X_test, y_test = split_train_test(data, 0.8)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        thetas = initialize_parameters(X_train.shape[1])
        print(f"The initial thetas are : {thetas}")
