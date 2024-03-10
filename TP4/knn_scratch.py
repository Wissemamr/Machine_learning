import numpy as np
import pandas as pd
from colorama import Fore, init  # for colorful log messages
from typing import List, Tuple
import math

# colorful logs
init(autoreset=True)
LOG_COLOR = Fore.GREEN
ERROR_COLOR = Fore.RED
SHOW_COLOR = Fore.CYAN


SAVE_DATA: bool = False
DEBUG: bool = False

data_dict = {
    "Feature 1": [4, 1, 9, 7, 10, 12, 14, 8],
    "Feature 2": [1, 2, 14, 5, 16, 18, 10, 10],
}
data = pd.DataFrame(data_dict)
if DEBUG:
    print(data)
if SAVE_DATA:
    data.to_csv("data\\knn_data_points.csv", index=False)
    print(LOG_COLOR + "[DATA POINTS FOR KNN SAVED SUCCESSFULLY]")


def euclidean_distance(point1: List[float], point2: List[float]) -> float:
    """
    Calculate the euclidean distance between two points
    """
    assert len(point1) == len(point2), "Points must have the same number of dimensions"
    distance = 0
    for i in range(len(point1)):
        distance += (point1[i] - point2[i]) ** 2
    return np.sqrt(distance)


def calculate_distances(
    data_point: List[float], data_points: List[List[float]]
) -> List[float]:
    """Calculate Euclidean distances between a data point and a list of data points."""
    return [euclidean_distance(data_point, other_point) for other_point in data_points]


def get_top_k_neighbors(
    data_point: List[float], data_points: List[List[float]], k: int
) -> List[Tuple[float, int]]:
    """Get the top k neighbors of a data point."""
    distances = calculate_distances(data_point, data_points)
    # specify that the sorted list should be sorted by the second element of the tuple which is the distance
    all_neighbors = sorted(enumerate(distances), key=lambda x: x[1])
    return all_neighbors[:k]


if __name__ == "__main__":
    OTHER_COLOR = Fore.MAGENTA
    data_point = [11, 3]
    k = 4
    print(OTHER_COLOR + f"[APPLYING THE KNN ALGORITHM FROM SCRATCH]")
    print(SHOW_COLOR + f"Data point: {data_point}")
    print(SHOW_COLOR + f"K: {k}")
    print(OTHER_COLOR + "-" * 60)
    print(SHOW_COLOR + f"The nearest neighbors of the data point are:")
    neighbors = get_top_k_neighbors(data_point, data.values.tolist(), k)
    # print(SHOW_COLOR + f"Top {k} neighbors:  \n {neighbors}")
    for i, d in neighbors:
        print(SHOW_COLOR + f"\t--> Neighbor at index {i}  |  Distance: {round(d, 4)}")
