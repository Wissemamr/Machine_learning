from sklearn.neighbors import NearestNeighbors
from typing import List, Tuple
import numpy as np
import pandas as pd
from colorama import Fore, init
import math

init(autoreset=True)


def get_top_k_neighbors(
    data_point: List[float], data_points: List[List[float]], k: int
) -> List[Tuple[float, int]]:
    """Get the top k neighbors of a data point using scikit-learn's NearestNeighbors."""

    data_points_array = np.array(data_points)
    neighbors_model = NearestNeighbors(n_neighbors=k, algorithm="auto", p=2)
    neighbors_model.fit(data_points_array)
    distances, indices = neighbors_model.kneighbors([data_point])
    distances = distances.flatten()
    indices = indices.flatten()
    neighbors = list(zip(distances, indices))

    return neighbors


if __name__ == "__main__":
    SHOW_COLOR = Fore.GREEN

    data_dict = {
        "Feature 1": [4, 1, 9, 7, 10, 12, 14, 8],
        "Feature 2": [1, 2, 14, 5, 16, 18, 10, 10],
    }
    data = pd.DataFrame(data_dict)

    OTHER_COLOR = Fore.MAGENTA
    data_point = [11, 3]
    k = 4
    print(OTHER_COLOR + f"\n[APPLYING THE KNN ALGORITHM FROM SCRATCH]")
    print(SHOW_COLOR + f"Data point: {data_point}")
    print(SHOW_COLOR + f"K: {k}")
    print(OTHER_COLOR + "-" * 60)
    print(SHOW_COLOR + f"The nearest neighbors of the data point are:")
    neighbors = get_top_k_neighbors(data_point, data.values.tolist(), k)
    # print(SHOW_COLOR + f"Top {k} neighbors:  \n {neighbors}")
    for d, i in neighbors:
        print(
            SHOW_COLOR
            + f"\t--> Neighbor at index {round(i)}  |  Distance: {round(d, 4)}"
        )
