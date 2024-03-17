from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from colorama import Fore, init

init(autoreset=True)

LOG_COLOR = Fore.MAGENTA


iris = load_iris()
X, y = iris.data, iris.target

# Create and train KNN model
knn_model = KNeighborsClassifier(n_neighbors=20)
knn_model.fit(X, y)
observations = [[5.4, 3.3, 5.9, 1.1], [1, 1, 1, 1]]


if __name__ == "__main__":
    predicted_classes = knn_model.predict(observations)
    for i, prediction in enumerate(predicted_classes):
        print(
            f"Observation {i + 1}: Predicted Class --> {LOG_COLOR + iris.target_names[prediction]}"
        )
