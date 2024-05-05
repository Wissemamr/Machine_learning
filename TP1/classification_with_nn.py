from typing import List

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch import nn as nn

"""
Predicting covid test results with neural networks
BUilding a dense neural network with pytorch and training it to 
predict covid test results
"""


DEBUG: bool = False


df = pd.read_csv("dataset\\processed_covid_df.csv")
if DEBUG:
    print(f"The size of the loaded df {df.shape}")


# define the features
features = [
    "Patient age quantile",
    "Patient addmited to regular ward (1=yes, 0=no)",
    "Patient addmited to semi-intensive unit (1=yes, 0=no)",
    "Patient addmited to intensive care unit (1=yes, 0=no)",
]


# define the target variable
target = "SARS-Cov-2 exam result"


def load_and_split_data(
    df: pd.DataFrame, features: List[str], target: str
) -> np.ndarray:
    # Extract features and target
    X = df[features]
    y = df[target]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test


def standardize_data(X_train: torch.Tensor, X_test: torch.Tensor) -> np.ndarray:
    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test


X_train, X_test, y_train, y_test = load_and_split_data(
    df=df, features=features, target=target
)
X_train, X_test = standardize_data(X_train=X_train, X_test=X_test)
y_train = np.array(y_train)
y_test = np.array(y_test)


# convert np_array to  torch tensors
X_train = torch.as_tensor(X_train, dtype=torch.float32)
X_test = torch.as_tensor(X_test, dtype=torch.float32)
y_train = torch.as_tensor(y_train, dtype=torch.float32)
y_test = torch.as_tensor(y_test, dtype=torch.float32)

if DEBUG:
    print(f" the shape of X_train : {X_train.shape} ")
    print(f" the shape of X_test : {X_test.shape} ")
    print(f" the shape of y_train : {y_train.shape}")
    print(f" the shape of y_test : {y_test.shape} \n")


# make the code device agnostic
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define the classifier class
class Covid_results_classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # . Create 2 nn.Linear layers capable of handling X and y input and output shapes
        self.layer_1 = nn.Linear(
            in_features=4, out_features=5
        )  # takes in 2 features (X), produces 5 features
        self.layer_2 = nn.Linear(
            in_features=5, out_features=1
        )  # takes in 5 features, produces 1 feature (y)

    def forward(self, x):
        return self.layer_2(self.layer_1(x.float()))


my_model = Covid_results_classifier()

criterion = nn.BCEWithLogitsLoss()  # It has sigmoid built in
optimizer = torch.optim.SGD(params=my_model.parameters(), lr=0.1)


def accuracy_metric(y_true, y_pred):
    if len(y_true) != len(y_pred):
        raise ValueError("the predicted and true labels must have the same length")
    # torch.eq() calculates where two tensors are equal
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


torch.manual_seed(42)

# Set the number of epochs
epochs = 500

# Put data to target device
X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

train_losses = []
test_losses = []
# Build training and evaluation loop
for epoch in range(epochs):
    # Training
    my_model.train()

    # 1. Forward pass (model outputs raw logits)
    y_logits = my_model(
        X_train
    ).squeeze()  # squeeze to remove extra `1` dimensions, this won't work unless model and data are on same device
    y_pred = torch.round(
        torch.sigmoid(y_logits)
    )  # turn logits -> pred probs -> pred labels

    # 2. Calculate loss/accuracy
    # loss = loss_fn(torch.sigmoid(y_logits), # Using nn.BCELoss you need torch.sigmoid()
    #                y_train)
    loss = criterion(
        y_logits, y_train  # Using nn.BCEWithLogitsLoss works with raw logits
    )
    train_losses.append(loss)
    acc = accuracy_metric(y_true=y_train, y_pred=y_pred)

    # 3. Optimizer zero grad
    optimizer.zero_grad()

    # 4. Loss backwards
    loss.backward()

    # 5. Optimizer step
    optimizer.step()

    ### Testing
    my_model.eval()
    with torch.inference_mode():
        # 1. Forward pass
        test_logits = my_model(X_test).squeeze()
        test_pred = torch.round(torch.sigmoid(test_logits))
        # 2. Caculate loss/accuracy
        test_loss = criterion(test_logits, y_test)
        test_losses.append(test_loss)
        test_acc = accuracy_metric(y_true=y_test, y_pred=test_pred)

    # Print out what's happening every 10 epochs
    if epoch % 10 == 0:
        print(
            f" \tEpoch: {epoch} : \n --> Loss: {loss:.5f}| Accuracy: {acc:.2f}% \n --> Test loss: {test_loss:.5f}| Test acc: {test_acc:.2f}% \n"
        )


train_losses = [x.item() for x in train_losses]
test_losses = [x.item() for x in test_losses]


if DEBUG:
    min_train_loss = np.min(train_losses)
    smax_test_loss = np.max(test_losses)
