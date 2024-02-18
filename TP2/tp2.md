### TP2 content :

In this lab, we will explore and implement linear regression from scratch, its model, loss function and gradient


| `ID`              | `Matrix representation`   | 
|----               |----                       |
| Model             | $ F = X \theta $          | 
| Loss function     |  $ J(\theta) = \frac{1}{2n} \sum_{i=1}^{n} (X\theta - Y)^2  $ | 
|  Gradient         |  $  \nabla J(\theta) = \frac{1}{m} X^T(X\theta - Y)$ | 
|  Gradient descent |  $ \theta = \theta - \alpha \cdot \frac{1}{m} X^T(X\theta - Y)$ | 



### Approach : 
#### Step01 : Load the data 
We can take for example the hospital data that we want to perform the linear regression on in order to predict the cost of the medical cost. We load the data from a CSV file

#### Step 02 : Preprocess the data
Clean and normalize the data, applyign z-score normalization

$z = \frac{x - \mu}{\sigma} $



#### Step 03 : Split the data into training and testing data
