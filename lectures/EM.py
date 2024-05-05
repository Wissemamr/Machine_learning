import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import stats

# initial some points along the line
samples = [0, 1, 3, 4, 5, 6, 2, 4.5, 7, 8, 10, 12]
# plot them
plt.rcParams["figure.figsize"] = [10, 5]
y_value = 0
y = np.zeros_like(samples) + y_value
plt.scatter(samples, y, ls="dotted")
plt.show()


# first initial
mu1 = 0
sigma1 = 1
mu2 = 1
sigma2 = 1
# sample 100000 times from this two initial normal distribution
samples_1 = np.random.normal(mu1, sigma1, 100000)
samples_2 = np.random.normal(mu2, sigma2, 100000)
# get pdf from the samples
y_normal_1 = scipy.stats.norm(mu1, sigma1).pdf(np.sort(samples_1))
y_normal_2 = scipy.stats.norm(mu2, sigma2).pdf(np.sort(samples_2))
# plot
plt.rcParams["figure.figsize"] = [10, 5]
y_value = 0
y = np.zeros_like(samples) + y_value
plt.scatter(samples, y, ls="dotted")
plt.plot(np.sort(samples_1), y_normal_1, c="purple")
plt.plot(np.sort(samples_2), y_normal_2, c="yellow")
# get likelihood for each x from gaussian_1 and gaussian_0
# get likelihood of x in gaussian_1
likelihood_x_gaussian_1 = [
    1 / np.sqrt(2 * 3.14) * np.exp((-1 / 2) * (xi**2)) for xi in samples
]
# get likelihood of x in gaussian_0
likelihood_x_gaussian_0 = [
    1 / np.sqrt(2 * 3.14) * np.exp((-1 / 2) * ((xi - 1) ** 2)) for xi in samples
]
# create a dataframe
df_likelihood = pd.DataFrame(
    [samples, likelihood_x_gaussian_1, likelihood_x_gaussian_0]
).transpose()
df_likelihood.columns = ["samples", "p(X=x|class0)", "p(X=x|class1)"]
print(df_likelihood)


# get likelihood of class0 and class1 occurs given samples
# define prior
prior_class_0 = 0.5
prior_class_1 = 0.5
# get likelihood of gaussian_1 occurs based on x1
df_likelihood["p(class0|x)"] = df_likelihood.apply(
    lambda row: row["p(X=x|class0)"]
    * 0.5
    / (row["p(X=x|class1)"] * 0.5 + row["p(X=x|class0)"] * 0.5),
    axis=1,
)
# get likelihood of gaussian_0 occurs based on x1
df_likelihood["p(class1|x)"] = df_likelihood.apply(
    lambda row: row["p(X=x|class1)"]
    * 0.5
    / (row["p(X=x|class1)"] * 0.5 + row["p(X=x|class0)"] * 0.5),
    axis=1,
)
print(df_likelihood)

# standardization of p(Gaussian_1|x) and p(Gaussian_0|x)
df_likelihood["p(class0|x)_standardized"] = df_likelihood["p(class0|x)"] / sum(
    df_likelihood["p(class0|x)"]
)
df_likelihood["p(class1|x)_standardized"] = df_likelihood["p(class1|x)"] / sum(
    df_likelihood["p(class1|x)"]
)
print(df_likelihood)


# update mu1, mu2, sigma1, sigma2
mu1_new = sum(df_likelihood["samples"] * df_likelihood["p(class0|x)_standardized"])
print(mu1_new)
variance1_new = sum(
    (df_likelihood["samples"] - mu1_new) ** 2
    * df_likelihood["p(class0|x)_standardized"]
)
print(variance1_new)
sigma1_new = np.sqrt(variance1_new / len(samples))
print(sigma1_new)
mu2_new = sum(df_likelihood["samples"] * df_likelihood["p(class1|x)_standardized"])
print(mu2_new)
variance2_new = sum(
    (df_likelihood["samples"] - mu2_new) ** 2
    * df_likelihood["p(class1|x)_standardized"]
)
print(variance2_new)
sigma2_new = np.sqrt(variance2_new / len(samples))
print(sigma2_new)


# finish 1st round
# generate class label if we need a hard assignment
p0 = [scipy.stats.norm(mu1_new, sigma1_new).pdf(x) for x in samples]
p1 = [scipy.stats.norm(mu2_new, sigma2_new).pdf(x) for x in samples]
class_pred = []
for x, y in zip(p0, p1):
    if x < y:
        class_pred_i = 1
    else:
        class_pred_i = 0
    class_pred.append(class_pred_i)
# sample 100000 times from this two updated normal distribution from initial
samples_1 = np.random.normal(mu1_new, sigma1_new, 100000)
samples_2 = np.random.normal(mu2_new, sigma2_new, 100000)
# get pdf from the samples
y_normal_1 = scipy.stats.norm(mu1_new, sigma1_new).pdf(np.sort(samples_1))
y_normal_2 = scipy.stats.norm(mu2_new, sigma2_new).pdf(np.sort(samples_2))
# plot
plt.rcParams["figure.figsize"] = [10, 5]
y_value = 0
y = np.zeros_like(samples) + y_value
plt.scatter(samples, y, ls="dotted", c=class_pred)
plt.plot(np.sort(samples_1), y_normal_1, c="purple")
plt.plot(np.sort(samples_2), y_normal_2, c="yellow")
