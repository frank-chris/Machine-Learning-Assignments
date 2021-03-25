import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linearRegression import LinearRegression

x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = 4*x + 7 + np.random.normal(0,3,len(x))

max_degree = 10

degree_list = [i for i in range(1, max_degree, 2)]

colors = ['#636EFA', '#EF553B', '#00CC96', '#AB63FA', '#FFA15A', '#19D3F3']

# varying N
for N, color in zip([10, 20, 30, 40, 50, 60], colors):
    theta_list = []

    # varying degree
    for degree in degree_list:
        poly = PolynomialFeatures(degree)
        LR = LinearRegression(fit_intercept=False)
        X = pd.DataFrame(poly.transform(x[:N].reshape((N, 1))))
        LR.fit_vectorised(X, pd.Series(y[:N]), N, n_iter=10, lr = 0.001)
        theta_list.append(np.linalg.norm(LR.coef_, ord=np.inf))

    # plotting
    plt.plot(degree_list, theta_list, label = 'N = '+str(N), color=color)
    plt.title(r'$|\theta|_\infty =$ max$_{i}$ |$\theta_i$| vs Degree, d')
    plt.xlabel('Degree, d')
    plt.ylabel(r'$|\theta|_\infty =$ max$_{i}$ |$\theta_i$| (log scale)')
    plt.yscale('log')
    plt.legend()

plt.savefig('images/q6_plot.png')
plt.show()