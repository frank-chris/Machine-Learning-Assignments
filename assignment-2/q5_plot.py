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

degree_list = [i for i in range(1, max_degree+1)]

theta_list = []

# varying degree
for degree in degree_list:
    poly = PolynomialFeatures(degree)
    LR = LinearRegression(fit_intercept=False)
    X = pd.DataFrame(poly.transform(x.reshape((x.shape[0], 1))))
    LR.fit_vectorised(X, pd.Series(y), x.shape[0], n_iter=10, lr = 0.001)
    theta_list.append(np.linalg.norm(LR.coef_, ord=np.inf))

# plotting
plt.plot(degree_list, theta_list, color='#636EFA')
plt.title(r'$|\theta|_\infty =$ max$_{i}$ |$\theta_i$| vs Degree, d')
plt.xlabel('Degree, d')
plt.ylabel(r'$|\theta|_\infty =$ max$_{i}$ |$\theta_i$| (log scale)')
plt.yscale('log')
plt.savefig('images/q5_plot.png')
plt.show()