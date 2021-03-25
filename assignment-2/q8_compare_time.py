import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from linearRegression.linearRegression import LinearRegression
import time

np.random.seed(42)

N = 1000
P = 500
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

vectorised_time = []
normal_time = []
predict_time = []
N_list = [i for i in range(10, N+10, 10)]

LR = LinearRegression(fit_intercept=True)

# Varying N, number of samples
for n in N_list:   
    elapsed_time = time.time()
    LR.fit_vectorised(X[X.columns[0:100]][:n], y[:n], batch_size=n)
    elapsed_time = time.time() - elapsed_time
    vectorised_time.append(elapsed_time)

    elapsed_time = time.time()
    LR.fit_normal(X[X.columns[0:100]][:n], y[:n])
    elapsed_time = time.time() - elapsed_time
    normal_time.append(elapsed_time)

    elapsed_time = time.time()
    y_hat = LR.predict(X[X.columns[0:100]][:n])
    elapsed_time = time.time() - elapsed_time
    predict_time.append(elapsed_time)

plt.plot(N_list, vectorised_time, label = 'Gradient Descent Vectorised-Fit', color='#636EFA')
plt.plot(N_list, normal_time, label = 'Normal Equation-Fit', color='#EF553B')
plt.plot(N_list, predict_time, label = 'Prediction Time', color='#00CC96')
plt.title('Time(s) vs Number of samples, N')
plt.xlabel('Number of samples, N')
plt.ylabel('Time(s)')
plt.legend()

plt.savefig('images/q8_plot_1.png')
plt.show()

vectorised_time = []
normal_time = []
predict_time = []
P_list = [i for i in range(2, P+2, 2)]

# Varying P, number of features
for p in P_list:
    elapsed_time = time.time()
    LR.fit_vectorised(X[X.columns[0:p]], y, batch_size=N)
    elapsed_time = time.time() - elapsed_time
    vectorised_time.append(elapsed_time)

    elapsed_time = time.time()
    LR.fit_normal(X[X.columns[0:p]], y)
    elapsed_time = time.time() - elapsed_time
    normal_time.append(elapsed_time)

    elapsed_time = time.time()
    y_hat = LR.predict(X[X.columns[0:p]])
    elapsed_time = time.time() - elapsed_time
    predict_time.append(elapsed_time)

plt.plot(P_list, vectorised_time, label = 'Gradient Descent Vectorised-Fit', color='#636EFA')
plt.plot(P_list, normal_time, label = 'Normal Equation-Fit', color='#EF553B')
plt.plot(P_list, predict_time, label = 'Prediction Time', color='#00CC96')
plt.title('Time(s) vs Number of features, P')
plt.xlabel('Number of features, P')
plt.ylabel('Time(s)')
plt.legend()

plt.savefig('images/q8_plot_2.png')
plt.show()

vectorised_time = []
normal_time = []
predict_time = []
t_list = [i for i in range(1, 501, 1)]

# Varying t, number of iterations
for t in t_list:
    elapsed_time = time.time()
    LR.fit_vectorised(X[X.columns[0:50]], y, batch_size=N, n_iter=t)
    elapsed_time = time.time() - elapsed_time
    vectorised_time.append(elapsed_time)

    elapsed_time = time.time()
    LR.fit_normal(X[X.columns[0:50]], y)
    elapsed_time = time.time() - elapsed_time
    normal_time.append(elapsed_time)

    elapsed_time = time.time()
    y_hat = LR.predict(X[X.columns[0:50]])
    elapsed_time = time.time() - elapsed_time
    predict_time.append(elapsed_time)

plt.plot(t_list, vectorised_time, label = 'Gradient Descent Vectorised-Fit', color='#636EFA')
plt.plot(t_list, normal_time, label = 'Normal Equation-Fit', color='#EF553B')
plt.plot(t_list, predict_time, label = 'Prediction Time', color='#00CC96')
plt.title('Time(s) vs Number of iterations, t')
plt.xlabel('Number of iterations, t')
plt.ylabel('Time(s)')
plt.legend()

plt.savefig('images/q8_plot_3.png')
plt.show()