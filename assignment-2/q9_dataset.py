import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linearRegression.linearRegression import LinearRegression
from metrics import rmse, mae

np.random.seed(42)

# creating dataset with multicollinearity
N = 30
P = 3
X = pd.DataFrame(np.random.randn(N, P))
X = pd.concat([X, 2*X[2], 3*X[1]], axis=1)
y = pd.Series(np.random.randn(N))


for fit_intercept in [True, False]:
    LR = LinearRegression(fit_intercept=fit_intercept)
    
    for lr_type in ['constant', 'inverse']:
        print('\nVectorised Gradient Descent, lr_type = '+lr_type+', fit_intercept = '+str(fit_intercept))
        LR.fit_vectorised(X, y, batch_size=10, lr_type=lr_type)
        y_hat = LR.predict(X)
        print('RMSE: ', rmse(y_hat, y))
        print('MAE: ', mae(y_hat, y))

        print('\nNon-vectorised Gradient Descent, lr_type = '+lr_type+', fit_intercept = '+str(fit_intercept))
        LR.fit_non_vectorised(X, y, batch_size=10, lr_type=lr_type)
        y_hat = LR.predict(X)
        print('RMSE: ', rmse(y_hat, y))
        print('MAE: ', mae(y_hat, y))

        print('\nGradient Descent(using Autograd), lr_type = '+lr_type+', fit_intercept = '+str(fit_intercept))
        LR.fit_autograd(X, y, batch_size=10, lr_type=lr_type)
        y_hat = LR.predict(X)
        print('RMSE: ', rmse(y_hat, y))
        print('MAE: ', mae(y_hat, y))

    print('\nNormal Equation, fit_intercept = '+str(fit_intercept))
    LR.fit_normal(X, y)
    y_hat = LR.predict(X)
    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))
