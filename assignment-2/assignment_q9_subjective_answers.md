# ES654-2020 Assignment 3

*Chris Francis* - *18110041*

------

To create a dataset with multicollinearity,I first created a random dataset and then added 2 extra features which were multiples of existing features. 

Results corresponding to gradient descent on dataset with multicollinearity, obtained from q9_dataset.py

```
Vectorised Gradient Descent, lr_type = constant, fit_intercept = True
RMSE:  0.7915109636391143
MAE:  0.6066149292225439

Non-vectorised Gradient Descent, lr_type = constant, fit_intercept = True
RMSE:  0.7915109636391142
MAE:  0.6066149292225439

Gradient Descent(using Autograd), lr_type = constant, fit_intercept = True
RMSE:  0.7915109636391143
MAE:  0.6066149292225439

Vectorised Gradient Descent, lr_type = inverse, fit_intercept = True
RMSE:  0.8295471832732143
MAE:  0.5865653310379864

Non-vectorised Gradient Descent, lr_type = inverse, fit_intercept = True
RMSE:  0.8295471832732143
MAE:  0.5865653310379864

Gradient Descent(using Autograd), lr_type = inverse, fit_intercept = True
RMSE:  0.8295471832732143
MAE:  0.5865653310379864

Vectorised Gradient Descent, lr_type = constant, fit_intercept = False
RMSE:  0.79178498474082
MAE:  0.6055229777362922

Non-vectorised Gradient Descent, lr_type = constant, fit_intercept = False
RMSE:  0.79178498474082
MAE:  0.605522977736292

Gradient Descent(using Autograd), lr_type = constant, fit_intercept = False
RMSE:  0.79178498474082
MAE:  0.605522977736292

Vectorised Gradient Descent, lr_type = inverse, fit_intercept = False
RMSE:  0.8295554721866657
MAE:  0.5870319657414721

Non-vectorised Gradient Descent, lr_type = inverse, fit_intercept = False
RMSE:  0.8295554721866657
MAE:  0.5870319657414721

Gradient Descent(using Autograd), lr_type = inverse, fit_intercept = False
RMSE:  0.8295554721866657
MAE:  0.5870319657414721
```
