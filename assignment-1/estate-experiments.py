
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import rmse, mae
from sklearn.tree import DecisionTreeRegressor

np.random.seed(42)

# reading dataset and cleaning
estate_data = pd.read_csv('estate.csv', names=['No', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'Y'], skiprows=1)
estate_data.drop(['No'], axis=1, inplace=True)

# splitting into test and train sets
train_data = estate_data.iloc[:int(0.8*estate_data.index.size)]
test_data = estate_data.iloc[int(0.8*estate_data.index.size):]

# splitting into X and y
train_X = train_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']]
train_y = train_data['Y']

test_X = test_data[['X1', 'X2', 'X3', 'X4', 'X5', 'X6']]
test_y = test_data['Y']

depths = []
rmse_base = []
mae_base = []
rmse_sk = []
mae_sk = []

# varying the max_depth and comparing my implementation 
# with scikit learn implementation on the dataset
for depth in range(1,11):
    depths.append(depth)
    tree = DecisionTree(criterion='information_gain', max_depth=depth) 
    tree.fit(train_X, train_y)
    y_hat = tree.predict(test_X)
    y = test_y.reset_index(drop=True)
    print('------------------------------------------')
    print('My implementation(max_depth='+ str(depth) +'):')
    print('------------------------------------------')
    print('RMSE: ', rmse(y_hat, y))
    print('MAE: ', mae(y_hat, y))
    rmse_base.append(rmse(y_hat, y))
    mae_base.append(mae(y_hat, y))
    
    
    sk_tree = DecisionTreeRegressor(max_depth=depth)
    sk_tree.fit(train_X, train_y)

    y_pred = pd.Series(sk_tree.predict(test_X))

    print('------------------------------------------')
    print('scikit-learn implementation(max_depth='+ str(depth) +'):')
    print('------------------------------------------')
    print('RMSE: ', rmse(y_pred, y))
    print('MAE: ', mae(y_pred, y))
    rmse_sk.append(rmse(y_pred, y))
    mae_sk.append(mae(y_pred, y))

# plotting results
rmse_df = pd.DataFrame({'depth':depths, 'my_implementation':rmse_base, 'scikit_implementation':rmse_sk})
mae_df = pd.DataFrame({'depth':depths, 'my_implementation':mae_base, 'scikit_implementation':mae_sk})

rmse_df.plot(x='depth', kind='line')

plt.title('RMSE vs Max-depth')
plt.xlabel('Max-depth')
plt.ylabel('RMSE')
plt.show()

mae_df.plot(x='depth', kind='line')

plt.title('MAE vs Max-depth')
plt.xlabel('Max-depth')
plt.ylabel('MAE')
plt.show()