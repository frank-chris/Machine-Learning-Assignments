
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from tree.base import DecisionTree
from metrics import *

np.random.seed(42)
num_average_time = 100

# variables to control how M and N are varied
SAMPLE_STEP = 100
MAX_SAMPLE_COUNT = 1100
FEATURE_STEP = 2
MAX_FEATURE_COUNT = 22


def plot_data(N_vs_time, M_vs_time, case):
    '''
    Function to create plots
    Inputs:
    > N_vs_time: pd.DataFrame
    > M_vs_time: pd.DataFrame
    > case: str
    Returns:
    none
    '''
    N_vs_time['x'] = pd.Series([i for i in range(SAMPLE_STEP, MAX_SAMPLE_COUNT, SAMPLE_STEP)])
    M_vs_time['x'] = pd.Series([i for i in range(FEATURE_STEP, MAX_FEATURE_COUNT, FEATURE_STEP)])

    N_vs_time[['x']+['Learning: M = '+str(i) for i in [FEATURE_STEP*3, FEATURE_STEP*4, FEATURE_STEP*5]]].plot(x='x', kind='line')

    plt.title('Learning Time(s) vs N (Number of Samples) - '+str(case))
    plt.xlabel('N (Number of Samples)')
    plt.ylabel('Time(s)')
    plt.show()

    N_vs_time[['x']+['Prediction: M = '+str(i) for i in [FEATURE_STEP*3, FEATURE_STEP*4, FEATURE_STEP*5]]].plot(x='x', kind='line')

    plt.title('Prediction Time(s) vs N (Number of Samples) - '+str(case))
    plt.xlabel('N (Number of Samples)')
    plt.ylabel('Time(s)')
    plt.show()

    M_vs_time[['x']+['Learning: N = '+str(i) for i in [SAMPLE_STEP*3, SAMPLE_STEP*4, SAMPLE_STEP*5]]].plot(x='x', kind='line')

    plt.title('Learning Time(s) vs M (Number of Features) - '+str(case))
    plt.xlabel('M (Number of Features)')
    plt.ylabel('Time(s)')
    plt.show()

    M_vs_time[['x']+['Prediction: N = '+str(i) for i in [SAMPLE_STEP*3, SAMPLE_STEP*4, SAMPLE_STEP*5]]].plot(x='x', kind='line')

    plt.title('Prediction Time(s) vs M (Number of Features) - '+str(case))
    plt.xlabel('M (Number of Features)')
    plt.ylabel('Time(s)')
    plt.show()


N_vs_time = pd.DataFrame(np.zeros(shape=(int((MAX_SAMPLE_COUNT-SAMPLE_STEP)/SAMPLE_STEP), 6)))
M_vs_time = pd.DataFrame(np.zeros(shape=(int((MAX_FEATURE_COUNT-FEATURE_STEP)/FEATURE_STEP), 6)))

N_vs_time.columns = ['Learning: M = '+str(i) for i in [FEATURE_STEP*3, FEATURE_STEP*4, FEATURE_STEP*5]] + ['Prediction: M = '+str(i) for i in [FEATURE_STEP*3, FEATURE_STEP*4, FEATURE_STEP*5]]
M_vs_time.columns = ['Learning: N = '+str(i) for i in [SAMPLE_STEP*3, SAMPLE_STEP*4, SAMPLE_STEP*5]] + ['Prediction: N = '+str(i) for i in [SAMPLE_STEP*3, SAMPLE_STEP*4, SAMPLE_STEP*5]]

# Test case 1
# Real Input and Real Output
print('Real Input and Real Output:')

print('\nVarying N')
j = 0
for M in [FEATURE_STEP*3, FEATURE_STEP*4, FEATURE_STEP*5]:
    i = 0
    print('M =', M)
    for N in range(SAMPLE_STEP, MAX_SAMPLE_COUNT, SAMPLE_STEP):
        print('N =', N)
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randn(N))
        tree = DecisionTree(criterion='information_gain', max_depth=6)
        start = time.time()
        tree.fit(X, y)
        end = time.time()
        N_vs_time.iloc[i, j] = end - start
        start = time.time()
        y_hat = tree.predict(X)
        end = time.time()
        N_vs_time.iloc[i, j+3] = end - start
        i += 1
    j += 1

print('\nVarying M')
j = 0
for N in [SAMPLE_STEP*3, SAMPLE_STEP*4, SAMPLE_STEP*5]:
    i = 0
    print('N =', N)
    for M in range(FEATURE_STEP, MAX_FEATURE_COUNT, FEATURE_STEP):
        print('M =', M)
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randn(N))
        tree = DecisionTree(criterion='information_gain', max_depth=6)
        start = time.time()
        tree.fit(X, y)
        end = time.time()
        M_vs_time.iloc[i, j] = end - start
        start = time.time()
        y_hat = tree.predict(X)
        end = time.time()
        M_vs_time.iloc[i, j+3] = end - start
        i += 1
    j += 1

plot_data(N_vs_time, M_vs_time, 'RIRO')

# Test case 2
# Real Input and Discrete Output

print('Real Input and Discrete Output:')

print('\nVarying N')
j = 0
for M in [FEATURE_STEP*3, FEATURE_STEP*4, FEATURE_STEP*5]:
    i = 0
    print('M =', M)
    for N in range(SAMPLE_STEP, MAX_SAMPLE_COUNT, SAMPLE_STEP):
        print('N =', N)
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randint(M, size = N), dtype="category")
        tree = DecisionTree(criterion='information_gain', max_depth=6)
        start = time.time()
        tree.fit(X, y)
        end = time.time()
        N_vs_time.iloc[i, j] = end - start
        start = time.time()
        y_hat = tree.predict(X)
        end = time.time()
        N_vs_time.iloc[i, j+3] = end - start
        i += 1
    j += 1

print('\nVarying M')
j = 0
for N in [SAMPLE_STEP*3, SAMPLE_STEP*4, SAMPLE_STEP*5]:
    i = 0
    print('N =', N)
    for M in range(FEATURE_STEP, MAX_FEATURE_COUNT, FEATURE_STEP):
        print('M =', M)
        X = pd.DataFrame(np.random.randn(N, M))
        y = pd.Series(np.random.randint(M, size = N), dtype="category")
        tree = DecisionTree(criterion='information_gain', max_depth=6)
        start = time.time()
        tree.fit(X, y)
        end = time.time()
        M_vs_time.iloc[i, j] = end - start
        start = time.time()
        y_hat = tree.predict(X)
        end = time.time()
        M_vs_time.iloc[i, j+3] = end - start
        i += 1
    j += 1

plot_data(N_vs_time, M_vs_time, 'RIDO')

# Test case 3
# Discrete Input and Discrete Output

print('Discrete Input and Discrete Output:')

print('\nVarying N')
j = 0
for M in [FEATURE_STEP*3, FEATURE_STEP*4, FEATURE_STEP*5]:
    i = 0
    print('M =', M)
    for N in range(SAMPLE_STEP, MAX_SAMPLE_COUNT, SAMPLE_STEP):
        print('N =', N)
        X = pd.DataFrame({i:pd.Series(np.random.randint(M, size = N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randint(M, size = N), dtype="category")
        tree = DecisionTree(criterion='information_gain', max_depth=6)
        start = time.time()
        tree.fit(X, y)
        end = time.time()
        N_vs_time.iloc[i, j] = end - start
        start = time.time()
        y_hat = tree.predict(X)
        end = time.time()
        N_vs_time.iloc[i, j+3] = end - start
        i += 1
    j += 1

print('\nVarying M')
j = 0
for N in [SAMPLE_STEP*3, SAMPLE_STEP*4, SAMPLE_STEP*5]:
    i = 0
    print('N =', N)
    for M in range(FEATURE_STEP, MAX_FEATURE_COUNT, FEATURE_STEP):
        print('M =', M)
        X = pd.DataFrame({i:pd.Series(np.random.randint(M, size = N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randint(M, size = N), dtype="category")
        tree = DecisionTree(criterion='information_gain', max_depth=6)
        start = time.time()
        tree.fit(X, y)
        end = time.time()
        M_vs_time.iloc[i, j] = end - start
        start = time.time()
        y_hat = tree.predict(X)
        end = time.time()
        M_vs_time.iloc[i, j+3] = end - start
        i += 1
    j += 1

plot_data(N_vs_time, M_vs_time, 'DIDO')

# Test case 4
# Discrete Input and Real Output

print('Discrete Input and Real Output:')

print('\nVarying N')
j = 0
for M in [FEATURE_STEP*3, FEATURE_STEP*4, FEATURE_STEP*5]:
    i = 0
    print('M =', M)
    for N in range(SAMPLE_STEP, MAX_SAMPLE_COUNT, SAMPLE_STEP):
        print('N =', N)
        X = pd.DataFrame({i:pd.Series(np.random.randint(M, size = N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randn(N))
        tree = DecisionTree(criterion='information_gain', max_depth=6)
        start = time.time()
        tree.fit(X, y)
        end = time.time()
        N_vs_time.iloc[i, j] = end - start
        start = time.time()
        y_hat = tree.predict(X)
        end = time.time()
        N_vs_time.iloc[i, j+3] = end - start
        i += 1
    j += 1

print('\nVarying M')
j = 0
for N in [SAMPLE_STEP*3, SAMPLE_STEP*4, SAMPLE_STEP*5]:
    i = 0
    print('N =', N)
    for M in range(FEATURE_STEP, MAX_FEATURE_COUNT, FEATURE_STEP):
        print('M =', M)
        X = pd.DataFrame({i:pd.Series(np.random.randint(M, size = N), dtype="category") for i in range(M)})
        y = pd.Series(np.random.randn(N))
        tree = DecisionTree(criterion='information_gain', max_depth=6)
        start = time.time()
        tree.fit(X, y)
        end = time.time()
        M_vs_time.iloc[i, j] = end - start
        start = time.time()
        y_hat = tree.predict(X)
        end = time.time()
        M_vs_time.iloc[i, j+3] = end - start
        i += 1
    j += 1

plot_data(N_vs_time, M_vs_time, 'DIRO')