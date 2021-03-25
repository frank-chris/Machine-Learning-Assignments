import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
sns.set()
# Import Autograd modules here
from autograd import grad
import autograd.numpy as anp

def cost_func(theta, X, y):
    error = np.matmul(X,theta)-y
    return (1/X.shape[0])*np.matmul(error.T,error)

def RSS(theta, X, y):
    error = np.matmul(X,theta)-y
    return np.matmul(error.T,error)

def dot(x, y):
    dot_product = 0 
    for i in range(len(x)):
        dot_product += x[i]*y[i]
    return dot_product


class LinearRegression():
    def __init__(self, fit_intercept=True):
        '''
        :param fit_intercept: Whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (i.e. data is expected to be centered).
        '''
        self.fit_intercept = fit_intercept
        self.coef_ = None #Replace with numpy array or pandas series of coefficients learned using using the fit methods


    def fit_non_vectorised(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using non-vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data. 
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        assert(X.index.size == y.size)

        if self.fit_intercept:
            X = np.concatenate((np.ones((X.shape[0], 1)), np.array(X)), axis=1)
        else:
            X = np.array(X)

        y = np.array(y)

        n, m = X.shape

        self.coef_ = np.zeros(m)

        i = 0
        for iteration in range(0, n_iter):
            if lr_type == 'inverse':
                alpha = lr/(iteration+1)
            else:
                alpha = lr

            gradient = np.zeros(m)
            X_i = X[i:i+batch_size]
            y_i = y[i:i+batch_size]
            batch_n = X_i.shape[0]

            for j in range(batch_n):
                y_pred = dot(X_i[j], self.coef_)
                for k in range(m):
                    gradient[k] -= (y_i[j]-y_pred)*X_i[j][k]
            gradient = gradient * (2/batch_n)

            self.coef_ = self.coef_ - alpha*gradient
            if i+batch_size >= n:
                i = 0
            else:
                i += batch_size


    def fit_vectorised(self, X, y,batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using vectorised gradient descent.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        assert(X.index.size == y.size)

        if self.fit_intercept:
            X = np.concatenate((np.ones((X.shape[0], 1)), np.array(X)), axis=1)
        else:
            X = np.array(X)

        y = np.array(y)

        n, m = X.shape

        self.coef_ = np.zeros(m)

        i = 0
        for iteration in range(0, n_iter):
            if lr_type == 'inverse':
                alpha = lr/(iteration+1)
            else:
                alpha = lr

            X_i = X[i:i+batch_size]
            y_i = y[i:i+batch_size]
            batch_n = X_i.shape[0]

            self.coef_ -= (2*alpha/batch_n)*np.matmul(X_i.T,(np.matmul(X_i,self.coef_)-y_i))
            if i+batch_size >= n:
                i = 0
            else:
                i += batch_size

    def fit_autograd(self, X, y, batch_size, n_iter=100, lr=0.01, lr_type='constant'):
        '''
        Function to train model using gradient descent with Autograd to compute the gradients.
        Autograd reference: https://github.com/HIPS/autograd

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param batch_size: int specifying the  batch size. Batch size can only be between 1 and number of samples in data.
        :param n_iter: number of iterations (default: 100)
        :param lr: learning rate (default: 0.01)
        :param lr_type: If lr_type = 'constant', then the learning rate remains constant,
                        if lr_type = 'inverse', then learning rate = lr / t, where t = current iteration number

        :return None
        '''

        assert(X.index.size == y.size)

        if self.fit_intercept:
            X = np.concatenate((np.ones((X.shape[0], 1)), np.array(X)), axis=1)
        else:
            X = np.array(X)

        y = np.array(y)

        n, m = X.shape

        self.t_0 = [0]
        self.t_1 = [0]
        if m == 2:
            self.RSS = [RSS(np.array([0, 0]), X, y)]

        self.coef_ = np.zeros(m)

        grad_func = grad(cost_func, 0)

        i = 0
        for iteration in range(0, n_iter):
            if lr_type == 'inverse':
                alpha = lr/(iteration+1)
            else:
                alpha = lr

            X_i = X[i:i+batch_size]
            y_i = y[i:i+batch_size]

            self.coef_ -= alpha*grad_func(self.coef_, X_i, y_i)
            self.t_0.append(self.coef_[0])
            self.t_1.append(self.coef_[1])
            if m == 2:
                self.RSS.append(RSS(self.coef_, X, y))

            if i+batch_size >= n:
                i = 0
            else:
                i += batch_size

    def fit_normal(self, X, y):
        '''
        Function to train model using the normal equation method.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))

        :return None
        '''

        assert(X.index.size == y.size)

        if self.fit_intercept:
            X = np.concatenate((np.ones((X.shape[0], 1)), np.array(X)), axis=1)
        else:
            X = np.array(X)

        y = np.array(y)

        XT_X_inverse = np.linalg.pinv(np.matmul(X.T, X))
        XT_y = np.matmul(X.T, y)

        self.coef_ = np.matmul(XT_X_inverse, XT_y)



    def predict(self, X):
        '''
        Funtion to run the LinearRegression on a data point

        :param X: pd.DataFrame with rows as samples and columns as features

        :return: y: pd.Series with rows corresponding to output variable. The output variable in a row is the prediction for sample in corresponding row in X.
        '''

        if self.fit_intercept:
            X = np.concatenate((np.ones((X.shape[0], 1)), np.array(X)), axis=1)
        else:
            X = np.array(X)

        y_pred = np.matmul(X, self.coef_)
        y_pred = y_pred.reshape(y_pred.shape[0])
        return pd.Series(y_pred)

    def plot_surface(self, X, y, t_0=None, t_1=None):
        """
        Function to plot RSS (residual sum of squares) in 3D. A surface plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1 by a
        red dot. Uses self.coef_ to calculate RSS. Plot must indicate error as the title.

        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to indicate RSS
        :param t_1: Value of theta_1 for which to indicate RSS

        :return matplotlib figure plotting RSS
        """

        assert(X.index.size == y.size)

        if t_0 == None:
            t_0 = self.t_0
        if t_1 == None:
            t_1 = self.t_1

        if self.fit_intercept:
            X = np.concatenate((np.ones((X.shape[0], 1)), np.array(X)), axis=1)
        else:
            X = np.array(X)

        y = np.array(y)

        theta_0, theta_1 = np.meshgrid(np.linspace(0,10,100),np.linspace(0,10,100))
        z = np.array([RSS(np.array([t0, t1]), X, y) for t0, t1 in zip(np.ravel(theta_0), np.ravel(theta_1))])
        z = z.reshape(theta_0.shape)

        fig = plt.figure(figsize = (15, 10))
        ax = Axes3D(fig)
        ax.plot_surface(theta_0, theta_1, z, rstride = 5, cstride = 5, cmap = 'plasma', alpha=0.5)
        ax.set_xlabel(r'$\theta_0$')
        ax.set_ylabel(r'$\theta_1$')
        ax.set_zlabel('RSS')
        ax.view_init(45, -45)
        ax.set_facecolor('#FFFFFF')

        line, = ax.plot([], [], label = 'Gradient descent', color='#00CC96', lw = 1.5)
        point, = ax.plot([], [], "^", color = '#00CC96', markersize = 5)

        def animate(i):
            # animate function for animation
            iteration = 'Iteration ' + str(i)
            theta_values = '\n'+r'$\theta_1$ = '+str(t_1[i])+',  '+r'$\theta_0$ = '+str(t_0[i])
            line.set_data(np.array(t_0[:i+1]), np.array(t_1[:i+1]))
            line.set_3d_properties(np.array(self.RSS[:i+1]))
            fig.suptitle(iteration+'  RSS:'+str(self.RSS[i])+theta_values)
            point.set_data(np.array(t_0[i]), np.array(t_1[i]))
            point.set_3d_properties(np.array(self.RSS[i]))
            return line, point

        ax.legend(loc ="lower right")
        anim = FuncAnimation(fig, animate, frames=np.arange(0, len(t_0)), interval=800, blit=True)
        return anim

    def plot_line_fit(self, X, y, t_0=None, t_1=None):
        """
        Function to plot fit of the line (y vs. X plot) based on chosen value of t_0, t_1. Plot must
        indicate t_0 and t_1 as the title.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting line fit
        """
        assert(X.index.size == y.size)

        if t_0 == None:
            t_0 = self.t_0
        if t_1 == None:
            t_1 = self.t_1

        X = np.array(X).reshape(X.index.size)
        y = np.array(y).reshape(y.size)

        fig, ax = plt.subplots(figsize=(15, 10))
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.scatter(X, y, color='#636EFA', alpha=0.9, s=12)
        line, = ax.plot(X, X*0+0, color='#EF553B', alpha=0.9,  linewidth=2)

        def animate(i):
            # animate function for animation
            iteration = 'Iteration ' + str(i) + '\n'
            line.set_ydata(X*t_1[i]+t_0[i])
            ax.set_title(iteration+r'$\theta_1$ = '+str(t_1[i])+',  '+r'$\theta_0$ = '+str(t_0[i]))
            return line, ax

        anim = FuncAnimation(fig, animate, frames=np.arange(0, len(t_0)), interval=800)
        return anim

    def plot_contour(self, X, y, t_0=None, t_1=None):
        """
        Plots the RSS as a contour plot. A contour plot is obtained by varying
        theta_0 and theta_1 over a range. Indicates the RSS based on given value of t_0 and t_1, and the
        direction of gradient steps. Uses self.coef_ to calculate RSS.

        :param X: pd.DataFrame with rows as samples and columns as features (shape: (n_samples, n_features))
        :param y: pd.Series with rows corresponding to output (shape: (n_samples,))
        :param t_0: Value of theta_0 for which to plot the fit
        :param t_1: Value of theta_1 for which to plot the fit

        :return matplotlib figure plotting the contour
        """

        assert(X.index.size == y.size)

        if t_0 == None:
            t_0 = self.t_0
        if t_1 == None:
            t_1 = self.t_1

        if self.fit_intercept:
            X = np.concatenate((np.ones((X.shape[0], 1)), np.array(X)), axis=1)
        else:
            X = np.array(X)

        y = np.array(y)

        theta_0, theta_1 = np.meshgrid(np.linspace(-60,60,120),np.linspace(-7.5,17.5,125))
        z = np.array([RSS(np.array([t0, t1]), X, y) for t0, t1 in zip(np.ravel(theta_0), np.ravel(theta_1))])
        z = z.reshape(theta_0.shape)

        fig, ax = plt.subplots(figsize = (15, 10))
        ax.contour(theta_0, theta_1, z, 100, cmap='plasma', alpha=0.8)
        ax.set_xlabel(r'$\theta_0$')
        ax.set_ylabel(r'$\theta_1$')

        line, = ax.plot([], [], label = 'Gradient descent', color='#00CC96', lw = 1.5)
        point, = ax.plot([], [], "^", color = '#00CC96', markersize = 5)
        
        def animate(i):
            # animate function for animation
            iteration = 'Iteration ' + str(i)
            theta_values = '\n'+r'$\theta_1$ = '+str(t_1[i])+',  '+r'$\theta_0$ = '+str(t_0[i])
            line.set_data(t_0[:i+1], t_1[:i+1])
            ax.set_title(iteration+'  RSS:'+str(self.RSS[i])+theta_values)
            point.set_data(t_0[i], t_1[i])
            return line, point

        ax.legend()

        anim = FuncAnimation(fig, animate, frames=np.arange(0, len(t_0)), interval=800, blit=True)
        return anim