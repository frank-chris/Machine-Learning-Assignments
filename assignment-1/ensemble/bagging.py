import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class BaggingClassifier():
    def __init__(self, base_estimator, n_estimators=100):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''
        # initialising attributes
        self.trees_list = []
        self.samples_list = []
        self.n_estimators = n_estimators
        self.base_estimator = base_estimator
        
        # initialising estimators
        for i in range(self.n_estimators):
            self.trees_list.append(self.base_estimator(criterion='entropy'))


    def fit(self, X, y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        assert(X.index.size == y.size)

        self.X = X
        self.y = y
        for tree in self.trees_list:
            # sampling before training estimator
            X_sample = X.sample(frac=1, replace=True)
            y_sample = y[X_sample.index]
            self.samples_list.append([X_sample, y_sample])
            tree.fit(X_sample, y_sample)

    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        predictions = {}
        for i, tree in enumerate(self.trees_list):
            predictions[str(i)] = pd.Series(tree.predict(X))

        pred_df = pd.DataFrame(predictions)

        # return mode of predictions of estimators as final prediction
        return pred_df.mode(axis=1)[0]

    def plot(self):
        """
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]

        """
        # plotting individual decision boundaries of estimators
        x_min, x_max = self.X.iloc[:, 0].min() - 1, self.X.iloc[:, 0].max() + 1
        y_min, y_max = self.X.iloc[:, 1].min() - 1, self.X.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05), np.arange(y_min, y_max, 0.05))

        fig1, axarr = plt.subplots(1, self.n_estimators, sharex='col', sharey='row', figsize=(24, 5))

        for idx, clf, tt in zip([i for i in range(self.n_estimators)],
                                self.trees_list,
                                [str(i+1) for i in range(self.n_estimators)]):

            Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)

            Z = pd.DataFrame(Z)

            to_replace = list(pd.unique(Z.values.ravel('K')))
            to_replace.sort()
            value = [i for i in range(len(to_replace))]
            
            Z = Z.replace(to_replace=to_replace, value=value)
            c = self.samples_list[idx][1].replace(to_replace=to_replace, value=value)

            axarr[idx].contourf(xx, yy, Z, alpha=0.5)
            axarr[idx].scatter(self.samples_list[idx][0].iloc[:, 0], self.samples_list[idx][0].iloc[:, 1], c=c, s=20, edgecolor='k')
            axarr[idx].set_title(tt)

        for axis in axarr.flat:
            axis.set(xlabel='x1='+str(self.X.columns[0]), ylabel='x2='+str(self.X.columns[1]))

        for axis in axarr.flat:
            axis.label_outer()

        fig1.suptitle('Decision Boundaries of Estimators-BaggingClassifier')

        # plotting overall decision boundary
        fig2, ax = plt.subplots(1, 1, sharex='col', sharey='row', figsize=(8, 5))

        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.to_numpy()
        Z = Z.reshape(xx.shape)

        Z = pd.DataFrame(Z)

        to_replace = list(pd.unique(Z.values.ravel('K')))
        to_replace.sort()
        value = [i for i in range(len(to_replace))]
        
        Z = Z.replace(to_replace=to_replace, value=value)
        c = self.y.replace(to_replace=to_replace, value=value)

        ax.contourf(xx, yy, Z, alpha=0.5)
        ax.scatter(self.X.iloc[:, 0], self.X.iloc[:, 1], c=c, s=25, edgecolor='k')
        ax.set_title('Overall Decision Boundary - BaggingClassifier')
        ax.set(xlabel='x1='+str(self.X.columns[0]), ylabel='x2='+str(self.X.columns[1]))
        plt.show()
        return [fig1,fig2]
