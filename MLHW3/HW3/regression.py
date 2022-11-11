import numpy as np


class Regression(object):

    def __init__(self):
        pass

    def rmse(self, pred, label):  # [5pts]
        '''
        This is the root mean square error.
        Args:
            pred: numpy array of length N * 1, the prediction of labels
            label: numpy array of length N * 1, the ground truth of labels
        Return:
            a float value
        '''
        return np.sqrt(np.mean(np.square((label - pred))))

    def construct_polynomial_feats(self, x, degree):  # [5pts]
        """
        Args:
            x: N x D numpy array, where N is number of instances and D is the
               dimensionality of each instance.
            degree: the max polynomial degree
        Return:
            feat: 
                For 1-D array, numpy array of shape Nx(degree+1), remember to include
                the bias term. feat is in the format of:
                [[1.0, x1, x1^2, x1^3, ....,],
                 [1.0, x2, x2^2, x2^3, ....,],
                 ......
                ]
                
                For D-dimensional array: numpy array of shape N x (degree+1) x D, remember to include
                  the bias term.

                Example: print(feat)
                For an input where N=3, D=2, and degree=3...

                [[[ 1.0        1.0]
                  [ x_{1,1}    x_{1,1}]
                  [ x_{1,1}^2  x_{1,2}^2]
                  [ x_{1,1}^3  x_{1,2}^3]]

                 [[ 1.0        1.0]
                  [ x_{2,1}    x_{2,2}]
                  [ x_{2,1}^2  x_{2,2}^2]
                  [ x_{2,1}^3  x_{2,2}^3]]

                 [[ 1.0        1.0]
                  [ x_{3,1}    x_{3,2}]
                  [ x_{3,1}^2  x_{3,2}^2]
                  [ x_{3,1}^3  x_{3,2}^3]]]

        """
        if len(x.shape) > 1:
            return np.power(x[:, np.newaxis, :].repeat(degree + 1, axis = 1), np.arange(degree + 1)[np.newaxis, :, np.newaxis])
        else:
            return np.power(x[:, np.newaxis],
                            np.arange(degree + 1))


    def predict(self, xtest, weight):  # [5pts]
        """
        Args:
            xtest: NxD numpy array, where N is number
                   of instances and D is the dimensionality of each
                   instance
            weight: Dx1 numpy array, the weights of linear regression model
        Return:
            prediction: Nx1 numpy array, the predicted labels
        """
        return xtest@weight

    # =================
    # LINEAR REGRESSION
    # Hints: in the fit function, use close form solution of the linear regression to get weights.
    # For inverse, you can use numpy linear algebra function
    # For the predict, you need to use linear combination of data points and their weights (y = theta0*1+theta1*X1+...)

    def linear_fit_closed(self, xtrain, ytrain):  # [5pts]
        """
        Args:
            xtrain: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: N x 1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        a = np.linalg.pinv(np.dot(xtrain.T, xtrain))
        return np.dot(np.dot(a, xtrain.T), ytrain)

    def linear_fit_GD(self, xtrain, ytrain, epochs=5, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        N, D = xtrain.shape
        w = np.zeros((D, 1))
        for i in range(epochs):
            w += learning_rate * np.dot(xtrain.T, ytrain - np.dot(xtrain, w)) / N
        return w

    def linear_fit_SGD(self, xtrain, ytrain, epochs=100, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        N, D = xtrain.shape
        w = np.zeros((D, 1))
        for i in range(epochs):
            for j in range(N):
                h = self.predict(xtrain[j], w)
                w += (learning_rate * (xtrain[j]) * (ytrain[j] - h)[0]).reshape(-1, 1)
        return w

    # =================
    # RIDGE REGRESSION

    def ridge_fit_closed(self, xtrain, ytrain, c_lambda):  # [5pts]
        """
        Args:
            xtrain: N x D numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: N x 1 numpy array, the true labels
            c_lambda: floating number
        Return:
            weight: Dx1 numpy array, the weights of ridge regression model
        """
        N, D = xtrain.shape
        w = np.zeros((D, 1))
        I = np.identity(D)
        I[0, :] = 0
        return np.linalg.pinv(xtrain.T @ xtrain + c_lambda*I) @ xtrain.T @ ytrain

    def ridge_fit_GD(self, xtrain, ytrain, c_lambda, epochs=500, learning_rate=1e-7):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
            c_lambda: floating number
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        N, D = xtrain.shape
        w = np.zeros((D, 1))
        for i in range(epochs):
            a = xtrain.T @ (ytrain - (xtrain @ w))
            w += learning_rate * (a + c_lambda * w) / N
        return w

    def ridge_fit_SGD(self, xtrain, ytrain, c_lambda, epochs=100, learning_rate=0.001):  # [5pts]
        """
        Args:
            xtrain: NxD numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: Nx1 numpy array, the true labels
        Return:
            weight: Dx1 numpy array, the weights of linear regression model
        """
        N, D = xtrain.shape
        w = np.zeros((D, 1))
        for i in range(epochs):
            for j in range(N):
                h = self.predict(xtrain[j], w)
                l = ytrain[j] - h
                w[0] += learning_rate * (l*xtrain[j][0] - c_lambda*w[0])
                for k in range(1, D):
                    w[k] += learning_rate*(l*xtrain[j][k] - c_lambda*w[k])
        return w

    def ridge_cross_validation(self, X, y, kfold=10, c_lambda=100):  # [8 pts]
        """
        Args: 
            X : NxD numpy array, where N is the number of instances and D is the dimensionality of each instance
            y : Nx1 numpy array, true labels
            kfold: Number of folds you should take while implementing cross validation.
            c_lambda: Value of regularization constant
        Returns:
            meanErrors: Float average rmse error
        Hint: np.concatenate might be helpful.
        Look at 3.5 to see how this function is being used.
        # For cross validation, use 10-fold method and only use it for your training data (you already have the train_indices to get training data).
        # For the training data, split them in 10 folds which means that use 10 percent of training data for test and 90 percent for training.
        """
        N, D = X.shape
        fold = N // kfold
        mean_e = 0.0
        for i in range(kfold):
            xtrain = np.concatenate((X[:i*fold, :], X[(i+1)*fold:, :]))
            ytrain = np.concatenate((y[:i * fold, :], y[(i + 1) * fold:, :]))
            weight = self.ridge_fit_closed(xtrain, ytrain, c_lambda)
            p = self.predict(X[i * fold:(i + 1) * fold, :], weight)
            mean_e += self.rmse(p, y[i * fold:(i + 1) * fold])
        return mean_e / kfold