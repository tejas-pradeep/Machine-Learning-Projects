import numpy as np
from matplotlib import pyplot as plt

class PCA(object):

    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X): # 5 points
        """
        Decompose dataset into principal components.
        You may use numpy.linalg.svd function and set full_matrices=False.

        Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
        corresponding values from PCA.

        Args:
            X: N*D array corresponding to a dataset
        Return:
            None
        """
        X = X - np.mean(X, axis=0, keepdims=True)
        self.U, self.S, self.V = np.linalg.svd(X, full_matrices=False)

    def transform(self, data, K=2): # 2 pts
        """
        Transform data to reduce the number of features such that final data has given number of columns

        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: N*D array corresponding to a dataset
            K: Int value for number of columns to be kept
        Return:
            X_new: N*K array corresponding to data obtained by applying PCA on data
        """
        self.fit(data)
        return np.dot(data, self.V.T[:,:K])


    def transform_rv(self, data, retained_variance=0.99): # 3 pts
        """
        Transform data to reduce the number of features such that a given variance is retained

        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: N*D array corresponding to a dataset
            retained_variance: Float value for amount of variance to be retained
        Return:
            X_new: N*K array corresponding to data obtained by applying PCA on data
        """
        var = np.cumsum(self.S ** 2)
        cumilative_variance = var / np.sum(self.S ** 2)
        for i in range(len(data)):
            if cumilative_variance[i] > retained_variance:
                break
        return np.dot(data, self.V.T[:, :i + 1])



    def get_V(self):
        """ Getter function for value of V """
        
        return self.V
    
    def visualize(self, X, y): 
        """
        Use your PCA implementation to reduce the dataset to only 2 features.

        Create a scatter plot of the reduced data set and differentiate points that
        have different true labels using color.

        Args:
            xtrain: NxD numpy array, where N is number of instances and D is the dimensionality 
            of each instance
            ytrain: numpy array (N,), the true labels
            
        Return: None
        """
        pca = PCA()
        X = pca.transform(X, 2)
        x_0 = np.argwhere(y == 0)[:, 0]
        x_1 = np.argwhere(y == 1)[:, 0]
        plt.scatter(X[x_0][:, 0], X[x_0][:, 1], c='blue', marker='x', label='0')
        plt.scatter(X[x_1][:, 0], X[x_1][:, 1], c='magenta', marker='x', label='1')
        
        ##################### END YOUR CODE ABOVE #######################
        plt.legend()
        plt.show()