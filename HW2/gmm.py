from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import sys
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from tqdm import tqdm
# Load image
import imageio

from kmeans import KMeans

SIGMA_CONST = 1e-6
LOG_CONST = 1e-32


class GMM(object):
    def __init__(self, X, K, max_iters=100):  # No need to change
        """
        Args:
            X: the observations/datapoints, N x D numpy array
            K: number of clusters/components
            max_iters: maximum number of iterations (used in EM implementation)
        """
        self.points = X
        self.max_iters = max_iters

        self.N = self.points.shape[0]  # number of observations
        self.D = self.points.shape[1]  # number of features
        self.K = K  # number of components/clusters

    # Helper function for you to implement
    def softmax(self, logit):  # [5pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            prob: N x D numpy array. See the above function.
        """
        num = np.exp(logit - np.max(logit, axis=-1, keepdims=True))
        return num / np.sum(num, axis=-1, keepdims=True)

    def logsumexp(self, logit):  # [3pts]
        """
        Args:
            logit: N x D numpy array
        Return:
            s: N x 1 array where s[i,0] = logsumexp(logit[i,:]). See the above function
        """
        return np.log(np.sum(np.exp(logit - np.max(logit, axis=-1, keepdims=True)), axis=-1, keepdims=True)) + np.max(logit, axis=-1, keepdims=True)

    # for undergraduate student
    def normalPDF(self, logit, mu_i, sigma_i):  # [5pts]
        """
        Args:
            logit: N x D numpy array
            mu_i: 1xD numpy array, the center for the ith gaussian.
            sigma_i: 1xDxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            pdf: (N,) numpy array, the probability distribution of N data for the ith gaussian

        Hint:
            np.diagonal() should be handy.
        """
        n = logit.shape[1]
        det = np.linalg.det(sigma_i)
        if det == 0:
            det = 1e-10
        inv = np.linalg.inv(sigma_i + SIGMA_CONST)
        const = np.power(det, -0.5) / (np.power((2 * np.pi), (n / 2)))
        ans = np.exp(-0.5 * np.sum(np.matmul(logit - mu_i, inv).T * (logit - mu_i).T, axis=0))
        return const * ans

    # for grad students
    def multinormalPDF(self, logits, mu_i, sigma_i):  # [5pts]
        """
        Args:
            logit: N x D numpy array
            mu_i: 1xD numpy array, the center for the ith gaussian.
            sigma_i: 1xDxD numpy array, the covariance matrix of the ith gaussian.
        Return:
            normal_pdf: (N,) numpy array, the probability distribution of N data for the ith gaussian

        Hint:
            np.linalg.det() and np.linalg.inv() should be handy.
            Add SIGMA_CONST if you encounter LinAlgError
        """
        n = logits.shape[1]
        det = np.linalg.det(sigma_i)
        if det == 0:
            det = 1e-10
        inv = np.linalg.inv(sigma_i + SIGMA_CONST)
        const = np.power(det, -0.5) / (np.power((2 * np.pi), (n/2)))
        ans = np.exp(-0.5 * np.sum(np.matmul(logits - mu_i, inv).T * (logits - mu_i).T, axis=0))
        return const * ans

    def _init_components(self, **kwargs):  # [5pts]
        """
        Args:
            kwargs: any other arguments you want
        Return:
            pi: numpy array of length K, prior
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.
                You will have KxDxD numpy array for full covariance matrix case
        Hint: You can use KMeans to initialize the centers for each gaussian
        """
        _, mu, _ = KMeans().__call__(self.points, self.K)
        pi = np.array([1/self.K] * self.K)
        sigma = np.array([np.eye(self.points.shape[1])] * self.K)
        return pi, mu, sigma

    def _ll_joint(self, pi, mu, sigma, full_matrix=True, **kwargs):  # [10 pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case
            full_matrix: whether we use full covariance matrix in Normal PDF or not. Default is True.

        Return:
            ll(log-likelihood): NxK array, where ll(i, k) = log pi(k) + log NormalPDF(points_i | mu[k], sigma[k])

        Hint: You might find it useful to add LOG_CONST to the expression before taking the log
        """
        # List to np array to fix issue in gradescope -> unable to allocate space from array
        join = np.empty((self.points.shape[0], self.K))
        for i in range(self.K):
               join[:, i] = np.log(pi[i] + 1e-32) + np.log(self.multinormalPDF(self.points, mu[i], sigma[i]) + LOG_CONST)
        return join
    def _E_step(self, pi, mu, sigma, **kwargs):  # [5pts]
        """
        Args:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian.You will have KxDxD numpy
            array for full covariance matrix case
        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.

        Hint:
            You should be able to do this with just a few lines of code by using _ll_joint() and softmax() defined above.
        """
        return self.softmax(self._ll_joint(pi, mu, sigma))


    def _M_step(self, gamma, full_matrix=True, **kwargs):  # [10pts]
        """
        Args:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
        Return:
            pi: np array of length K, the prior of each component
            mu: KxD numpy array, the center for each gaussian.
            sigma: KxDxD numpy array, the diagonal standard deviation of each gaussian. You will have KxDxD numpy
            array for full covariance matrix case

        Hint:
            There are formulas in the slide and in the above description box.
        """
        gamma_sum = np.sum(gamma, axis=0, keepdims=True)
        pi = gamma_sum / len(self.points)
        mu = (np.dot(gamma.T, self.points).T / gamma_sum).T
        sigma = np.zeros((gamma.shape[1], self.points.shape[1], self.points.shape[1]))
        for i in range(gamma.shape[1]):
            sigma[i, :, :] = np.matmul((self.points - mu[i]).T, (self.points - mu[i]) * gamma[:, i, np.newaxis]) / np.sum(gamma[:, i])
        return pi.reshape(self.K, ), mu, sigma


    def __call__(self, full_matrix=True, abs_tol=1e-16, rel_tol=1e-16, **kwargs):  # No need to change
        """
        Args:
            abs_tol: convergence criteria w.r.t absolute change of loss
            rel_tol: convergence criteria w.r.t relative change of loss
            kwargs: any additional arguments you want

        Return:
            gamma(tau): NxK array, the posterior distribution (a.k.a, the soft cluster assignment) for each observation.
            (pi, mu, sigma): (1xK np array, KxD numpy array, KxDxD numpy array)

        Hint:
            You do not need to change it. For each iteration, we process E and M steps, then update the paramters.
        """
        pi, mu, sigma = self._init_components(**kwargs)
        pbar = tqdm(range(self.max_iters))

        for it in pbar:
            # E-step
            gamma = self._E_step(pi, mu, sigma)

            # M-step
            pi, mu, sigma = self._M_step(gamma, full_matrix)

            # calculate the negative log-likelihood of observation
            joint_ll = self._ll_joint(pi, mu, sigma, full_matrix)
            loss = -np.sum(self.logsumexp(joint_ll))
            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            pbar.set_description('iter %d, loss: %.4f' % (it, loss))
        return gamma, (pi, mu, sigma)