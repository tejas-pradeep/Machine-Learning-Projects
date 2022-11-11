import numpy as np

class ImgCompression(object):
    def __init__(self):
        pass

    def svd(self, X): # [5pts]
        """
        Do SVD. You could use numpy SVD.
        Your function should be able to handle black and white
        images (N*D arrays) as well as color images (N*D*3 arrays)
        In the image compression, we assume that each colum of the image is a feature. Image is the matrix X.
        Args:
            X: N * D array corresponding to an image (N * D * 3 if color image)
        Return:
            U: N * N for black and white images / N * N * 3 for color images
            S: min(N, D) * 1 for black and white images / min(N, D) * 3 for color images
            V: D * D for black and white images / D * D * 3 for color images
        """
        if len(X.shape) == 2:
            return np.linalg.svd(X)
        a, b, c = X.shape
        u = np.zeros((a, a, 3))
        s = np.zeros((min(a, b), 3))
        v = np.zeros((b, b, 3))
        for i in range(3):
            u[:, :, i], s[:, i], v[:, :, i] = self.svd(X[:, :, i])
        return u, s, v


    def rebuild_svd(self, U, S, V, k): # [5pts]
        """
        Rebuild SVD by k componments.
        Args:
            U: N*N (*3 for color images)
            S: min(N, D)*1 (*3 for color images)
            V: D*D (*3 for color images)
            k: int corresponding to number of components
        Return:
            Xrebuild: N*D array of reconstructed image (N*D*3 if color image)

        Hint: numpy.matmul may be helpful for reconstructing color images
        """
        N = len(U)
        D = len(V)
        if len(U.shape) == 2:
            return U[:, :k]@np.diag(S[:k])@V[:k, :]
        else:
            Xrebuild = np.zeros((N, D, 3))
            for i in range(3):
                u_temp = U[:, :k, i]
                v_temp = V[:k, :, i]
                s_temp = np.diag(S[:, i][:k])
                Xrebuild[:, :, i] = u_temp@s_temp@v_temp
        return Xrebuild
    def compression_ratio(self, X, k): # [5pts]
        """
        Compute compression of an image: (num stored values in compressed)/(num stored values in original)
        Args:
            X: N * D array corresponding to an image (N * D * 3 if color image)
            k: int corresponding to number of components
        Return:
            compression_ratio: float of proportion of storage used by compressed image
        """
        return (k * (1 + len(X) + len(X[0]))) / (len(X) * len(X[0]))

    def recovered_variance_proportion(self, S, k): # [5pts]
        """
        Compute the proportion of the variance in the original matrix recovered by a rank-k approximation

        Args:
           S: min(N, D)*1 (*3 for color images) of singular values for the image
           k: int, rank of approximation
        Return:
           recovered_var: float (array of 3 floats for color image) corresponding to proportion of recovered variance
        """
        S = np.square(S)
        # a = np.array(S)
        if S.ndim == 1:
            return np.sum(S[:k]) / np.sum(S)
        else:
            return np.array([np.sum(S[:k, i]) / np.sum(S[:, i]) for i in range(3)])
