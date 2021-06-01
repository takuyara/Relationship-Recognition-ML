from math import log, sqrt

import numpy as np
from scipy import linalg



def randomized_range_finder(A, size, n_iter,
                            power_iteration_normalizer='auto',
                            random_state=None):

    Q = random_state.normal(size=(A.shape[1], size))
    if A.dtype.kind == 'f':
        # Ensure f32 is preserved as f32
        Q = Q.astype(A.dtype, copy=False)

    for i in range(n_iter):
        Q, _ = linalg.lu(np.dot(A, Q), permute_l=True)
        Q, _ = linalg.lu(np.dot(A.T, Q), permute_l=True)

    Q, _ = linalg.qr(np.dot(A, Q), mode='economic')
    return Q

def svd_flip(u, v, u_based_decision=True):
    max_abs_cols = np.argmax(np.abs(u), axis=0)
    signs = np.sign(u[max_abs_cols, range(u.shape[1])])
    u *= signs
    v *= signs[:, np.newaxis]
    return u, v



def randomized_svd(M, n_components, n_oversamples=10, n_iter='auto',
                   power_iteration_normalizer='auto', transpose='auto',
                   flip_sign=True, random_state=0):

    n_random = n_components + n_oversamples
    n_samples, n_features = M.shape

    n_iter = 7 if n_components < .1 * min(M.shape) else 4


    Q = randomized_range_finder(M, n_random, n_iter,
                                power_iteration_normalizer, random_state)

    # project M to the (k + p) dimensional space using the basis vectors
    B = np.dot(Q.T, M)

    # compute the SVD on the thin matrix: (k + p) wide
    Uhat, s, V = linalg.svd(B, full_matrices=False)
    del B
    U = np.dot(Q, Uhat)
    U, V = svd_flip(U, V)
    return U[:, :n_components], s[:n_components], V[:n_components, :]



class PCA:

    def __init__(self, n_components=None, copy=True, whiten=False,
                 svd_solver='auto', tol=0.0, iterated_power='auto',
                 random_state=None):
        self.n_components = n_components
        self.copy = copy
        self.whiten = whiten
        self.svd_solver = svd_solver
        self.tol = tol
        self.random_state = random_state

    def fit_transform(self, X, y=None):
        U, S, V = self._fit(X)
        U = U[:, :self.n_components]

        U *= S[:self.n_components]

        return U

    def _fit(self, X):
        """Dispatch to the right submethod depending on the chosen solver."""

        # Raise an error for sparse input.
        # This is more informative than the generic one raised by check_array.
        
        return self._fit_truncated(X, self.n_components, None)

 
    def _fit_truncated(self, X, n_components, svd_solver):
        """Fit the model by computing truncated SVD (by ARPACK or randomized)
        on X
        """
        X = X.astype("float64")

        n_samples, n_features = X.shape


        random_state = np.random.RandomState(20170208)

        # Center data
        self.mean_ = np.mean(X, axis=0)



        X -= self.mean_

            # sign flipping is done inside
        U, S, V = randomized_svd(X, n_components=n_components,
                                 flip_sign=True,
                                 random_state=random_state)

        
        self.components_ = V

        return U, S, V
    def transform(self, X):
        X = X - self.mean_
        X_transformed = np.dot(X, self.components_.T)
        return X_transformed
    def inverse_transform(self, X):
        return np.dot(X, self.components_) + self.mean_



