
import numpy as np

from scipy import linalg

from scipy.special import logsumexp

def _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar):

    n_components, n_features = means.shape
    covariances = np.empty((n_components, n_features, n_features))
    for k in range(n_components):
        diff = X - means[k]
        covariances[k] = np.dot(resp[:, k] * diff.T, diff) / nk[k]
        covariances[k].flat[::n_features + 1] += reg_covar
    return covariances




def _estimate_gaussian_parameters(X, resp, reg_covar, covariance_type):

    nk = resp.sum(axis=0) + 10 * np.finfo(resp.dtype).eps
    means = np.dot(resp.T, X) / nk[:, np.newaxis]
    covariances = _estimate_gaussian_covariances_full(resp, X, nk, means, reg_covar)
    print(covariances.shape)
    return nk, means, covariances


def _compute_precision_cholesky(covariances, covariance_type):

    n_components, n_features, _ = covariances.shape
    precisions_chol = np.empty((n_components, n_features, n_features))
    for k, covariance in enumerate(covariances):
        cov_chol = linalg.cholesky(covariance, lower=True)
        precisions_chol[k] = linalg.solve_triangular(cov_chol, np.eye(n_features), lower=True).T
    return precisions_chol


def _compute_log_det_cholesky(matrix_chol, covariance_type, n_features):

    n_components, _, _ = matrix_chol.shape
    log_det_chol = (np.sum(np.log(matrix_chol.reshape(n_components, -1)[:, ::n_features + 1]), 1))

    return log_det_chol


def _estimate_log_gaussian_prob(X, means, precisions_chol, covariance_type):

    n_samples, n_features = X.shape
    n_components, _ = means.shape
    # det(precision_chol) is half of det(precision)
    log_det = _compute_log_det_cholesky(
        precisions_chol, covariance_type, n_features)
    log_prob = np.empty((n_samples, n_components))
    for k, (mu, prec_chol) in enumerate(zip(means, precisions_chol)):
        y = np.dot(X, prec_chol) - np.dot(mu, prec_chol)
        log_prob[:, k] = np.sum(np.square(y), axis=1)

    return -.5 * (n_features * np.log(2 * np.pi) + log_prob) + log_det


class BaseMixture:

    def __init__(self, n_components, tol, reg_covar,
                 max_iter, n_init, init_params, random_state, warm_start,
                 verbose, verbose_interval):
        self.n_components = n_components
        self.tol = tol
        self.reg_covar = reg_covar
        self.max_iter = max_iter
        self.n_init = n_init
        self.init_params = init_params
        self.random_state = np.random.RandomState(20191223)
        self.warm_start = warm_start
        self.verbose = verbose
        self.verbose_interval = verbose_interval

    



    



class GMM(BaseMixture):

    def __init__(self, n_com=1, covariance_type='full', tol=1e-3,
                 reg_covar=1e-6, max_iter=100, n_init=1, init_params='kmeans',
                 weights_init=None, means_init=None, precisions_init=None,
                 random_state=None, warm_start=False,
                 verbose=0, verbose_interval=10):
        super().__init__(
            n_components=n_com, tol=tol, reg_covar=reg_covar,
            max_iter=max_iter, n_init=n_init, init_params=init_params,
            random_state=random_state, warm_start=warm_start,
            verbose=verbose, verbose_interval=verbose_interval)

        self.covariance_type = covariance_type
        self.weights_init = weights_init
        self.means_init = means_init
        self.precisions_init = precisions_init


    def fit(self, X, y=None):

        X = X.astype("float64")
        

        # if we enable warm_start, we will have a unique initialisation
        n_init = 1

        max_lower_bound = -np.infty
        self.converged_ = False

        

        n_samples, _ = X.shape
        
        self._initialize_parameters(X, self.random_state)

        lower_bound = -np.infty

        for n_iter in range(1, self.max_iter + 1):
            prev_lower_bound = lower_bound

            log_prob_norm, log_resp = self._e_step(X)
            self._m_step(X, log_resp)
            lower_bound = log_prob_norm
            change = lower_bound - prev_lower_bound

            if abs(change) < self.tol:
                self.converged_ = True
                break

        if lower_bound > max_lower_bound:
            max_lower_bound = lower_bound
            best_params = self._get_parameters()
            best_n_iter = n_iter


        self._set_parameters(best_params)
        self.n_iter_ = best_n_iter
        self.lower_bound_ = max_lower_bound


        _, log_resp = self._e_step(X)

        return log_resp.argmax(axis=1)

    def _estimate_weighted_log_prob(self, X):
        return self._estimate_log_prob(X) + self._estimate_log_weights()


    def _estimate_log_prob(self, X):
        return _estimate_log_gaussian_prob(
            X, self.means_, self.precisions_cholesky_, self.covariance_type)

    def _estimate_log_weights(self):
        return np.log(self.weights_)

    def _estimate_log_prob_resp(self, X):
        weighted_log_prob = self._estimate_weighted_log_prob(X)
        log_prob_norm = logsumexp(weighted_log_prob, axis=1)
        with np.errstate(under='ignore'):
            log_resp = weighted_log_prob - log_prob_norm[:, np.newaxis]
        return log_prob_norm, log_resp

    def _initialize_parameters(self, X, random_state):
        """Initialize the model parameters.

        Parameters
        ----------
        X : array-like, shape  (n_samples, n_features)

        random_state : RandomState
            A random number generator instance.
        """
        n_samples, _ = X.shape

        resp = random_state.rand(n_samples, self.n_components)
        resp /= resp.sum(axis=1)[:, np.newaxis]
        self._initialize(X, resp)

    def _e_step(self, X):
        log_prob_norm, log_resp = self._estimate_log_prob_resp(X)
        return np.mean(log_prob_norm), log_resp

    def _initialize(self, X, resp):
        """Initialization of the Gaussian mixture parameters.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)

        resp : array-like, shape (n_samples, n_components)
        """
        n_samples, _ = X.shape

        weights, means, covariances = _estimate_gaussian_parameters(
            X, resp, self.reg_covar, self.covariance_type)
        weights /= n_samples

        self.weights_ = (weights if self.weights_init is None
                         else self.weights_init)
        self.means_ = means if self.means_init is None else self.means_init

        
        self.covariances_ = covariances
        self.precisions_cholesky_ = _compute_precision_cholesky(covariances, self.covariance_type)
        

    def _m_step(self, X, log_resp):

        n_samples, _ = X.shape
        self.weights_, self.means_, self.covariances_ = (
            _estimate_gaussian_parameters(X, np.exp(log_resp), self.reg_covar,
                                          self.covariance_type))
        self.weights_ /= n_samples
        self.precisions_cholesky_ = _compute_precision_cholesky(
            self.covariances_, self.covariance_type)


    def _get_parameters(self):
        return (self.weights_, self.means_, self.covariances_,
                self.precisions_cholesky_)

    def _set_parameters(self, params):
        (self.weights_, self.means_, self.covariances_,
         self.precisions_cholesky_) = params

        # Attributes computation
        _, n_features = self.means_.shape


        self.precisions_ = np.empty(self.precisions_cholesky_.shape)
        for k, prec_chol in enumerate(self.precisions_cholesky_):
            self.precisions_[k] = np.dot(prec_chol, prec_chol.T)


    def _n_parameters(self):
        """Return the number of free parameters in the model."""
        _, n_features = self.means_.shape
        if self.covariance_type == 'full':
            cov_params = self.n_components * n_features * (n_features + 1) / 2.
        elif self.covariance_type == 'diag':
            cov_params = self.n_components * n_features
        elif self.covariance_type == 'tied':
            cov_params = n_features * (n_features + 1) / 2.
        elif self.covariance_type == 'spherical':
            cov_params = self.n_components
        mean_params = n_features * self.n_components
        return int(cov_params + mean_params + self.n_components - 1)

