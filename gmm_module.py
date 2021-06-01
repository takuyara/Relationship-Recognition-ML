import numpy as np
import time
from scipy import linalg
from scipy.special import logsumexp

def gausscov(X, resp, nk, mean, regcovar):
	n_com, n_fe = mean.shape
	cov = np.empty((n_com, n_fe, n_fe))
	for k in range(n_com):
		diff = X - mean[k]
		cov[k] = np.dot(resp[ : , k] * diff.T, diff) / nk[k]
		cov[k].flat[ : : n_fe + 1] += regcovar
	return cov

def gausspara(X, resp, regcovar):
	nk = resp.sum(axis = 0) + 10 * np.finfo(resp.dtype).eps
	mean = np.dot(resp.T, X) / nk[ : , np.newaxis]
	cov = gausscov(X, resp, nk, mean, regcovar)
	return nk, mean, cov

def preccholesky(cov):
	n_com, n_fe, tmp = cov.shape
	res = np.empty((n_com, n_fe, n_fe))
	for i, j in enumerate(cov):
		res[i] = linalg.solve_triangular(linalg.cholesky(j, lower = True), np.eye(n_fe), lower = True).T
	return res

def logdetcholesky(mat, n_fe):
	n_com = mat.shape[0]
	return (np.sum(np.log(mat.reshape(n_com, -1)[ : , : : n_fe + 1]), 1))

def loggaussprob(X, mean, precchol):
	n_sp, n_fe = X.shape
	n_com = mean.shape[0]
	logdet = logdetcholesky(precchol, n_fe)
	logprob = np.empty((n_sp, n_com))
	for k, (mu, jchol) in enumerate(zip(mean, precchol)):
		logprob[ : , k] = np.sum(np.square(np.dot(X, jchol) - np.dot(mu, jchol)), axis = 1)
	return -0.5 * (n_fe * np.log(2 * np.pi) + logprob) + logdet

class GMM:
	def __init__(self, n_com, n_calc = 5, max_iter = 100):
		self.n_com = n_com
		self.n_calc = n_calc
		self.max_iter = max_iter
		self.randstate = np.random.RandomState(int(time.time()))
	def clcwtlogprob(self, X):
		return self.clclogprob(X) + self.clclogwt()
	def clclogprob(self, X):
		return loggaussprob(X, self.mean, self.precchol)
	def clclogwt(self):
		return np.log(self.weight)
	def clclogprobresp(self, X):
		wtlogprob = self.clcwtlogprob(X)
		logprobnorm = logsumexp(wtlogprob, axis = 1)
		with np.errstate(under = "ignore"):
			logresp = wtlogprob - logprobnorm[ : , np.newaxis]
		return logprobnorm, logresp
	def init(self, X):
		n_sp = X.shape[0]
		resp = self.randstate.rand(n_sp, self.n_com)
		resp /= resp.sum(axis = 1)[ : , np.newaxis]
		self.weight, self.mean, self.cov = gausspara(X, resp, 1e-6)
		self.weight /= n_sp
		self.precchol = preccholesky(self.cov)
	def getpara(self):
		return (self.weight, self.mean, self.cov, self.precchol)
	def setpara(self, para):
		(self.weight, self.mean, self.cov, self.precchol) = para
		n_fe = self.mean.shape[1]
		self.prec = np.empty(self.precchol.shape)
		for k, precchol in enumerate(self.precchol):
			self.prec[k] = np.dot(precchol, precchol.T)
	def estep(self, X):
		logprobnorm, logresp = self.clclogprobresp(X)
		return np.mean(logprobnorm), logresp
	def mstep(self, X, logresp):
		n_sp = X.shape[0]
		self.weight, self.mean, self.cov = gausspara(X, np.exp(logresp), 1e-6)
		self.weight /= n_sp
		self.precchol = preccholesky(self.cov)
	def fit(self, X):
		X = X.astype("float64")
		n_sp = X.shape[0]
		mxlb = -np.infty
		for i in range(self.n_calc):
			self.init(X)
			lb = -np.infty
			for j in range(self.max_iter):
				prelb = lb
				logprobnorm, logresp = self.estep(X)
				self.mstep(X, logresp)
				lb = logprobnorm
				if abs(lb - prelb) < 1e-5:
					break
			if lb > mxlb:
				mxlb = lb
				bestpara = self.getpara()
		self.setpara(bestpara)
		self.lb = mxlb
