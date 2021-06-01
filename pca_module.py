#Pionniers du TJ, benissiez-moi par votre Esprits Saints!
import numpy as np
import time
from scipy import linalg

def rangefinder(A, size, randstate):
	Q = randstate.normal(size = (A.shape[1], size)).astype("float64", copy = False)
	for i in range(7):
		Q = linalg.lu(np.dot(A, Q), permute_l = True)[0]
		Q = linalg.lu(np.dot(A.T, Q), permute_l = True)[0]
	Q = linalg.qr(np.dot(A, Q), mode = "economic")[0]
	return Q

def svd_flip(u, v):
	col = np.argmax(np.abs(u), axis = 0)
	sgn = np.sign(u[col, range(u.shape[1])])
	u *= sgn
	v *= sgn[ : , np.newaxis]
	return u, v

def randsvd(M, n_com, n_over = 10, randstate = None):
	Q = rangefinder(M, n_com + n_over, randstate)
	B = np.dot(Q.T, M)
	Um, s, V = linalg.svd(B, full_matrices = False)
	del B
	U = np.dot(Q, Um)
	U, V = svd_flip(U, V)
	return U[ : , : n_com], s[ : n_com], V[ : n_com, : ]

class PCA:
	def __init__(self, n_com):
		self.n_com = n_com
		self.randstate = np.random.RandomState(int(time.time()))
	def fit_apply(self, X):
		U, S, V = self.fit(X)
		U = U[ : , : self.n_com]
		return U * S[ : self.n_com]
	def fit(self, X):
		X = X.astype("float64")
		self.mean = np.mean(X, axis = 0)
		X -= self.mean
		U, S, V = randsvd(X, n_com = self.n_com, randstate = self.randstate)
		self.component = V
		return U, S, V
	def apply(self, X):
		X = X - self.mean
		return np.dot(X, self.component.T)
	def inv(self, X):
		return np.dot(X, self.component) + self.mean
