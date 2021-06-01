#Pionniers du TJ, benissiez-moi par votre Esprits Saints!
import numpy as np
import time
from scipy import linalg

def classmean(X, Y):
	classes, Y = np.unique(Y, return_inverse = True)
	cnt = np.bincount(Y)
	means = np.zeros(shape = (len(classes), X.shape[1]))
	np.add.at(means, Y, X)
	return means / cnt[ : , None]

class LDA:
	def __init__(self, n_com):
		self.n_com = n_com
	def predict(self, X):
		return np.dot(X, self.coef.T) + self.intercept
	def fit(self, X, Y):
		n_sp, n_fe = X.shape
		Yunq, Yt = np.unique(Y, return_inverse = True)
		n_cls = len(Yunq)
		pri = np.bincount(Yt) / float(len(Y))
		mean = classmean(X, Y)
		Xlst = []
		for i, j in enumerate(Yunq):
			Xlst.append(X[Y == j, : ] - mean[i])
		Xbar = np.dot(pri, mean)
		Xlst = np.concatenate(Xlst, axis = 0)
		std = Xlst.std(axis = 0)
		std[std == 0] = 1.0
		fac = 1.0 / (n_sp - n_cls)
		X = np.sqrt(fac) * (Xlst / std)
		U, S, V = linalg.svd(X, full_matrices = False)
		rank = np.sum(S > 1e-4)
		scal = (V[ : rank] / std).T / S[ : rank]
		X = np.dot(((np.sqrt((n_sp * pri) * fac)) * (mean - Xbar).T).T, scal)
		tmp, S, V = linalg.svd(X, full_matrices = 0)
		rank = np.sum(S > 1e-4 * S[0])	
		scal = np.dot(scal, V.T[ : , : rank])
		coef = np.dot(mean - Xbar, scal)
		self.intercept = -0.5 * np.sum(coef ** 2, axis = 1) + np.log(pri)
		self.coef = np.dot(coef, scal.T)
		self.intercept -= np.dot(Xbar, self.coef.T)
		if n_cls == 2:
			self.coef = np.array(self.coef[1, : ] - self.coef[0, : ], ndmin = 2, dtype = "float64")
			self.intercept = np.array(self.intercept[1] - self.intercept[0], ndmin = 1, dtype = "float64")
			