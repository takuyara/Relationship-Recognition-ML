# Pionniers du TJ, benissiez-moi par votre Esprits Saints!
import cv2
import numpy as np
from pca_module import PCA
class reader:
	def readbmp(bmpname, shape = None):
		img = cv2.cvtColor(cv2.imread(bmpname), cv2.COLOR_BGR2GRAY)
		# print(img.shape)
		# img = img[ ]
		assert(img.shape[0] == shape[0] and img.shape[1] == shape[1])
		return np.reshape(np.array(img), (-1))
	def runpca(self, ncom = 50):
		if hasattr(self, "vecP") == False:
			self.pca = PCA(n_com = ncom)
			self.vecP = self.pca.fit_apply(self.vecL)
		return self.vecP
	def concatenate(rd1, rd2):
		res = reader("", 0)
		res.vecL = np.concatenate([rd1.vecL, rd2.vecL])
		return res
	def applypca(self, vec):
		return self.pca.apply(vec)
	def __init__(self, folder, idrange, digcnt = 3, split = False):
		if folder == "":
			return
		self.vecL = []
		for i in range(idrange[0], idrange[1]):
			nm = str(i)
			nm = folder + "0" * (digcnt - len(nm)) + nm
			if not split:
				self.vecL.append(np.concatenate([reader.readbmp(nm + "_C.bmp", (100, 60)), reader.readbmp(nm + "_P.bmp", (100, 60))]))
			else:
				self.vecL.append(reader.readbmp(nm + "_C.bmp", (100, 60)))
				self.vecL.append(reader.readbmp(nm + "_P.bmp", (100, 60)))
		self.vecL = np.array(self.vecL)
	def inverse(self, vec, split = True):
		res = self.pca.inv(vec)
		if split:
			# print(res.shape)
			cnt = res.shape[1] // 2
			return (res[ : , : cnt], res[ : , cnt : ])
		return res

