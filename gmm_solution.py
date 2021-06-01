# Pionniers du TJ, benissiez-moi par votre Esprits Saints!
import numpy as np
import cv2
from reader import reader
from gmm_module import GMM
krd = reader("facedata\\kinship_resize\\", (1, 399), split = True)
krd_t = reader("facedata\\kinship_resize\\", (400, 494), split = True)
nkrd = reader("facedata\\nonkinship_resize\\", (1, 299), split = True)
nkrd_t = reader("facedata\\nonkinship_resize\\", (300, 345), split = True)
train = reader.concatenate(krd, nkrd)
train.runpca()
krd.vecP = train.applypca(krd.vecL)
nkrd.vecP = train.applypca(nkrd.vecL)
trainX = np.concatenate([krd.vecP, nkrd.vecP])
# print(trainX.shape)
# trainY = np.array([1] * krd.vecL.shape[0] + [0] * nkrd.vecL.shape[0])
# testX = np.concatenate([krd_t.vecL, nkrd_t.vecL])
# testY = np.array([1] * krd_t.vecL.shape[0] + [0] * nkrd_t.vecL.shape[0])
gmm = GMM(n_com = 8)
gmm.fit(trainX)
# print(gmm.means_)
def outputimage(img, flname):
	cv2.imwrite(flname, np.reshape(img, (100, 60)))
folder = "GaussMixture\\"
for i in range(8):
	outputimage(train.inverse(gmm.mean[i], False), folder + str(i) + ".bmp")
