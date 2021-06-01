# Pionniers du TJ, benissiez-moi par votre Esprits Saints!
import numpy as np
import cv2
from reader import reader
from lda_module import LDA
def outputimage(img, flname):
	# img = tile(np.reshape(img, (img.shape[0], img.shape[1], 1)), (1, 1, 3))
	# cv2.imwrite(img, flname)
	# img = np.reshape(img, img.shape + (1, ))
	# img = np.tile(img, (1, 1, 3))
	# img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
	# print(img.shape)
	cv2.imwrite(flname, np.reshape(img, (100, 60)))
krd = reader("facedata\\kinship_resize\\", (1, 399))
krd_t = reader("facedata\\kinship_resize\\", (400, 494))
nkrd = reader("facedata\\nonkinship_resize\\", (1, 279))
nkrd_t = reader("facedata\\nonkinship_resize\\", (280, 345))
train = reader.concatenate(krd, nkrd)
train.runpca()
krd.vecP = train.applypca(krd.vecL)
krd_invC, krd_invP = train.inverse(vec = krd.vecP)
# print(krd_invC.shape, krd_invP.shape)
nkrd.vecP = train.applypca(nkrd.vecL)
nkrd_invC, nkrd_invP = train.inverse(vec = nkrd.vecP)
for i in range(krd_invC.shape[0]):
	flstr = "faceres\\kinship\\" + "0" * (3 - len(str(i))) + str(i)
	outputimage(krd_invC[i], flstr + "_C.bmp")
	outputimage(krd_invP[i], flstr + "_P.bmp")
for i in range(nkrd_invC.shape[0]):
	flstr = "faceres\\nonkinship\\" + "0" * (3 - len(str(i))) + str(i)
	outputimage(nkrd_invC[i], flstr + "_C.bmp")
	outputimage(nkrd_invP[i], flstr + "_P.bmp")
trainX = np.concatenate([krd.vecP, nkrd.vecP])
invC, invP = train.inverse(trainX)
for i in range(invC.shape[0]):
	flstr = "faceres\\ttt\\" + "0" * (3 - len(str(i))) + str(i)
	outputimage(invC[i], flstr + "_C.bmp")
	outputimage(invP[i], flstr + "_P.bmp")

trainY = np.array([1] * krd.vecL.shape[0] + [0] * nkrd.vecL.shape[0])
testX = np.concatenate([train.applypca(krd_t.vecL), train.applypca(nkrd_t.vecL)])
testY = np.array([1] * krd_t.vecL.shape[0] + [0] * nkrd_t.vecL.shape[0])
lda = LDA(n_com = 1)
lda.fit(trainX, trainY)
trainX1 = np.reshape(lda.predict(trainX), (-1, 1))
predictY = np.reshape(lda.predict(testX), (-1, 1))
predictY1 = []
for i in predictY:
	mindis = np.infty
	for j in range(len(trainX1)):
		if abs(i - trainX1[j]) < mindis:
			mindis = abs(i - trainX1[j])
			res = trainY[j]
	predictY1.append(res)
predictY1 = np.array(predictY1)
res = np.mean(predictY1 == testY)
print(res)
