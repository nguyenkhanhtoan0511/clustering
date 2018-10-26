import sys
import os
import pickle
import numpy as np
from sklearn import svm

def test_data(db, md):
	with open(md, 'rb') as file:  
		clf = pickle.load(file)

	with open(db,'r') as f:
		pathImages = f.read().strip('\n').split('\n')

	dataTest = []

	for path in pathImages:
		pos = path.rfind('\\')
		pathFeature = 'features/vgg16_fc2/{}.npy'.format(path[path[0:pos].rfind('\\')+1:path.rfind('.')])
		dataTest.append(np.load(pathFeature)[0])

	print('Prepare Data Test: Done!')

	os.chdir("exp/svm_linear/")

	with open("result.dat", "w") as f:
		for path, data in zip(pathImages, dataTest):
			pos = path.rfind('\\')
			temp = str(path[pos+1:]).replace(".npy", ".jpg") + " => " + str(clf.predict([data]))
			f.write(temp)
			f.write("\n")

	# pickle.dump(clf, open("model.dat", 'wb'))


def main():
	db = "db/test.txt" #E:/MayHocTrongThiGiacMayTinh/deadline/baitap2/db/db1/test.txt
	md = "exp/svm_linear/model.dat" #E:/MayHocTrongThiGiacMayTinh/deadline/baitap2/exp/svm_linear/model.dat
	test_data(db, md)

if __name__=='__main__':
	main()