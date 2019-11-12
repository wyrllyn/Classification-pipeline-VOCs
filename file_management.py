import numpy as np
import pandas as pd
import csv
from scipy.io import arff
import sklearn
from sklearn.preprocessing import KBinsDiscretizer #k bins
from caimcaim import CAIMD
from mdlp.discretization import MDLP #MDLP = fayyad




# file pre-processing requirement: positive = 0, negative = 1
def load_file(data_path):
	with open(data_path, 'r') as f:
	    reader = csv.reader(f, delimiter=',')
	    headers = next(reader)
	    data = list(reader)
	    data = np.array(data).astype(float)
	X = data[:, 0:-1]
	y = data[:, -1:]
	for line in X:
		for elem in line:
			elem = round(elem,3)
	return headers, X, y


def get_interval(enc, attribute_index, value, max):
	if value == max -1:
		return str(round(enc.bin_edges_[attribute_index][value], 3)) +";"+ "inf"
	else:
		return str(round(enc.bin_edges_[attribute_index][value], 3)) +";"+ str(round(enc.bin_edges_[attribute_index][value+1],3))

def get_real_label(int_val):
	if (int_val == 0):
		return "positive"
	else:
		return "negative"

#ok après k bins discretization
def arff_after_discretization(name, headers, X, y, enc):
	full = "@relation " + name + "\n\n"
	attribute_index = 0
	for voc in headers[0:-1]:
		full += "@attribute "+ voc +" {"
		max_index_intervall = enc.bin_edges_[attribute_index].size-1
		for nb in range(0, max_index_intervall):
			full += "'\\'(" + get_interval(enc, attribute_index, nb, max_index_intervall) + "]\\''"
			if nb < max_index_intervall-1:
				full += ","
		full +="}\n"
		attribute_index += 1
	full+= "@attribute "+ headers[-1] +" {positive, negative}\n"
	full += "\n@data \n"
	nb_voc = X[0].size
	nb_indiv = int(X.size/X[0].size)


	for ind in range(0,nb_indiv):
		for j in range(0, nb_voc):
			max_index_intervall = enc.bin_edges_[j].size-1
			full += "'\\'(" + get_interval(enc, j, int(X[ind][j]), max_index_intervall) + "]\\'',"
		full += get_real_label(y[ind])
		if ind < nb_indiv-1:
			full += "\n"

	with open(name, "w") as text_file:
			print(full, file=text_file) 


def k_bins_discretization(X, k, encode, strategy):
	enc = KBinsDiscretizer(n_bins=k, encode=encode, strategy=strategy)
	enc.fit(X)
	return enc, enc.transform(X)



def main():
	data_path = 'data/T3_VOC.csv'
	data_path = 'data/prostate_voc.csv'
	headers, X, y = load_file(data_path)

	print(headers)
	print(X)


	##### discretization
	##############################
	# K BINS
	#encode = 'ordinal'
	#strategy = 'uniform'
	#k = 10
	#enc, dataset_binned = k_bins_discretization(X, k, encode, strategy)
	#name = data_path + "_" + str(k) +"bins.arff"
	#arff_after_discretization(name, headers, dataset_binned, y, enc)

	#k = 5
	#enc, dataset_binned = k_bins_discretization(X, k, encode, strategy)
	#name = data_path + "_" + str(k) +"bins.arff"
	#arff_after_discretization(name, headers, dataset_binned, y, enc)

	#print(name)


	#########################################################################
	######################################################################
	# to check
	#pd.DataFrame(dataset_binned).to_csv("test.csv")

	caim = CAIMD()
	x_disc = caim.fit_transform(X, y)

	print(str(type(x_disc)))

	print(x_disc)

	########## test fayyad (problem with VOC instances)

	#from sklearn.datasets import load_iris
	#iris = load_iris()
	#X, y = iris.data, iris.target

	#print(X)

	#print(str(type(y)))

	#mdlp = MDLP()
	#conv_X = mdlp.fit_transform(X, y)
	#print(mdlp.cat2intervals(conv_X, 10))
	#print (mdlp.cut_points_)

	#print(conv_X)


if __name__ == "__main__":
    main()