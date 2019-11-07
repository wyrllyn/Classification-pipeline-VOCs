import glob
import os
import shutil
import time
import sys

import weka.core.converters
from weka.core.converters import Loader

import weka.core.jvm as jvm
from weka.classifiers import Classifier
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
from weka.filters import Filter
from weka.experiments import Tester, ResultMatrix, SimpleCrossValidationExperiment

from weka_variables import set_weka_options 


##### FUNCTIONS ####
####################

#voir commentaire + en détail pour la doc
# selects attributes and return a string containing filtered data
def use_filter(data, str_eval, str_search):
    """
    Uses the AttributeSelection filter for attribute selection.
    :param data: the dataset to use
    :type data: Instances
    """
  #  print("\n2. Filter")
    flter = Filter(classname="weka.filters.supervised.attribute.AttributeSelection")
    aseval = ASEvaluation(classname=str_eval) #weka.attributeSelection.CfsSubsetEval
    assearch = ASSearch(classname=str_search, options=["-B"]) #weka.attributeSelection.GreedyStepwise
    flter.set_property("evaluator", aseval.jobject)
    flter.set_property("search", assearch.jobject)
    flter.inputformat(data)
    filtered = flter.filter(data)
    return str(filtered)

#creates temporary files with selected variables and return a list of these files path
def create_temp_filtered_files(datasets, str_eval, str_search, tmp_dir):
	list_temp_files = list()
	tmp_eval_name = str_eval.split(".")[-1]
	tmp_search_name = str_search.split(".")[-1]
	loader = Loader(classname="weka.core.converters.ArffLoader") 
	for ds in datasets:
		data_filtered =  use_filter(loader.load_file(ds), str_eval, str_search)
		tmp_file_name = ds.split("/")[-1]
		full_name = tmp_eval_name +"-" + tmp_search_name + "_" + tmp_file_name
		list_temp_files.append("tmpFiles/"+ full_name)
	#	print(list_temp_files[-1])
		with open(tmp_dir + full_name, "w") as text_file:
			print(data_filtered, file=text_file) 		
	return list_temp_files

# creates all datasets filtered
def attribute_selection(list_attribute_selection, datasets):
	all_datasets_filtered = list()
	for element in list_attribute_selection:
		datasets_filtered = create_temp_filtered_files(datasets, element[0], element[1], tmp_dir)
		all_datasets_filtered.append(datasets_filtered)
	return all_datasets_filtered


#experimenter unfiltered
def experimenter(datasets, base_res, nb_runs, nb_folds, classifiers):
	tmpres=list()
	result = base_res + "_" + str(nb_folds) + "folds.arff"
	exp = SimpleCrossValidationExperiment(
		classification=True,
		runs=nb_runs,
	 	folds=nb_folds,
	    datasets=datasets,
	    classifiers=classifiers,
	    result=result)
	exp.setup()
	exp.run()
	tmpres.append(result)
	return tmpres

#run experimenter on filtered datasets	
def experimenter_filtered (all_datasets_filtered, base_res, nb_runs, nb_folds, classifiers_for_filtered):
	tmpres=list()
	for ds in all_datasets_filtered:
		attrib_name = (ds[-1].split("_")[0]).split("/")[-1]
		res = base_res + "_" + attrib_name + "_" + str(nb_folds) + "folds.arff"
		exp = SimpleCrossValidationExperiment(
	    	classification=True,
	    	runs=nb_runs,
	    	folds=nb_folds,
	    	datasets=ds,
	    	classifiers=classifiers_for_filtered,
	    	result=res)
		exp.setup()
		exp.run()
		tmpres.append(res)
	return tmpres

# display results of one experiment according to a comparison metric
def expe_printer(res_file, comparison_metric):
	loader = weka.core.converters.loader_for_file(res_file)
	data = loader.load_file(res_file)
	matrix = ResultMatrix(classname="weka.experiment.ResultMatrixPlainText")
	tester = Tester(classname="weka.experiment.PairedCorrectedTTester")
	tester.resultmatrix = matrix
	comparison_col = data.attribute_by_name(comparison_metric).index
	tester.instances = data
	print(tester.header(comparison_col))
	print(tester.multi_resultset_full(0, comparison_col))

def expe_printer_to_latex(res_file, comparison_metric):
	text = ""
	loader = weka.core.converters.loader_for_file(res_file)
	data = loader.load_file(res_file)
	matrix = ResultMatrix(classname="weka.experiment.ResultMatrixLatex")
	tester = Tester(classname="weka.experiment.PairedCorrectedTTester")
	tester.resultmatrix = matrix
	comparison_col = data.attribute_by_name(comparison_metric).index
	tester.instances = data
	text += tester.header(comparison_col)
	text += tester.multi_resultset_full(0, comparison_col)
	return text

#display results of a list of experiments for several comparison metrics
def full_expe_printer(list_of_res_files, list_of_comparison_metric, destination):
	latex_table = ""
	for R in list_of_res_files:
		for CM in list_of_comparison_metric:
			expe_printer(R, CM)
			latex_table += expe_printer_to_latex(R, CM)
	with open(destination, "w") as text_file:
			print(latex_table, file=text_file) 	

#function to run autoweka
#examples of call autoweka(data, "1", "areaUnderROC")
#autoweka(data, "1", "fMeasure")
#autoweka(data, "1", "fMeasure")
def autoweka(data, duration, metric, nb_folds):
	classifier = Classifier(classname="weka.classifiers.meta.AutoWEKAClassifier", options=["-x", nb_folds, "-timeLimit", duration, "-metric", metric])  #classname="weka.classifiers.functions.Logistic", options=["-R", "1.0E-2"]
	classifier.build_classifier(data)
	print(classifier)


# GLOBAL VARIABLES
path = os.getcwd()
tmp_dir = path + '/tmpFiles/'
data_dir = "../datasets/weka/"
res_dir = path + "/results/"
res_exp_dir = res_dir + "weka_exp_arff/"
res_latex_dir = res_dir + "weka_latex_tabs/"
nb_folds = 5
nb_runs = 1



##### GET FILES ####
####################

begin = time.time()
pattern = "T3_VOC" #input("Simple instance name: ")

print(sys.argv[1])
pattern = sys.argv[1]
nb_folds = int(sys.argv[2])
#print(nb_folds)
base_res = res_exp_dir + pattern

datasets = [f for f in glob.glob(data_dir + pattern + "*.arff", recursive=True)]

#### CREATE DIRECTORIES ####
############################
if not os.path.exists(res_dir):
    os.mkdir(res_dir)
if not os.path.exists(res_exp_dir):
   	os.mkdir(res_exp_dir)
if not os.path.exists(res_latex_dir):
	os.mkdir(res_latex_dir)


if not os.path.exists(tmp_dir):
    os.mkdir(tmp_dir)




##### START ########
###################

jvm.start(packages=True)

##### LIST OF CLASSIFIERS ####
####################

classifiers, classifiers_for_filtered, list_attribute_selection, list_of_comparison_metric = set_weka_options()


#### ATTRIBUTE SELECTION ####
#############################
print("-- ATTRIBUTE SELECTION")
# to convert files before using function use_filter


all_datasets_filtered = attribute_selection(list_attribute_selection, datasets)

end = time.time()
print("---- duration of first phase: " + str(end-begin))



##### EXPERIMENTER ####
####################

results = list()

print("-- EXPERIMENTER FULL")

begin = time.time()
results += experimenter(datasets, base_res, nb_runs, nb_folds, classifiers)
end = time.time()
print("---- duration of phase: " + str(end-begin))


print("-- EXPERIMENTER FILTERED")
begin = time.time()
results += experimenter_filtered(all_datasets_filtered, base_res, nb_runs, nb_folds, classifiers_for_filtered)
end = time.time()
print("---- duration of phase: " + str(end-begin))



#### VISUALISATION + RESULTS #####
##################################

#TODO: check possibilities

full_expe_printer(results, list_of_comparison_metric, res_latex_dir+ pattern + "_" + str(nb_folds) + "folds_latex")

jvm.stop()

shutil.rmtree(tmp_dir, ignore_errors=True)
print("-- THE END --")
