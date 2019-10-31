import glob
import os
import shutil
import time

import weka.core.converters
from weka.core.converters import Loader

import weka.core.jvm as jvm
from weka.classifiers import Classifier
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
from weka.filters import Filter
from weka.experiments import Tester, ResultMatrix


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
	for ds in datasets:
		data_filtered =  use_filter(loader.load_file(ds), str_eval, str_search)
		tmp_file_name = ds.split("/")[-1]
		full_name = tmp_eval_name +"-" + tmp_search_name + "_" + tmp_file_name
		list_temp_files.append("tmpFiles/"+ full_name)
	#	print(list_temp_files[-1])
		with open(tmp_dir + full_name, "w") as text_file:
			print(data_filtered, file=text_file) 		
	return list_temp_files

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

#display results of a list of experiments for several comparison metrics
def full_expe_printer(list_of_res_files, list_of_comparison_metric):
	for R in list_of_res_files:
		for CM in list_of_comparison_metric:
			expe_printer(R, CM)



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

pattern = "T3_VOC_1" #input("Simple instance name: ")
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



jvm.start()


##### LIST OF CLASSIFIERS ####
####################

## use without attribute selection
classifiers = [
    Classifier(classname="weka.classifiers.functions.SMO"),
    Classifier(classname="weka.classifiers.trees.J48"),
    Classifier(classname="weka.classifiers.functions.Logistic"),
    Classifier(classname="weka.classifiers.bayes.NaiveBayes"),
    Classifier(classname="weka.classifiers.trees.RandomForest"),
    Classifier(classname="weka.classifiers.lazy.IBk")
]
## use with attribute selection
classifiers_for_filtered = [
    Classifier(classname="weka.classifiers.functions.SMO"),
    Classifier(classname="weka.classifiers.trees.J48"),
    Classifier(classname="weka.classifiers.functions.Logistic"),
    Classifier(classname="weka.classifiers.bayes.NaiveBayes"),
    Classifier(classname="weka.classifiers.trees.RandomForest"),
    Classifier(classname="weka.classifiers.bayes.BayesNet"),
 #   Classifier(classname="weka.classifiers.functions.MultilayerPerceptron"),
    Classifier(classname="weka.classifiers.lazy.IBk")
]

list_attribute_selection = [
	("weka.attributeSelection.CfsSubsetEval", "weka.attributeSelection.GreedyStepwise"),
	#("weka.attributeSelection.CfsSubsetEval", "weka.attributeSelection.BestFirst"),
	#("weka.attributeSelection.ClassifierAttributeEval", "weka.attributeSelection.Ranker"),
	#("weka.attributeSelection.ClassifierSubsetEval", "weka.attributeSelection.GreedyStepwise"),#too many
	#("weka.attributeSelection.CorrelationAttributeEval", "weka.attributeSelection.Ranker"),#too many
	#("weka.attributeSelection.GainRatioAttributeEval", "weka.attributeSelection.Ranker"),#too many
	#("weka.attributeSelection.InfoGainAttributeEval", "weka.attributeSelection.Ranker"), #too many
	#("weka.attributeSelection.OneRAttributeEval", "weka.attributeSelection.Ranker"),
	#("weka.attributeSelection.PrincipalComponents", "weka.attributeSelection.Ranker") #bug O____O
	#("weka.attributeSelection.ReliefAttributeEval", "weka.attributeSelection.Ranker"),
	#("weka.attributeSelection.SymmetricalUncertAttributeEval", "weka.attributeSelection.Ranker"), #too many
	#("weka.attributeSelection.WrapperSubsetEval", "weka.attributeSelection.GreedyStepwise")
]

list_of_comparison_metric = [
	"True_positive_rate",
	"True_negative_rate",
	"Area_under_ROC",
	"Matthews_correlation"
]

#### ATTRIBUTE SELECTION ####
#############################
print("-- ATTRIBUTE SELECTION")
# to convert files before using function use_filter
loader = Loader(classname="weka.core.converters.ArffLoader") 


all_datasets_filtered = list()
for element in list_attribute_selection:
	print("********************")
	datasets_filtered = create_temp_filtered_files(datasets, element[0], element[1], tmp_dir)
	all_datasets_filtered.append(datasets_filtered)

end = time.time()
print("---- duration of first phase: " + str(end-begin))

##### EXPERIMENTER ####
####################

results = list()

print("-- EXPERIMENTER FULL")
begin = time.time()
result = res_exp_dir + pattern + "_" + str(nb_folds) + "folds.arff"
from weka.experiments import SimpleCrossValidationExperiment
exp = SimpleCrossValidationExperiment(
    classification=True,
    runs=nb_runs,
    folds=nb_folds,
    datasets=datasets,
    classifiers=classifiers,
    result=result)
exp.setup()
exp.run()
results.append(result)

end = time.time()
print("---- duration of phase: " + str(end-begin))


print("-- EXPERIMENTER FILTERED")
begin = time.time()
for ds in all_datasets_filtered:
	attrib_name = (ds[-1].split("_")[0]).split("/")[-1]
	res = res_exp_dir + pattern + "_" + attrib_name + "_" + str(nb_folds) + "folds.arff"
	exp = SimpleCrossValidationExperiment(
    	classification=True,
    	runs=nb_runs,
    	folds=nb_folds,
    	datasets=ds,
    	classifiers=classifiers_for_filtered,
    	result=res)
	exp.setup()
	exp.run()
	results.append(res)

end = time.time()
print("---- duration of phase: " + str(end-begin))


#### VISUALISATION + RESULTS #####
##################################

#TODO: check possibilities

full_expe_printer(results, list_of_comparison_metric)



jvm.stop()

shutil.rmtree(tmp_dir, ignore_errors=True)
print("-- THE END --")
