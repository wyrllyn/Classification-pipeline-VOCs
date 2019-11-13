import autosklearn.classification
import sklearn.model_selection
import sklearn.metrics

def list_metrics():
	print("Available CLASSIFICATION metrics autosklearn.metrics.*:")
	print("\t*" + "\n\t*".join(autosklearn.metrics.CLASSIFICATION_METRICS))

def print_scores(y_test, predictions):
	tn, fp, fn, tp = sklearn.metrics.confusion_matrix(y_test, predictions).ravel()
	specificity = tn/(tn+fp)

	print("Sensibility", sklearn.metrics.recall_score(y_test, predictions))
	print("Specificity", specificity)
	print("ROC AUC", sklearn.metrics.roc_auc_score(y_test, predictions))
	print("MCC", sklearn.metrics.matthews_corrcoef(y_test, predictions))

#X_train, X_test, y_train, y_test = split_train_test_sets(X,y, 1)
def split_train_test_sets(X,y, seed):
	return sklearn.model_selection.train_test_split(X, y, random_state=seed)

def run_autosklearn(X_train, y_train, duration, nb_folds, metric, seed, name):
	automl = autosklearn.classification.AutoSklearnClassifier(
		seed=seed,
        time_left_for_this_task=duration,
        tmp_folder='tmp/autosklearn_cv_example_tmp',
        output_folder='tmp/autosklearn_cv_example_out',
        delete_tmp_folder_after_terminate=True,
        resampling_strategy='cv',
        resampling_strategy_arguments={'folds': nb_folds},
	)
	automl.fit(X_train.copy(), y_train.copy(), dataset_name=name, metric=metric)
	automl.refit(X_train.copy(), y_train.copy())
	return automl