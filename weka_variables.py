from weka.classifiers import Classifier
##### LIST OF CLASSIFIERS ####
####################

## use without attribute selection
def set_weka_options():
  classifiers = [
    #  Classifier(classname="weka.classifiers.functions.SMO", options=["-C", "0.1", "-K", "weka.classifiers.functions.supportVector.PolyKernel"]),
      Classifier(classname="weka.classifiers.functions.SMO", options=["-C", "1", "-K", "weka.classifiers.functions.supportVector.PolyKernel"]),
      Classifier(classname="weka.classifiers.functions.SMO", options=["-C", "10", "-K", "weka.classifiers.functions.supportVector.PolyKernel"]), # is ok
      Classifier(classname="weka.classifiers.functions.SMO", options=["-C", "100", "-K", "weka.classifiers.functions.supportVector.PolyKernel"]), #is ok
      #Classifier(classname="weka.classifiers.functions.SMO", options=["-C", "0.1","-K", "weka.classifiers.functions.supportVector.RBFKernel"]),
      #Classifier(classname="weka.classifiers.functions.SMO", options=["-C", "1","-K", "weka.classifiers.functions.supportVector.RBFKernel"]),
      #Classifier(classname="weka.classifiers.functions.SMO", options=["-C", "10","-K", "weka.classifiers.functions.supportVector.RBFKernel"]),
      #Classifier(classname="weka.classifiers.functions.SMO", options=["-C", "100","-K", "weka.classifiers.functions.supportVector.RBFKernel"]),
      #Classifier(classname="weka.classifiers.functions.SMO", options=["-K", "weka.classifiers.functions.supportVector.Puk"]),
      # Classifier(classname="weka.classifiers.functions.SMO", options=["-K", "weka.classifiers.functions.supportVector.NormalizedPolyKernel"]), 

      Classifier(classname="weka.classifiers.trees.J48"),
    #  Classifier(classname="weka.classifiers.trees.J48", options=["-R", "-N", "3"]),

      Classifier(classname="weka.classifiers.functions.Logistic"),
    #is ok  Classifier(classname="weka.classifiers.functions.Logistic", options=["-R", "1.0E-6"]),
   #is ok   Classifier(classname="weka.classifiers.functions.Logistic", options=["-R", "1.0E-4"]),
    #is ok  Classifier(classname="weka.classifiers.functions.Logistic", options=["-R", "1.0E-2"]),


      Classifier(classname="weka.classifiers.bayes.NaiveBayes"),
      Classifier(classname="weka.classifiers.trees.RandomForest"),
      #################################################################
      Classifier(classname="weka.classifiers.lazy.IBk", options=["-K", "1"]),
    #  Classifier(classname="weka.classifiers.lazy.IBk", options=["-K", "3"]),
     # Classifier(classname="weka.classifiers.lazy.IBk", options=["-K", "5"])
    #  Classifier(classname="weka.classifiers.trees.J48")
  ]
  ## use with attribute selection
  classifiers_for_filtered = [
   #   Classifier(classname="weka.classifiers.functions.SMO", options=["-C", "0.1", "-K", "weka.classifiers.functions.supportVector.PolyKernel"]),
      Classifier(classname="weka.classifiers.functions.SMO", options=["-C", "1", "-K", "weka.classifiers.functions.supportVector.PolyKernel"]),
      Classifier(classname="weka.classifiers.functions.SMO", options=["-C", "10", "-K", "weka.classifiers.functions.supportVector.PolyKernel"]),
      Classifier(classname="weka.classifiers.functions.SMO", options=["-C", "100", "-K", "weka.classifiers.functions.supportVector.PolyKernel"]),
     # Classifier(classname="weka.classifiers.functions.SMO", options=["-C", "0.1","-K", "weka.classifiers.functions.supportVector.RBFKernel"]),
     # Classifier(classname="weka.classifiers.functions.SMO", options=["-C", "1","-K", "weka.classifiers.functions.supportVector.RBFKernel"]),
     # Classifier(classname="weka.classifiers.functions.SMO", options=["-C", "10","-K", "weka.classifiers.functions.supportVector.RBFKernel"]),
     # Classifier(classname="weka.classifiers.functions.SMO", options=["-C", "100","-K", "weka.classifiers.functions.supportVector.RBFKernel"]),

      Classifier(classname="weka.classifiers.trees.J48"),
     # Classifier(classname="weka.classifiers.trees.J48", options=["-R", "-N", "3"]),

      Classifier(classname="weka.classifiers.functions.Logistic"),
      Classifier(classname="weka.classifiers.functions.Logistic", options=["-R", "1.0E-6"]),
      Classifier(classname="weka.classifiers.functions.Logistic", options=["-R", "1.0E-4"]),
      Classifier(classname="weka.classifiers.functions.Logistic", options=["-R", "1.0E-2"]),

      Classifier(classname="weka.classifiers.bayes.NaiveBayes"),
      Classifier(classname="weka.classifiers.trees.RandomForest"),
      Classifier(classname="weka.classifiers.bayes.BayesNet"),
      #Classifier(classname="weka.classifiers.functions.MultilayerPerceptron"),
      Classifier(classname="weka.classifiers.lazy.IBk", options=["-K", "1"]),
     # Classifier(classname="weka.classifiers.lazy.IBk", options=["-K", "3"]),
     # Classifier(classname="weka.classifiers.lazy.IBk", options=["-K", "5"])
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

  return classifiers, classifiers_for_filtered, list_attribute_selection, list_of_comparison_metric