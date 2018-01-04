#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from tester import dump_classifier_and_data, main as tester_main
from poi_feature_selection import select_best
from poi_remove_outliers import remove_outliers
#from poi_nb import pipeline, tree, naive_bayes, svm, grid_search_svm
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from poi_new_features import add_new_features

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ["poi","bonus","deferral_payments", "deferred_income", "director_fees",
             "exercised_stock_options", "expenses", "from_poi_to_this_person", 
             "from_this_person_to_poi", "loan_advances", "long_term_incentive", "other",
             "restricted_stock", "restricted_stock_deferred", "salary", 
             "shared_receipt_with_poi", "total_payments", "total_stock_value"]
### Load the dictionary containing the dataset

with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
    
### Task 2: Remove outliers
    
data_dict_cleaned = remove_outliers(data_dict)

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = add_new_features(data_dict_cleaned)

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
skb, features_train, features_test, labels_train, labels_test, features_selected = select_best(features_list, labels, features)


'''

AFTER MANUALLY TESTING THROUGH DECISION TREE CLASSIFIER, GAUSSIAN NAIVE BAYES, AND SVM,
GAUSSIAN NAIVE BAYES IS THE ALGORITHM THAT PERFORMED THE BEST (WHICH PERFORMED AT THE HIGHEST ACCURACY OF 87%)

BELOW IS MY CODE....

'''

sk_fold = StratifiedShuffleSplit(labels, 1000, random_state = 42)

pca = PCA()
clf_nb = GaussianNB()
pipe = Pipeline(steps=[ ("SKB", skb), ("PCA", pca), ("GNB", clf_nb)])
    
pca_params = {"PCA__n_components":range(1,3), "PCA__whiten": [True, False]}

kbest_params = {"SKB__k":range(3,5)}

pca_params.update(kbest_params)

gs =  GridSearchCV(pipe, pca_params, scoring='f1', cv=sk_fold)

gs.fit(features, labels)
    
clf = gs.best_estimator_
    
dump_classifier_and_data(clf, my_dataset, features_list)
    
tester_main()


'''

BELOW IS WHAT I HAD BEFORE.  YOU CAN SEE HOW I EXPLORED GAUSSIAN NAIVE BAYES, DECISION TREE CLASSIFICATION, AND SVM

IN THE END I CHOSE GAUSSIAN NAIVE BAYES, WHICH HAD THE HIGHEST ACCURACY.

sk_fold = StratifiedShuffleSplit(labels, 1000, random_state = 42)

pca = PCA()
clf_nb = GaussianNB()
clf_dt = DecisionTreeClassifier()
clf_lsvc = LinearSVC()
scaler = MinMaxScaler()

algo_list = [clf_lsvc, clf_nb, clf_dt]
for al in algo_list:
    
    an = ""
    if al == clf_lsvc:
        an = "LSVC__"
    elif al == clf_nb:
        an = "GNB"
    elif al == clf_dt:
        an = "DT"
    
    pipe = Pipeline(steps=[ ("SKB", skb), ("PCA", pca), (an, al)])
    
    pca_params = {"PCA__n_components":range(1,3), "PCA__whiten": [True, False]}
    
    if an == "LSCV":
        #Cs = np.logspace(-2.3, -1.3, 10)
        Cs = [0.01, 1, 10, 100, 1000]
        tol = [1e-3, 1e-5, 1e-7, 1e-9, 1e-11, 1e-12, 1e-13]
        fit_int = [True, False]
        lsvc_params = {"LSVC__C":Cs,"LSVC__penalty":[ 'l2'], "LSVC__tol":tol, "LSVC_fit_intercept":fit_int} 
        pca_params.update(lsvc_params)
    kbest_params = {"SKB__k":range(3,5)}

    pca_params.update(kbest_params)

    gs =  GridSearchCV(pipe, pca_params, scoring='f1', cv=sk_fold)
        # Fit GridSearchCV
    gs.fit(features, labels)
    
    clf = gs.best_estimator_
    
    dump_classifier_and_data(clf, my_dataset, features_list)
    
    tester_main()
'''