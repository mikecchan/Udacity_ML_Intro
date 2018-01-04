# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pickle
import numpy
numpy.random.seed(42)
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn import cross_validation

def select_best(features_list, labels, features):
    features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)    
    
    skb = SelectKBest(f_classif, k = 5)
    skb.fit(features_train, labels_train)
    
    features_selected=[features_list[i+1] for i in skb.get_support(indices=True)]
    
    feature_scores = skb.scores_
    
    features_scores_selected = [feature_scores[i] for i in skb.get_support(indices=True)]

    print ' '
    print 'Selected Features', features_selected
    print 'Feature Scores', features_scores_selected
    
    #print features_selected
    
    return skb, features_train, features_test, labels_train, labels_test, features_selected
