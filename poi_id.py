#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

features_list = ['poi', 'deferral_payments', 'expenses', 'from_messages', 'bonus', 'total_emails', 
                 'from_this_person_to_poi', 'restricted_stock', 'salary', 'total_payments']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    enron_data = pickle.load(data_file)

### Task 2: Remove outliers

enron_data.pop("TOTAL")
enron_data.pop("THE TRAVEL AGENCY IN THE PARK")
enron_data.pop('LOCKHART EUGENE E')

### Task 3: Create new feature(s)

for name in enron_data.keys():
    edn = enron_data[name]
    edn["perc_poi_contact"] = 0
    edn["total_emails"] = 0
    edn["total_poi_contact"] = 0
    if (edn["from_this_person_to_poi"] != "NaN" and edn["from_poi_to_this_person"] != "NaN"
        and edn["to_messages"] != "NaN" and edn["from_messages"] != "NaN"):
        edn["total_poi_contact"] = edn["from_this_person_to_poi"] + edn["from_poi_to_this_person"]
        edn["total_emails"] = edn["to_messages"] + edn["from_messages"]
        edn["perc_poi_contact"] = float(edn["total_poi_contact"]) / edn["total_emails"]
        
### Store to my_dataset for easy export below.

my_dataset = enron_data

### Extract features and labels from dataset for local testing

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.ensemble import AdaBoostClassifier  

clf = AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=1.875, n_estimators=2, random_state=None)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)
