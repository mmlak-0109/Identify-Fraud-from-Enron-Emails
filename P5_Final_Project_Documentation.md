
# Identify Fraud from Enron Email
#### Matthew Miller 
#### April 10, 2017

## Understanding the Dataset and Question

- Summarize for us the goal of this project and how machine learning is useful in trying to accomplish it. As part of your answer, give some background on the dataset and how it can be used to answer the project question. Were there any outliers in the data when you got it, and how did you handle those?  [relevant rubric items: “data exploration”, “outlier investigation”]

### Project Overview

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. In this project, I will build a person of interest identifier based on financial and email data made public as a result of the Enron scandal. The data used has been combined with a hand-generated list of persons of interest in the fraud case, which means individuals who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity. ([Citation](https://classroom.udacity.com/nanodegrees/nd002/parts/0021345409/modules/317428862475461/lessons/3174288624239847/concepts/31803986370923))

During the course of initial exploration of the dataset, I found it to have the following characteristics:

- 146 people
    - 18 of which, or ~12%, are POIs (the POI to Non-POI ratio is very lopsided)
- 21 features

### Outlier Investigation

For my outlier investigation, I decided to look at the features "salary and "bonus".  I plotted the data points to get a look at the distribution of the data and see if any data points popped out at me.

![outlier plot](https://github.com/mmlak-0109/Identify-Fraud-from-Enron-Emails/blob/master/Pictures/outlier.png?raw=true)

Right away it is easy to see that there is definitely something wrong here.  Most all of the data points are clumped together at the lower left and one data point is way off by itself in the top right.  As it turned out, the outlier was not a person at all, but instead was the value for the total of all of the salaries in the data set (and by extension, the total category for all the features).  After I removed the outlier, I replotted the data to see it looked.

![outlier removed](https://github.com/mmlak-0109/Identify-Fraud-from-Enron-Emails/blob/master/Pictures/outlier_removed.png?raw=true)

There, that's much better!  There are still a few features that stick out on both the high and low end of the spectrum for both salary and bonus.  The points on the high end were Enron executives "Kenneth Lay" and "Jeffery Skilling", so they are definitely legitimate data points.  On the low end, I found many legitimate data points and one interesting data point.  While looking at the list of employees I noticed a person named "THE TRAVEL AGENCY IN THE PARK".  This definitely didn't seem like an employee, so I did a quick Google search and found out that this was a [travel agency co-owned by Ken Lay's younger sister Sharon Lay](http://content.time.com/time/magazine/article/0,9171,198885,00.html) (it was renamed Alliance Worldwide in 2002).  According to Sharon Lay's [LinkedIn page](https://www.linkedin.com/in/sharon-lay-aa985b13/), she left the company in 2005.  Because the travel agency is not an Enron employee, I decided to get rid of it.

Next, I decided to look people that had "NaN" values for the "total_payments" and "total_stock_value" features (I chose those two because they are totals for all other financial features), and discovered three employees that met this criteria.  Even though two of the three employees (William Powers and Ronnie Chan) did not receive payments of any kind, they did have some recorded email exchanges and William Powers even had 12 shared receipts with a POI.  As a result, I kept them.  The third employee named Eugene E Lockhart, however, had "NaN" values for all his features (with the exception of the "poi" feature), so I decided to remove him due to a lack of information.

Seeing how "LOCKHART EUGENE E" had essentially all "NaN" values for his features, it made me wonder how many of the features in general had "NaN" values.  To finish off my outlier investigation, I decided to find out what percentage of each feature has "NaN" as it's value.

| Feature | Percent |
|---|:---:|
| loan_advances | 0.98 |
| restricted_stock_deferred | 0.89 |
| director_fees | 0.88 |
| deferral_payments | 0.73 |
| deferred_income | 0.66 |
| long_term_incentive | 0.55 |
| bonus | 0.43 |
| to_messages | 0.4 |
| shared_receipt_with_poi | 0.4 |
| from_messages | 0.4 |
| from_poi_to_this_person | 0.4 |
| from_this_person_to_poi | 0.4 |
| other | 0.36 |
| salary | 0.34 |
| expenses | 0.34 |
| exercised_stock_options | 0.29 |
| restricted_stock | 0.24 |
| email_address | 0.22 |
| total_payments | 0.14 |
| total_stock_value | 0.13 |
| poi | 0.0 |

So in summary, there were two definite outliers ("Total" and "The Travel Agency at the Park") and one judgment call outlier ("Eugene E Lockhart").  Everyone else on the list seems to be a real person with legitimate data points.  In addition, all of the features (with the exception of "poi") had at least 13% or more "NaN" values, and 6 features had more than 50% (with 'loan_advances' being the highest at 98%!).

## Optimize Feature Selection/Engineering

- What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]

### Create New Features

For making a new feature to test, I decided I wanted to see what percentage of someone's total emails were sent to POIs.  I called this feature "perc_poi_contact".  My reasoning is that if someone is a POI, they would likely have a higher percentage of their total emails going to other POIs than non POIs.  In order to do that, I wrote a function to create three new variables.  Two of them ("total_emails" and "total_poi_contact") were necessary to make in order to create the "perc_poi_contact" variable that I ultimately wanted.  I then plotted two of the new feature to get a look at the distribution of the data.

![created features](https://github.com/mmlak-0109/Identify-Fraud-from-Enron-Emails/blob/master/Pictures/created_features.png?raw=true)

### Feature Selection

Before I performed feature selection, I tested the full set of features, both with and without my new features, on an untuned GaussianNB, DecisionTreeClassifier, and AdaBoostClassifier.  This way I could see if my new features helped or hurt the algorithms, and to get some base numbers to compare with after I conducted feature selection and tuning (I did not perform any feature scaling because none of those algorithms require it).  I found that all but the DecisionTreeClassifier performed fairly well out of the box, but none of them could meet the minimum scoring requirements.

Here are the results:

| Algorithm              | Precision | Recall  | F1      |
|:----------------------:|:---------:|:-------:|:-------:|
|GaussianNB(w/o new features)| 0.22604 | 0.39500 | 0.28753 |
|GaussianNB(with new features)| 0.22927 | 0.41200 | 0.29460 |

| Algorithm              | Precision | Recall  | F1      |
|:----------------------:|:---------:|:-------:|:-------:|
|DecisionTreeClassifier(w/o new features)| 0.22933 | 0.22050 | 0.22483 |
|DecisionTreeClassifier(with new features)| 0.22301 | 0.22000 | 0.22150 |

| DecisionTreeClassifier Feature Ranking |
|:---:|
| 1 to_messages (0.307688506982) |
| 2 deferral_payments (0.114304029304) |
| 3 expenses (0.106292517007) | 
| 4 perc_poi_contact (0.104142857143) |
| 5 deferred_income (0.0761278195489) |

| Algorithm              | Precision | Recall  | F1      |
|:----------------------:|:---------:|:-------:|:-------:|
|AdaBoostClassifier(w/o new features)| 0.37345 | 0.28550 | 0.32360 |
|AdaBoostClassifier(with new features)| 0.34715 | 0.26800 | 0.30248 |

| AdaBoostClassifier Feature Ranking |
|:---:|
| 1 to_messages (0.18) |
| 2 deferral_payments (0.14) |
| 3 expenses (0.1) | 
| 4 perc_poi_contact (0.1) |
| 5 deferred_income (0.1) |

Well, it seems that the new features helped GaussianNB, but they hurt both DecisionTreeClassifier and AdaBoostClassifier.  The difference either way isn't that great, however, and my created feature "perc_poi_contact" is in the top 5 of the ranked features for both DecisionTreeClassifier and AdaBoostClassifier.  As a result, I'll just leave in my new features for now to see how well they do during feature selection.

For feature selection, I did both univariate and recursive feature selection using SelectKBest and RFECV.  According to the [Sklearn documentation](http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection), "Univariate feature selection works by selecting the best features based on univariate statistical tests. It can be seen as a preprocessing step to an estimator. Scikit-learn exposes feature selection routines as objects that implement the transform method."  SelectKBest will remove all but the specified "k" highest scoring features.

In order to find the best features using SelectKBest, I used GridSearchCV to test which number of "k" highest scoring features were the best.  According to the Sklearn documentation, GridSearchCV "exhaustively generates candidates from a grid of parameter values." Put another way, as it fits a chosen classifier on the dataset "all the possible combinations of parameter values are evaluated and the best combination is retained."

The parameters I used tested every number of features from 1 through 20:

- parameters = {'selector__k':[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]}

In the end, SelectKBest chose the following 5 best features with their corresponding scores:

- [('total_payments', '24.82'), ('bonus', '24.18'), ('director_fees', '20.79'), ('restricted_stock', '18.29'), ('perc_poi_contact', '11.46')]
     
        (Interestingly, even though my features didn't seem to help or hurt the algorithms that much on the tests above, one of my created features "perc_poi_contact" was selected as a best feature.)

I then tested these 5 best features on all three of the algorithms.  Besides GaussianNB, none of the results were promising. Even after some tweaking of the parameters for DecisionTreeClassifier and AdaBoostClassifier, I couldn't get Precision, Recall, and F1 scores all above 0.3:

| Algorithm              | Precision | Recall  | F1      |
|:----------------------:|:---------:|:-------:|:-------:|
| GaussianNB             | 0.42161   | 0.35900 | 0.38779 |
| DecisionTreeClassifier | 0.25057   | 0.60100 | 0.35369 |
| AdaBoostClassifier     | 0.17304   | 0.73750 | 0.28031 |

As a result, I decided to switch gears for DecisionTreeClassifier and AdaBoostClassifier. Instead of running univariate feature selection like I did above, I tried recursive feature selection with RFECV.  According to the [Sklearn documentation](http://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination), "Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and weights are assigned to each one of them. Then, features whose absolute weights are the smallest are pruned from the current set features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.  RFECV performs RFE in a cross-validation loop to find the optimal number of features."

Because RFECV is paired together with a specific algorithm, the features chosen are often different for each algorithm  The features selected for each algorithm are shown below.

DecisionTreeClassifier:

- features_list = ['poi', 'from_this_person_to_poi', 'salary', 'total_payments']

AdaBoostClassifier:

- features_list = ['poi', 'deferral_payments', 'expenses', 'from_messages', 'bonus', 'total_emails', 'from_this_person_to_poi', 'restricted_stock', 'salary', 'total_payments']

## Pick and Tune an Algorithm

- What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

- What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]

As stated previously, I tried out GaussianNB, DecisionTreeClassifier, and AdaBoostClassifier.  In order to find the best algorithm, I had to combine each algorithm with the best selected features and then tune each algorithm to find the best settings.  However, GaussianNB doesn't have any parameters to tune, so I only tuned DecisionTreeClassifier and AdaBoostClassifier.  Parameter tuning is important because many algorithms have different parameters that can be changed.  Tweaking these parameters and trying different parameter combinations can change the performance of the algorithm (sometimes drastically so).  This can be done by hand, but luckily GridSearchCV does this automatically.

The results for the tuned algorithms using SelectKBest were already presented above, so I will focus on the results for the tunning I performed on the features selected by RFECV.  For tuning with the features selected by both SelectKBest and REFCV, I here are examples of the parameters I tuned for each algorithm.

[DecisionTreeClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier):

- parameters = {'criterion':['gini', 'entropy'], 'splitter':['best', 'random'], 'max_features':[None, 'auto', 'log2'], 'max_depth':[None, 2, 4, 6, 8], 'min_samples_split':[2, 4, 6, 8], 'max_leaf_nodes':[None, 2, 4, 6, 8], 'class_weight': [None, 'balanced']}

[AdaBoostClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier):

- parameters = {'base_estimator':[None, DecisionTreeClassifier()], 'n_estimators':[1, 2, 3, 4, 5], 'learning_rate':[1.75, 1.8125, 1.875, 1.9375, 2.0], 'algorithm':['SAMME', 'SAMME.R']}

After tuning was performed for both DecisionTreeClassifier and AdaBoostClassifier using the RFECV selected best features, the best scores for each algorithm were these:

|Algorithm               | Precision | Recall  | F1      |
|:----------------------:|:---------:|:-------:|:-------:|
| DecisionTreeClassifier | 0.26839   | 0.57100 | 0.36515 |
| AdaBoostClassifier     | 0.56722   | 0.48100 | 0.52056 |

In the end I chose AdaBoostClassifier because it ended up performing the best by getting the highest scores during evaluation (which I will talk about next).

My final tuned classifier:

- AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.875, n_estimators=2,                                      random_state=None)

## Validate and Evaluate

- What is validation, and what’s a classic mistake you can make if you do it wrong? How did you validate your analysis?  [relevant rubric item: “validation strategy”]
- Give at least 2 evaluation metrics and your average performance for each of them.  Explain an interpretation of your metrics that says something human-understandable about your algorithm’s performance. [relevant rubric item: “usage of evaluation metrics”]

Validation allows the data to be slit into different datasets that the algorithms can be trained on and then tested on.  This is important for machine learning because, according to the [Sklearn documentation](http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation), "Learning the parameters of a prediction function and testing it on the same data is a methodological mistake: a model that would just repeat the labels of the samples that it has just seen would have a perfect score but would fail to predict anything useful on yet-unseen data. This situation is called overfitting. To avoid it, it is common practice when performing a (supervised) machine learning experiment to hold out part of the available data as a test set."

For validation, I ended up using StratifiedShuffleSplit.  I decided to use StratifiedShuffleSplit because it helped to ensure that I could retain a suitable proportion of POIs in both my training and testing sets due to the skewed proportions of POIs to non-POIs in the dataset, and because of the relatively small size of my dataset.

In order to evaluate the algorithms, I used Sklearn's "[classification_report](http://scikit-learn.org/stable/modules/model_evaluation.html#classification-report)" module, and it uses the [Precision](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn-metrics-precision-score), [Recall](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn-metrics-recall-score), and [F1](http://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score) scores for evaluation.  According to Sklearn, "The precision is the ratio `tp / (tp + fp)` where `tp` is the number of true positives and `fp` the number of false positives. The precision is intuitively the ability of the classifier not to label as positive a sample that is negative."  In addition, "The recall is the ratio `tp / (tp + fn)` where `tp` is the number of true positives and `fn` the number of false negatives. The recall is intuitively the ability of the classifier to find all the positive samples."  For both Precision and Recall, the best value is 1 and the worst value is 0.  In summary of these two metrics, "The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:

`F1 = 2 * (precision * recall) / (precision + recall)`.

In addition to this, I will also test each classifier on the tester function provided in tester.py.

Here were the best scores for each algorithm:

|Algorithm               | Precision | Recall  | F1      | Feature Selection Algorithm |
|:----------------------:|:---------:|:-------:|:-------:|:---------------------------:|
| GaussianNB             | 0.42161   | 0.35900 | 0.38779 | SelectKBest                 |
| DecisionTreeClassifier | 0.26839   | 0.57100 | 0.36515 | REFCV                       |
| AdaBoostClassifier     | 0.56722   | 0.48100 | 0.52056 | REFCV                       |

# Summary

In summary, the point of this project was to use machine learning to build a person of interest identifier based on financial and email data made public as a result of the Enron scandal.  During the initial exploration of the data, I found that there were a total of 146 people in the dataset (18 of which, or ~12%, are POIs) and 21 features.  I also discovered two definite outliers ("Total" and "The Travel Agency at the Park"), and I also discovered one judgment call outlier ("Eugene E Lockhart") who had "NaN" values for all his features.

Next, I moved on to selecting features and building classifiers.  First, I ran some baseline tests on my three chosen algorithms (GaussianNB, DecisionTreeClassifier, and AdaBoostClassifier) so that I could have some numbers to compare with after feature selection and tuning.  I did both univariate and recursive feature selection using SelectKBest and RFECV, and tuned and validated my algorithms using GridSearch and StratifiedShuffleSplit.  Finally, I evaluated the performance of my algorithms using the Precision, Recall, and F1 scores.  After all was said and done, I chose an AdaBoostClassifier that was tuned to work best with the best features selected by RFECV.

My chosen features:

- features_list = ['poi', 'deferral_payments', 'expenses', 'from_messages', 'bonus', 'total_emails', 'from_this_person_to_poi', 'restricted_stock', 'salary', 'total_payments']

My final tuned algorithm:

- AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.875, n_estimators=2,                                      random_state=None)

The algorithms evaluation scores:

|Algorithm               | Precision | Recall  | F1      |
|:----------------------:|:---------:|:-------:|:-------:|
| AdaBoostClassifier     | 0.56722   | 0.48100 | 0.52056 |

# Citations

- http://stackoverflow.com/questions/20577840/python-dictionary-sorting-in-descending-order-based-on-values
- http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html#sklearn.naive_bayes.GaussianNB
- http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html#sklearn.tree.DecisionTreeClassifier
- https://discussions.udacity.com/t/feature-importances-/173319
- http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html#sklearn.ensemble.AdaBoostClassifier
- http://scikit-learn.org/0.17/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html#sklearn.cross_validation.StratifiedShuffleSplit
- https://discussions.udacity.com/t/sklearn-feature-selection-stratifiedshufflesplit/199989
- https://discussions.udacity.com/t/gridsearchcv-and-stratifiedshufflesplit-giving-indexerror-list-index-out-of-range/39018
- https://discussions.udacity.com/t/validate-tester-py-function-never-completing-calculation/204371
- http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection
- http://scikit-learn.org/stable/modules/pipeline.html#pipeline-chaining-estimators
- http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV
- http://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
- https://discussions.udacity.com/t/recursive-feature-selection/204051
- https://discussions.udacity.com/t/trying-to-hit-over-0-3/196167/2
