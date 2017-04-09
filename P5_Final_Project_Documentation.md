
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

For my outlier investigation, I decided to look at the features "salary and "bonus".  I plotted the data points to get a look at the distribution of the data and see if any data points popped out at me.

![outlier plot](https://github.com/mmlak-0109/Identify-Fraud-from-Enron-Emails/blob/master/Pictures/outlier.png?raw=true)

Right away it is easy to see that there is definitely something wrong here.  Most all of the data points are clumped together at the lower left and one data point is way off by itself in the top right.  As it turned out, the outlier was not a person at all, but instead was the value for the total of all of the salaries in the data set (and by extension, the total category for all the features).  After I removed the outlier, I replotted the data to see it looked.

![outlier removed](https://github.com/mmlak-0109/Identify-Fraud-from-Enron-Emails/blob/master/Pictures/outlier_removed.png?raw=true)

There, that's much better!  There are still a few features that stick out on both the high and low end of the spectrum for both salary and bonus.  The points on the high end were Enron executives "Kenneth Lay" and "Jeffery Skilling", so they are definitely legitimate data points.  On the low end, I found many legitimate data points and one interesting data point.  While looking at the list of employees I noticed a person named "THE TRAVEL AGENCY IN THE PARK".  This definitely didn't seem like an employee, so I did a quick Google search and found out that this was a [travel agency co-owned by Ken Lay's younger sister Sharon Lay](http://content.time.com/time/magazine/article/0,9171,198885,00.html) (it was renamed Alliance Worldwide in 2002).  According to Sharon Lay's [LinkedIn page](https://www.linkedin.com/in/sharon-lay-aa985b13/), she left the company in 2005.  Because the travel agency is not an Enron employee, I decided to get rid of it.

To finish off my outlier investigation, I decided to look people that had "Nan" values for the "total_payments" and "total_stock_value" features (I chose those two because they are totals for all other financial features), and discovered three employees that met this criteria.  Even though two of the three employees (William Powers and Ronnie Chan) did not receive payments of any kind, they did have some recorded email exchanges and William Powers even had 12 shared receipts with a POI.  As a result, I kept them.  The third employee named Eugene E Lockhart, however, had "Nan" values for all his features (with the exception of the "poi" feature), so I decided to remove him due to a lack of information.

## Optimize Feature Selection/Engineering

- What features did you end up using in your POI identifier, and what selection process did you use to pick them? Did you have to do any scaling? Why or why not? As part of the assignment, you should attempt to engineer your own feature that does not come ready-made in the dataset -- explain what feature you tried to make, and the rationale behind it. (You do not necessarily have to use it in the final analysis, only engineer and test it.) In your feature selection step, if you used an algorithm like a decision tree, please also give the feature importances of the features that you use, and if you used an automated feature selection function like SelectKBest, please report the feature scores and reasons for your choice of parameter values.  [relevant rubric items: “create new features”, “properly scale features”, “intelligently select feature”]

### Create New Features

For making a new feature to test, I decided I wanted to see what percentage of someone's total emails were sent to POIs.  I called this feature "perc_poi_contact".  My reasoning is that if someone is a POI, they would likely have a higher percentage of their total emails going to other POIs than non POIs.  In order to do that, I wrote a function to create three new variables.  Two of them ("total_emails" and "total_poi_contact") were necessary to make in order to create the "perc_poi_contact" variable that I ultimately wanted.  I then plotted two of the new feature to get a look at the distrobution of the data.

![created features](https://github.com/mmlak-0109/Identify-Fraud-from-Enron-Emails/blob/master/Pictures/created_features.png?raw=true)

### Feature Selection

Before I performed feature selection, I test the full set of features on and untuned GaussianNB, DecisionTreeClassifier, and AdaBoostClassifier to get some base numbers to compare with after I conducted feature selection and tuning (I did not perform any feature scaling because none of those algorithms require it).  I found that all but the DecisionTreeClassifier performed fairly well out of the box, but none of them could meet the minimum scoring requirements.

For feature selection, I did both univatriate and recursive feature selection using SelectKBest and RFECV.  According to the [Sklearn documentation](http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection), "Univariate feature selection works by selecting the best features based on univariate statistical tests. It can be seen as a preprocessing step to an estimator. Scikit-learn exposes feature selection routines as objects that implement the transform method."  SelectKBest will remove all but the specified "k" highest scoring features.

SelectKBest ended up choosing the following best features with their corresponding scores:

- [('total_payments', '24.82'), ('bonus', '24.18'), ('director_fees', '20.79'), ('restricted_stock', '18.29'), ('perc_poi_contact', '11.46')]

Next I used recursive feature selection.  According to the [Sklearn documentation](http://scikit-learn.org/stable/modules/feature_selection.html#recursive-feature-elimination), "Given an external estimator that assigns weights to features (e.g., the coefficients of a linear model), recursive feature elimination (RFE) is to select features by recursively considering smaller and smaller sets of features. First, the estimator is trained on the initial set of features and weights are assigned to each one of them. Then, features whose absolute weights are the smallest are pruned from the current set features. That procedure is recursively repeated on the pruned set until the desired number of features to select is eventually reached.  RFECV performs RFE in a cross-validation loop to find the optimal number of features."

In the end, I chose the features selected by RFECV that went together with the algorithm I ended up choosing:

- features_list = ['poi', 'deferral_payments', 'expenses', 'from_messages', 'bonus', 'total_emails', 'from_this_person_to_poi', 'restricted_stock', 'salary', 'total_payments']

## Pick and Tune an Algorithm

- What algorithm did you end up using? What other one(s) did you try? How did model performance differ between algorithms?  [relevant rubric item: “pick an algorithm”]

- What does it mean to tune the parameters of an algorithm, and what can happen if you don’t do this well?  How did you tune the parameters of your particular algorithm? (Some algorithms do not have parameters that you need to tune -- if this is the case for the one you picked, identify and briefly explain how you would have done it for the model that was not your final choice or a different model that does utilize parameter tuning, e.g. a decision tree classifier).  [relevant rubric item: “tune the algorithm”]



As stated previously, I tried out GaussianNB, DecisionTreeClassifier, and AdaBoostClassifier.  In order to find the best algorithm, I had to combine each algorithm with the best selected features and then tune each algorithm to find the best settings.  However, GaussianNB doesn't have any parameters to tune, so I only tuned DecisionTreeClassifier and AdaBoostClassifier.  Parameter tuning is important because many algorithms have different parameters that can be changed.  Tweaking these parameters and trying different parameter combinations can change the performance of the algorithm (sometimes drastically so).  This can be done by hand, but luckily there modules in Sklearn that do this automatically.  The module I used is GridSearchCV.  According to the [Sklearn documentation](http://scikit-learn.org/stable/modules/grid_search.html#exhaustive-grid-search), GridSearchCV "exhaustively generates candidates from a grid of parameter values."  Put another way, as it fits a chosen classifier on the dataset "all the possible combinations of parameter values are evaluated and the best combination is retained."

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

|                        | Precision | Recall  | F1      |
|:----------------------:|:---------:|:-------:|:-------:|
| GaussianNB             | 0.42161   | 0.35900 | 0.38779 |
| DecisionTreeClassifier | 0.26839   | 0.57100 | 0.36515 |
| AdaBoostClassifier     | 0.56722   | 0.48100 | 0.52056 |

# Summary

In summary, the point of this project was to use machine learning to build a person of interest identifier based on financial and email data made public as a result of the Enron scandal.  During the initial exploration of the data, I found that there were a total of 146 people in the dataset (18 of which, or ~12%, are POIs) and 21 features.  I also discovered two definite outliers ("Total" and "The Travel Agency at the Park"), and I also discovered one judgment call outlier ("Eugene E Lockhart") who had "Nan" values for all his features.

Next, I moved on to selecting features and building classifiers.  First, I ran some baseline tests on my three chosen algorithms (GaussianNB, DecisionTreeClassifier, and AdaBoostClassifier) so that I could have some numbers to compare with after feature selection and tuning.  I did both univariate and recursive feature selection using SelectKBest and RFECV, and tuned and validated my algorithms using GridSearch and StratifiedShuffleSplit.  Finally, I evaluated the performance of my algorithms using the Precision, Recall, and F1 scores.  After all was said and done, I chose an AdaBoostClassifier that was tuned to work best with the best features selected by RFECV.

My chosen features:

- features_list = ['poi', 'deferral_payments', 'expenses', 'from_messages', 'bonus', 'total_emails', 'from_this_person_to_poi', 'restricted_stock', 'salary', 'total_payments']

My final tuned algorithm:

- AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None, learning_rate=1.875, n_estimators=2,                                      random_state=None)

The algorithms evaluation scores:

|                        | Precision | Recall  | F1      |
|:----------------------:|:---------:|:-------:|:-------:|
| AdaBoostClassifier     | 0.56722   | 0.48100 | 0.52056 |

# Citations

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
