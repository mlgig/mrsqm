from mrsqm import MrSQMClassifier
from sklearn import metrics
import util

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

X_train,y_train = util.load_from_arff_to_dataframe("data/Coffee/Coffee_TRAIN.arff")
X_test,y_test = util.load_from_arff_to_dataframe("data/Coffee/Coffee_TEST.arff")

#X_train, y_train,X_test,y_test = util.load_uea_arff_data("GunPoint")

clf = MrSQMClassifier(strat = 'SR')
clf.fit(X_train,y_train)

predicted = clf.predict(X_test)
metrics.accuracy_score(y_test, predicted)