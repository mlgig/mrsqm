from mrsqm import MrSQMClassifier
from sklearn import metrics
import util

import subprocess

import logging
import sys
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def run_sfa(itrain, itest, otrain, otest):
    subprocess.call(['java', '-jar', 'TestSFA-all.jar', itrain, itest, otrain, otest])


def basic_example():

    X_train,y_train = util.load_from_arff_to_dataframe("data/Coffee/Coffee_TRAIN.arff")
    X_test,y_test = util.load_from_arff_to_dataframe("data/Coffee/Coffee_TEST.arff")

    clf = MrSQMClassifier(strat = 'RS') # train MrSQM-RS
    clf.fit(X_train,y_train)

    predicted = clf.predict(X_test)
    print(metrics.accuracy_score(y_test, predicted))

def mrsqm_with_sfa():
    X_train,y_train = util.load_from_arff_to_dataframe("data/Coffee/Coffee_TRAIN.arff")
    X_test,y_test = util.load_from_arff_to_dataframe("data/Coffee/Coffee_TEST.arff")

    # SFA transform
    itrain = "data/Coffee/Coffee_TRAIN.txt"
    itest = "data/Coffee/Coffee_TEST.txt"
    otrain = 'sfa.train'
    otest = 'sfa.test'
    run_sfa(itrain, itest, otrain, otest)


    clf = MrSQMClassifier(strat = 'RS') # train MrSQM-RS
    clf.fit(X_train,y_train, ext_reps = otrain) # use ext_reps to add sfa transform

    predicted = clf.predict(X_test, ext_reps = otest) # use ext_reps to add sfa transform

    print(metrics.accuracy_score(y_test, predicted))

if __name__ == "__main__":
    mrsqm_with_sfa()