from mrsqm import MrSQMClassifier
from sklearn import metrics
import util

import logging
import sys
#logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)

def basic_example():

    X_train,y_train = util.load_from_arff_to_dataframe("data/Coffee/Coffee_TRAIN.arff")
    X_test,y_test = util.load_from_arff_to_dataframe("data/Coffee/Coffee_TEST.arff")

    # train MrSQM-R with SAX only
    clf = MrSQMClassifier(strat = 'R')
    clf.fit(X_train,y_train)
    predicted = clf.predict(X_test)
    print("MrSQM-R accuracy: " + str(metrics.accuracy_score(y_test, predicted)))

    # train MrSQM-RS with SAX only
    clf = MrSQMClassifier(strat = 'RS') # default
    clf.fit(X_train,y_train)
    predicted = clf.predict(X_test)
    print("MrSQM-RS accuracy: " + str(metrics.accuracy_score(y_test, predicted)))

    # train MrSQM-S with SAX only
    clf = MrSQMClassifier(strat = 'S')
    clf.fit(X_train,y_train)
    predicted = clf.predict(X_test)
    print("MrSQM-S accuracy: " + str(metrics.accuracy_score(y_test, predicted)))

    # train MrSQM-SR with SAX only
    clf = MrSQMClassifier(strat = 'SR')
    clf.fit(X_train,y_train)
    predicted = clf.predict(X_test)
    print("MrSQM-SR accuracy: " + str(metrics.accuracy_score(y_test, predicted)))



def mrsqm_with_sfa():
    X_train,y_train = util.load_from_arff_to_dataframe("data/Coffee/Coffee_TRAIN.arff")
    X_test,y_test = util.load_from_arff_to_dataframe("data/Coffee/Coffee_TEST.arff")



    # train MrSQM-RS with SFA only
    clf = MrSQMClassifier(nsax = 0, nsfa = 5) # train MrSQM-RS
    clf.fit(X_train,y_train) # use ext_rep to add sfa transform
    predicted = clf.predict(X_test) # use ext_rep to add sfa transform
    print("MrSQM-RS with SFA only accuracy: " + str(metrics.accuracy_score(y_test, predicted)))

    # train MrSQM-RS with both SAX and SFA
    clf = MrSQMClassifier(nsax = 5, nsfa = 5) # train MrSQM-RS
    clf.fit(X_train,y_train) # use ext_rep to add sfa transform
    predicted = clf.predict(X_test) # use ext_rep to add sfa transform
    print("MrSQM-RS with both SAX and SFA accuracy: " + str(metrics.accuracy_score(y_test, predicted)))

if __name__ == "__main__":
    basic_example()
    mrsqm_with_sfa() # require running jar file
