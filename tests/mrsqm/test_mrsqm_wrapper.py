from mrsqm import MrSQMClassifier
import numpy as np
from pytest import approx
import pytest
from sklearn.metrics import accuracy_score

def load_coffee_dataset(): 

    # read txt files
    Xy_train = np.loadtxt("data/Coffee_TRAIN.txt")
    X_train = Xy_train[:, 1:][:, np.newaxis, :]
    y_train = Xy_train[:, 0].astype(int)
   
    Xy_test = np.loadtxt("data/Coffee_TEST.txt")
    X_test = Xy_test[:, 1:][:, np.newaxis, :] 
    y_test = Xy_test[:, 0].astype(int)

    return X_train, X_test, y_train, y_test

def test_mrsqm_coffee_accuracy():    

    # Load the coffee dataset
    X_train, X_test, y_train, y_test = load_coffee_dataset()

    # Create a MrSQMClassifier
    clf = MrSQMClassifier(nsax = 0, nsfa = 1, random_state = 42)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict(X_test)

    # Check the accuracy
    assert accuracy_score(y_test, y_pred) > 0.9

@pytest.mark.parametrize("nsax,nsfa", [(0, 1), (1, 0)])
def test_mrsqm_random_state(nsax, nsfa):
    # Load the coffee dataset
    X_train, X_test, y_train, y_test = load_coffee_dataset()

    # Create a MrSQMClassifier
    clf1 = MrSQMClassifier(nsax = nsax, nsfa = nsfa, random_state = 42)
    clf2 = MrSQMClassifier(nsax = nsax, nsfa = nsfa, random_state = 42)

    # Train the classifiers
    clf1.fit(X_train, y_train)
    clf2.fit(X_train, y_train)

    # Make predictions
    y_pred1 = clf1.predict_proba(X_test)
    y_pred2 = clf2.predict_proba(X_test)

    # Check that the predictions are the same
    assert approx(y_pred1) == y_pred2

@pytest.mark.parametrize("nsax,nsfa", [(0, 1), (1, 0)])
def test_mrsqm_serialization(nsax, nsfa):
    import pickle 
    
    # Load the dataset
    X_train, X_test, y_train, y_test = load_coffee_dataset()

    # Create a MrSQMClassifier
    clf = MrSQMClassifier(nsax = nsax, nsfa = nsfa, random_state = 42)

    # Train the classifier
    clf.fit(X_train, y_train)

    # Make predictions
    y_pred = clf.predict_proba(X_test)

   

    # Serialize the classifier
    with open("mrsqm.pkl", "wb") as f:
        pickle.dump(clf, f)

    # Deserialize the classifier
    with open("mrsqm.pkl", "rb") as f:
        clf2 = pickle.load(f)

    # Make predictions
    y_pred2 = clf2.predict_proba(X_test)
    
    assert approx(y_pred) == y_pred2

