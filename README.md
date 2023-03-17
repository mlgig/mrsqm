# MrSQM: Fast Time Series Classification with Symbolic Representations

MrSQM (Multiple Representations Sequence Miner) is a time series classifier. The 
MrSQM method can quickly extract features from multiple symbolic representations of time series and train a linear classification model with logistic regression. The method has four variants with four different feature selection strategies:

  * MrSQM-R: Random feature selection.
  * MrSQM-RS: MrSQM-R with a follow-up Chi2 test to filter less important features.
  * MrSQM-S: Pruning the all-subsequence feature space with a Chi2 bound and selecting the optimal set of top *k* subsequences.
  * MrSQM-SR: Random sampling of the features from the output of MrSQM-S.

## Installation

Dependencies
```
cython >= 0.29
numpy >= 1.18
pandas >= 1.0.3
scikit-learn >= 0.22
fftw3 (http://www.fftw.org/)
```
## Installation using pip
pip install mrsqm


## Installation from source
Download the repository: 
```
git clone https://github.com/mlgig/mrsqm.git
```
Move into the code directory of the repository: 
```
cd mrsqm/mrsqm
```
Build package from source using: 
```
pip install .
```
## Example

Load data from arff files
```
X_train,y_train = util.load_from_arff_to_dataframe("data/Coffee/Coffee_TRAIN.arff")
X_test,y_test = util.load_from_arff_to_dataframe("data/Coffee/Coffee_TEST.arff")
```
Train with MrSQM
```
clf = MrSQMClassifier(nsax=0, nsfa=5)
clf.fit(X_train,y_train)
```

Make predictions
```
predicted = clf.predict(X_test)
```

![Alt](explanation-mrsqm-example.png) 

More examples can be found in the *example* directory, including a [Jupyter Notebook](https://github.com/mlgig/mrsqm/blob/main/example/Time_Series_Classification_and_Explanation_with_MrSQM.ipynb) with detailed steps for training, prediction and 
explanation.
The full UEA and UCR Archive can be downloaded from http://www.timeseriesclassification.com/.


This repository provides supporting code, results and instructions for reproducing the work presented in our publication:

"Fast Time Series Classification with Random Symbolic Subsequences", Thach Le Nguyen and Georgiana Ifrim
https://project.inria.fr/aaltd22/files/2022/08/AALTD22_paper_5778.pdf

"MrSQM: Fast Time Series Classification with Symbolic Representations and Efficient Sequence Mining", Thach Le Nguyen and Georgiana Ifrim
https://arxiv.org/abs/2109.01036

## Citation
If you use this work, please cite as:
```
@article{mrsqm2022,
  title={Fast Time Series Classification with Random Symbolic Subsequences},
  author={Le Nguyen, Thach and Ifrim, Georgiana},
  year={2022},
  booktitle = {AALTD},
  url = {https://project.inria.fr/aaltd22/files/2022/08/AALTD22_paper_5778.pdf},
  publisher={Springer}
}
@article{mrsqm2022-extended,
  title={MrSQM: Fast Time Series Classification with Symbolic Representations and Efficient Sequence Mining},
  author={Le Nguyen, Thach and Ifrim, Georgiana},
  year={2022},
  booktitle = {arxvi},
  url = {https://arxiv.org/abs/2109.01036},
  publisher={}
}
```
