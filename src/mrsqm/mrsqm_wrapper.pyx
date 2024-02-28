from libcpp.string cimport string
from libcpp cimport bool
from libcpp.vector cimport vector

import math
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression, RidgeClassifierCV

from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold



import logging

def debug_logging(message):
    logging.info(message)


def check_X_format(X):
    """Check the format of the input data.

    Parameters
    ----------
    X : nested pandas DataFrame or nested pandas Series
        The input data.

    Returns
    -------
    X : pandas DataFrame
        The input data in tabular format.
    """
    if isinstance(X, (pd.DataFrame, pd.Series)):
        X_3d = np.stack(X.applymap(lambda cell:cell.to_numpy()).apply(lambda row: np.stack(row),axis=1).to_numpy())
        return X_3d
    elif isinstance(X, np.ndarray):
        if X.ndim == 3:
            return X
        elif X.ndim == 2:
            return X[:, np.newaxis, :]
        else:
            raise ValueError(
                f"X must be 2D or 3D, but found: {X.ndim}D"
            )
    else:
        raise ValueError(
            f"X must be pandas DataFrame, pandas Series or numpy array, "
            f"but found: {type(X)}"
        )

    return None
        
    

######################### SAX #########################

cdef extern from "sax_converter.h":
    cdef cppclass SAX:
        SAX(int, int, int, int)        
        vector[string] timeseries2SAX(vector[double])
        vector[double] map_weighted_patterns(vector[double], vector[string], vector[double])

cdef class PySAX:
    '''
    Wrapper of SAX C++ implementation.
    '''
    cdef SAX * thisptr      # hold a C++ instance which we're wrapping

    def __cinit__(self, int N, int w, int a, int di = 1):
        self.thisptr = new SAX(N, w, a, di)

    def __dealloc__(self):
        del self.thisptr

    def timeseries2SAX(self, ts):
        return self.thisptr.timeseries2SAX(ts)
        

    def timeseries2SAXseq(self, ts):
        words = self.thisptr.timeseries2SAX(ts)
        seq = b''
        
        for w in words:
            seq = seq + b' ' + w
        if seq:  # remove extra space
            seq = seq[1:]
        return seq

    def map_weighted_patterns(self, ts, sequences, weights):
        return self.thisptr.map_weighted_patterns(ts, sequences, weights)

###########################################################################

cdef extern from "sfa/SFAWrapper.cpp":
    cdef cppclass SFAWrapper:
        SFAWrapper(int, int, int, bool, bool)        
        void fit(vector[vector[double]])
        vector[string] transform(vector[vector[double]])
        vector[string] transform_with_lookuptable(vector[vector[double]], vector[vector[double]])
        vector[vector[double]] get_lookuptable()
    # cdef void printHello()


cdef class PySFA:
    '''
    Wrapper of SFA C++ implementation.
    '''
    cdef SFAWrapper * thisptr      # hold a C++ instance which we're wrapping

    def __cinit__(self, int N, int w, int a, bool norm, bool normTS):
        self.thisptr = new SFAWrapper(N, w + 1, a, norm, normTS) # w+1 to skip second coef 

    def __dealloc__(self):
        del self.thisptr

    def fit(self, X):
        self.thisptr.fit(X)
        return self

    def transform(self, X):
        return self.thisptr.transform(X)

    def transform_with_lookuptable(self, X, lookuptable): # no need to fit 
        return self.thisptr.transform_with_lookuptable(X, lookuptable)

    def get_lookuptable(self):
        return self.thisptr.get_lookuptable()



#########################SQM wrapper#########################


cdef extern from "strie.cpp":
    cdef cppclass SeqTrie:
        SeqTrie(vector[string])
        vector[int] search(string)


cdef extern from "sqminer.h":
    cdef cppclass SQMiner:
        SQMiner(double, double)
        vector[string] mine(vector[string] &, vector[int] &)

cdef class PyFeatureTrie:
    cdef SeqTrie *thisptr

    def __cinit__(self, vector[string] sequences):
        self.thisptr = new SeqTrie(sequences)
    def __dealloc__(self):
        del self.thisptr

    def search(self, string sequence):
        return self.thisptr.search(sequence)


cdef class PySQM:
    cdef SQMiner *thisptr

    def __cinit__(self, double selection, double threshold):
        self.thisptr = new SQMiner(selection,threshold)
    def __dealloc__(self):
        del self.thisptr

    def mine(self, vector[string] sequences, vector[int] labels):
        return self.thisptr.mine(sequences, labels)     

######################### MrSQM Transformer #########################
class MrSQMTransformer:
    '''     
    Overview: Transformer implementation of MrSQM.
    

    Parameters
    ----------
    
    strat               : str, feature selection strategy, either 'R','S','SR', or 'RS'. R and S are single-stage filters while RS and SR are two-stage filters. By default set to 'RS'.
    features_per_rep    : int, (maximum) number of features selected per representation. By deafault set to 500.
    selection_per_rep   : int, (maximum) number of candidate features selected per representation. Only applied in two stages strategies (RS and SR). By deafault set to 2000.
    nsax                : int, control the number of representations produced by sax transformation.
    nsfa                : int, control the number of representations produced by sfa transformation.
    custom_config       : dict, customized parameters for the symbolic transformation.
    random_state        : set random seed for classifier. By default 'none'.
    sfa_norm            : to be passed to sfa transformer.
    first_diff          : randomly use of first difference in transformation.

    '''

    def __init__(self, strat = 'RS', features_per_rep = 500, selection_per_rep = 2000, nsax = 0, nsfa = 1, custom_config=None, random_state = None, sfa_norm = True, first_diff = True):
        

        self.nsax = nsax
        self.nsfa = nsfa
        self.first_diff = first_diff
        self.sfa_norm = sfa_norm
        self.random_state = random_state        
        self.custom_config = custom_config        
        self.strat = strat
        self.features_per_rep = features_per_rep
        self.selection_per_rep = selection_per_rep
        

        self.fitted = False
        self.multivariate = False       
        
     

    def create_pars(self, min_ws, max_ws, xrep, random_sampling=False, is_sfa=False):
        pars = []      
        if xrep > 0:      
            if random_sampling:                   

                debug_logging("Sampling window size, word length, and alphabet size.")       

                candidates = []
                for wd in [int(2**(w/xrep)) for w in range(3*xrep,xrep*int(np.log2(max_ws))+ 1)]:
                    for wo in [7,8]:
                        for als in [2,3,4,5]:
                            if wo <= (wd + 1):
                                candidates.append([wd,wo,als])

                nrep = xrep*int(np.log2(max_ws))                
                if nrep >= len(candidates):
                    pars = candidates
                else:
                    selected = self.rng.choice(range(len(candidates)), replace=False, size = nrep)
                    pars = [candidates[i] for i in selected]
                
               
            else:
                #debug_logging("Doubling the window while fixing word length and alphabet size.")                   
                #pars = [[int(2**(w/xrep)),8,4] for w in range(3*xrep,xrep*int(np.log2(max_ws))+ 1)]     
                #if not is_sfa:
                pars = [[w,8,4] for w in range(8,max_ws,int(np.sqrt(max_ws)))]     

        debug_logging("Symbolic Parameters: " + str(pars))      
            
        
        return pars            
   

    def transform_time_series(self, X):
        debug_logging("Transform time series to symbolic representations.")
        
        multi_tssr = []   

        #ts_x_array = from_nested_to_2d_array(ts_x).values
        X_diff = np.diff(X, axis=2, prepend=0)
     
        if not self.config:
            self.config = []
            
            min_ws = 16
            # min_len = max_len = X.shape[2]
            # for a in ts_x.iloc[:, 0]:
            #     min_len = min(min_len, len(a)) 
            #     max_len = max(max_len, len(a))
            # max_ws = (min_len + max_len)//2            
            max_ws = X.shape[2]
            
            
            pars = self.create_pars(min_ws, max_ws, self.nsax, random_sampling=True, is_sfa=False)                        
            for p in pars:
                self.config.append(
                        {'method': 'sax', 'window': p[0], 'word': p[1], 'alphabet': p[2], 
                        # 'dilation': np.int32(2 ** np.random.uniform(0, np.log2((min_len - 1) / (p[0] - 1))))})
                        'dilation': 1})
            
            pars = self.create_pars(min_ws, max_ws, self.nsfa, random_sampling=True, is_sfa=True)                        
            for p in pars:
                self.config.append(
                        {'method': 'sfa', 
                        'window': p[0], 
                        'word': p[1], 
                        'alphabet': p[2] , 
                        'normSFA': False, 
                        'normTS': self.sfa_norm,
                        'diff': self.rng.choice([True,False]),
                        'signature':[]
                        })        

        
        for cfg in self.config:
            for i in range(X.shape[1]):
                tssr = []

                if cfg['method'] == 'sax':  # convert time series to SAX                    
                    ps = PySAX(cfg['window'], cfg['word'], cfg['alphabet'], cfg['dilation'])
                    for ts in X[:,i,:]:
                        sr = ps.timeseries2SAXseq(ts)
                        tssr.append(sr)
                elif  cfg['method'] == 'sfa':    
                    if cfg['diff'] and self.first_diff:
                        X_tmp = X_diff     
                    else:               
                        X_tmp = X                               
                    if len(cfg['signature']) < (i+1):
                        cfg['signature'].append(PySFA(cfg['window'], cfg['word'], cfg['alphabet'], cfg['normSFA'], cfg['normTS']).fit(X_tmp[:,i,:]).get_lookuptable())

                    tssr = PySFA(cfg['window'], cfg['word'], cfg['alphabet'], cfg['normSFA'], cfg['normTS']).transform_with_lookuptable(X_tmp[:,i,:], cfg['signature'][i])
                multi_tssr.append(tssr)        

        return multi_tssr
  
    def sample_random_sequences(self, seqs, min_length, max_length, max_n_seq):  
                
        output = {} # because set doesn't maintain order
        splitted_seqs = [s.split(b' ') for s in seqs]
        n_input = len(seqs)       

        # while len(output) < n_seq: #infinity loop if sequences are too alike
        for i in range(0, max_n_seq):
            did = self.rng.integers(0,high=n_input)
            wid = self.rng.integers(0,high=len(splitted_seqs[did]))
            word = splitted_seqs[did][wid]
            
            s_length = self.rng.integers(min_length, high=min(len(word) + 1, max_length + 1))
            start = self.rng.integers(0,high=len(word) - s_length + 1)        
            sampled = word[start:(start + s_length)]
            output[sampled] = 1            
        
        return list(output.keys())

    def feature_selection_on_train(self, mr_seqs, y):
        debug_logging("Compute train data in subsequence space.")
        full_fm = []
        self.filters = []

        for i in range(0,len(mr_seqs)):
        #for rep, seq_features in zip(mr_seqs, self.sequences):            
            rep = mr_seqs[i]
            seq_features = self.sequences[i]
            fm = np.zeros((len(rep), len(seq_features)),dtype = np.int32)            
            ft = PyFeatureTrie(seq_features)
            for ii,s in enumerate(rep):                
                fm[ii,:] = ft.search(s)            
                
            fm = fm > 0 # binary only
            

            fs = SelectKBest(chi2, k=min(self.features_per_rep, fm.shape[1]))
            if self.strat == 'RS':
                debug_logging("Filter subsequences of this representation with chi2 (only with RS).")
                fm = fs.fit_transform(fm, y)
                
                self.sequences[i] = [seq_features[ii] for ii in fs.get_support(indices=True)]               


            self.filters.append(fs)
            full_fm.append(fm)


        #full_fm = np.hstack(full_fm)
        
        return np.hstack(full_fm)

    def feature_selection_on_test(self, mr_seqs):
        debug_logging("Compute test data in subsequence space.")
        full_fm = []
        

        for rep, seq_features, fs in zip(mr_seqs, self.sequences, self.filters):            
            fm = np.zeros((len(rep), len(seq_features)),dtype = np.int32)
            ft = PyFeatureTrie(seq_features)
            for i,s in enumerate(rep):
                fm[i,:] = ft.search(s)
            fm = fm > 0 # binary only

            # if self.strat == 'RS':
            #     fm = fs.transform(fm)        
            full_fm.append(fm)


        #full_fm = np.hstack(full_fm)
        #return self.final_vt.transform(full_fm)
        return np.hstack(full_fm)
    

    def mine(self,rep, int_y):        
        mined_subs = []
        if self.strat == 'S':
            debug_logging("Select " + str(self.features_per_rep) + " discrimative subsequences with SQM.")
            miner = PySQM(self.features_per_rep,0.0)
            mined_subs = miner.mine(rep, int_y)

        elif self.strat == 'SR':
            debug_logging("Select " + str(self.selection_per_rep) + " discrimative subsequences with SQM.")
            miner = PySQM(self.selection_per_rep,0.0)
            mined_subs = miner.mine(rep, int_y)
            debug_logging("Randomly pick " + str(self.features_per_rep) + " subsequences from the list.")
            mined_subs = self.rng.permutation(mined_subs)[:self.features_per_rep].tolist()

        elif self.strat == 'R':
            debug_logging("Random sampling " + str(self.features_per_rep) + " subsequences from this symbolic representation.")
            mined_subs = self.sample_random_sequences(rep,3,16,self.features_per_rep)
        elif self.strat == 'RS':
            debug_logging("Random sampling " + str(self.selection_per_rep) + " subsequences from this symbolic representation.")
            mined_subs = self.sample_random_sequences(rep,3,16,self.selection_per_rep)

        debug_logging("Found " + str(len(mined_subs)) + " unique subsequences.")
        return mined_subs



    def fit(self, X, y):
        self.rng = np.random.default_rng(self.random_state)
        if self.custom_config is None:
            self.config = [] # http://effbot.org/zone/default-values.htm
        else:
            self.config = self.custom_config
        X = check_X_format(X)
        self.classes_ = []
        self.filters = [] # feature filters (one filter for a rep) for test data transformation

        debug_logging("Fit training data.")
        self.classes_ = np.unique(y) #because sklearn also uses np.unique

        int_y = [np.where(self.classes_ == c)[0][0] for c in y]

        self.sequences = []

        debug_logging("Search for subsequences.")        
        mr_seqs = self.transform_time_series(X)
        
        
        
        for rep in mr_seqs:
            mined = self.mine(rep,int_y)
            self.sequences.append(mined)  
        
        
        
        self.feature_selection_on_train(mr_seqs, int_y)   
        self.fitted = True    
        self.multivariate = X.shape[1] > 1

        return self
    
    def fit_transform(self, X, y):
        self.rng = np.random.default_rng(self.random_state)
        if self.custom_config is None:
            self.config = [] # http://effbot.org/zone/default-values.htm
        else:
            self.config = self.custom_config
        X = check_X_format(X)
        self.classes_ = []
        self.filters = [] # feature filters (one filter for a rep) for test data transformation

        debug_logging("Fit training data.")
        self.classes_ = np.unique(y) #because sklearn also uses np.unique

        int_y = [np.where(self.classes_ == c)[0][0] for c in y]

        self.sequences = []

        debug_logging("Search for subsequences.")        
        mr_seqs = self.transform_time_series(X)
        
        
        
        for rep in mr_seqs:
            mined = self.mine(rep,int_y)
            self.sequences.append(mined)   
        
        
        debug_logging("Compute feature vectors.")
        train_x = self.feature_selection_on_train(mr_seqs, int_y)    

        self.fitted = True    
        self.multivariate = X.shape[1] > 1

        return train_x

    def transform(self,X):
        X = check_X_format(X)

        mr_seqs = self.transform_time_series(X)       
        X_transform = self.feature_selection_on_test(mr_seqs)
        return X_transform

    
    def get_saliency_map(self, ts, coefs):

        if not self.fitted:
            return None
        if self.multivariate: # only work with univariate data for the time being
            return None
        
        is_multiclass = len(self.classes_) > 2
        weighted_ts = np.zeros((len(self.classes_), len(ts)))

        fi = 0
        for cfg, features in zip(self.config, self.sequences):
            if cfg['method'] == 'sax':
                ps = PySAX(cfg['window'], cfg['word'], cfg['alphabet'])
                if is_multiclass:
                    for ci, cl in enumerate(self.classes_):
                        weighted_ts[ci, :] += ps.map_weighted_patterns(
                            ts, features, coefs[ci, fi:(fi+len(features))])
                else:
                    # because classes_[1] is the positive class
                    weighted_ts[1, :] += ps.map_weighted_patterns(
                        ts, features, coefs[0, fi:(fi+len(features))])

            fi += len(features)
        if not is_multiclass:
            weighted_ts[0, :] = -weighted_ts[1, :]
        return weighted_ts

    def get_params(self, deep=True):
        return {'strat': self.strat, 
                'features_per_rep': self.features_per_rep, 
                'selection_per_rep': self.selection_per_rep, 
                'nsax': self.nsax, 
                'nsfa': self.nsfa, 
                'custom_config': self.custom_config, 
                'random_state': self.random_state, 
                'sfa_norm': self.sfa_norm, 
                'first_diff': self.first_diff,
                }
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        setattr(self, 'fitted', False)
        setattr(self, 'multivariate', False)
        return self



######################### MrSQM Classifier #########################

class MrSQMClassifier:
    '''     
    Overview: MrSQM is an efficient time series classifier utilizing symbolic representations of time series. MrSQM implements four different feature selection strategies (R,S,RS,SR) that can quickly select subsequences from multiple symbolic representations of time series data.
    def __init__(self, strat = 'RS', features_per_rep = 500, selection_per_rep = 2000, nsax = 1, nsfa = 0, custom_config=None, random_state = None, sfa_norm = True):

    Parameters
    ----------
    
    strat               : str, feature selection strategy, either 'R','S','SR', or 'RS'. R and S are single-stage filters while RS and SR are two-stage filters. By default set to 'RS'.
    features_per_rep    : int, (maximum) number of features selected per representation. By deafault set to 500.
    selection_per_rep   : int, (maximum) number of candidate features selected per representation. Only applied in two stages strategies (RS and SR). By deafault set to 2000.
    nsax                : int, control the number of representations produced by sax transformation.
    nsfa                : int, control the number of representations produced by sfa transformation.
    custom_config       : dict, customized parameters for the symbolic transformation.
    random_state        : set random seed for classifier. By default 'none'.
    sfa_norm            : to be passed to sfa transformer.
    first_diff          : randomly use of first difference in transformation.


    '''
    def __init__(self, strat = 'RS', features_per_rep = 500, selection_per_rep = 2000, nsax = 0, nsfa = 1, custom_config=None, random_state = None, sfa_norm = True, first_diff = True):
        self.nsax = nsax
        self.nsfa = nsfa
        self.first_diff = first_diff
        self.sfa_norm = sfa_norm
        self.random_state = random_state        
        self.custom_config = custom_config
        self.strat = strat   
        self.features_per_rep = features_per_rep
        self.selection_per_rep = selection_per_rep
        

        self.fitted = False
        

        debug_logging("Initialize MrSQM Classifier.")
        debug_logging("Feature Selection Strategy: " + strat)
        debug_logging("SAX Reps: " + str(self.nsax) + "x")
        debug_logging("SFA Reps: " + str(self.nsfa) + "x")
        debug_logging("Number of features per rep: " + str(self.features_per_rep))
        debug_logging("Number of candidates per rep (only for SR and RS):" + str(self.selection_per_rep))
        
    
    def fit(self, X, y):
        self.transformer = MrSQMTransformer(strat = self.strat, features_per_rep = self.features_per_rep, selection_per_rep = self.selection_per_rep, nsax = self.nsax, nsfa = self.nsfa, custom_config = self.custom_config, random_state = self.random_state, sfa_norm = self.sfa_norm, first_diff = self.first_diff)
        debug_logging("Fit training data.")        
        train_x = self.transformer.fit_transform(X,y)
        debug_logging("Fit logistic regression model.")
        self.clf = LogisticRegression(solver='newton-cg',multi_class = 'multinomial', class_weight='balanced', random_state=0).fit(train_x, y)        
        # self.clf = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True).fit(train_x, y)        
        self.classes_ = self.clf.classes_ # shouldn't matter  
        self.fitted = True
        return self


    def predict_proba(self, X):         
        return self.clf.predict_proba(self.transformer.transform(X)) 

    def predict(self, X):        
        return self.clf.predict(self.transformer.transform(X))

    def decision_function(self, X):        
        return self.clf.decision_function(self.transformer.transform(X))

    def get_saliency_map(self, ts):
        return self.transformer.get_saliency_map(ts,self.clf.coef_)

    def get_params(self, deep=True):
        return {'strat': self.strat, 
                'features_per_rep': self.features_per_rep, 
                'selection_per_rep': self.selection_per_rep, 
                'nsax': self.nsax, 
                'nsfa': self.nsfa, 
                'custom_config': self.custom_config, 
                'random_state': self.random_state, 
                'sfa_norm': self.sfa_norm, 
                'first_diff': self.first_diff,
                }
    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        setattr(self, 'fitted', False)
        return self

    def score(self, X, y, sample_weight=None):
        return self.clf.score(self.transformer.transform(X), y, sample_weight)


    
