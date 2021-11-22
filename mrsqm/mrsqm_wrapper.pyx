from libcpp.string cimport string
from libcpp cimport bool
from libcpp.vector cimport vector

import numpy as np
import pandas as pd
from numpy.random import randint
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV

from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold

from sktime.datatypes._panel._convert import from_nested_to_2d_array

import logging

def debug_logging(message):
    logging.info(message)




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
    cdef void printHello()

cdef class PySFA:
    '''
    Wrapper of SFA C++ implementation.
    '''
    cdef SFAWrapper * thisptr      # hold a C++ instance which we're wrapping

    def __cinit__(self, int N, int w, int a, bool norm, bool normTS):
        self.thisptr = new SFAWrapper(N, w, a, norm, normTS)

    def __dealloc__(self):
        del self.thisptr

    def fit(self, X):
        self.thisptr.fit(X)
        return self

    def transform(self, X):
        return self.thisptr.transform(X)



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


######################### MrSQM Classifier #########################

class MrSQMClassifier:    
    '''     
    Overview: MrSQM is an efficient time series classifier utilizing symbolic representations of time series. MrSQM implements four different feature selection strategies (R,S,RS,SR) that can quickly select subsequences from multiple symbolic representations of time series data.
    
    Parameters
    ----------
    
    strat               : str, feature selection strategy, either 'R','S','SR', or 'RS'. R and S are single-stage filters while RS and SR are two-stage filters.
    
    use_sax             : bool, whether to use the sax transformation. if False, ext_rep must be provided in the fitting and predicting stage.
    
    custom_config       : dict, customized parameters for the symbolic transformation.

    features_per_rep    : int, (maximum) number of features selected per representation.

    selection_per_rep   : int, (maximum) number of candidate features selected per representation. Only applied in two stages strategies (RS and SR)

    xrep                : int, control the number of representations produced by sax transformation.

    '''

    def __init__(self, strat = 'SR', features_per_rep = 1000, selection_per_rep = 2000, nsax = 1, nsfa = 1, custom_config=None):

        # self.use_sax = use_sax
        # self.use_sfa = use_sfa

        self.nsax = nsax
        self.nsfa = nsfa

        if custom_config is None:
            self.config = [] # http://effbot.org/zone/default-values.htm
        else:
            self.config = custom_config

        self.strat = strat   

        # all the unique labels in the data
        # in case of binary data the first one is always the negative class
        self.classes_ = []
        self.clf = None # scikit-learn model       

        self.fpr = features_per_rep
        self.spr = selection_per_rep
        
        self.filters = [] # feature filters (one filter for a rep) for test data transformation

        debug_logging("Initialize MrSQM Classifier.")
        debug_logging("Feature Selection Strategy: " + strat)
        debug_logging("SAX Reps: " + str(self.nsax) + "x")
        debug_logging("SFA Reps: " + str(self.nsfa) + "x")
        debug_logging("Number of features per rep: " + str(self.fpr))
        debug_logging("Number of candidates per rep (only for SR and RS):" + str(self.spr))
        
     

    def create_pars(self, min_ws, max_ws, xrep, random_sampling=False, is_sfa=False):
        pars = []      
        if xrep > 0:      
            if random_sampling:    
                debug_logging("Sampling window size, word length, and alphabet size.")       
                ws_choices = [int(2**(w/xrep)) for w in range(3*xrep,xrep*int(np.log2(max_ws))+ 1)]            
                wl_choices = [6,8,10,12,14,16]
                if is_sfa:
                    wl_choices = [6,8,10,12,14] # can't handle 16x6 case
                alphabet_choices = [3,4,5,6]

                nrep = xrep*int(np.log2(max_ws))
                #if not is_sfa: # reduce number of sax rep
                #    nrep = nrep // 2       
                    
                #for w in range(3*self.xrep,self.xrep*int(np.log2(max_ws))+ 1):
                for w in range(nrep):
                    pars.append([np.random.choice(ws_choices) , np.random.choice(wl_choices), np.random.choice(alphabet_choices)])
            else:
                #debug_logging("Doubling the window while fixing word length and alphabet size.")                   
                #pars = [[int(2**(w/xrep)),8,4] for w in range(3*xrep,xrep*int(np.log2(max_ws))+ 1)]     
                #if not is_sfa:
                pars = [[w,8,4] for w in range(8,max_ws,int(np.sqrt(max_ws)))]     

        debug_logging("Symbolic Parameters: " + str(pars))      
            
        
        return pars            
            
            

    def transform_time_series(self, ts_x):
        debug_logging("Transform time series to symbolic representations.")
        
        multi_tssr = []   

        ts_x_array = from_nested_to_2d_array(ts_x).values
        
     
        if not self.config:
            self.config = []
            
            min_ws = 16
            min_len = max_len = len(ts_x.iloc[0, 0])
            for a in ts_x.iloc[:, 0]:
                min_len = min(min_len, len(a)) 
                max_len = max(max_len, len(a))
            max_ws = (min_len + max_len)//2            
            
            
            pars = self.create_pars(min_ws, max_ws, self.nsax, random_sampling=False, is_sfa=False)            
            for p in pars:
                self.config.append(
                        {'method': 'sax', 'window': p[0], 'word': p[1], 'alphabet': p[2], 
                        # 'dilation': np.int32(2 ** np.random.uniform(0, np.log2((min_len - 1) / (p[0] - 1))))})
                        'dilation': 1})
            
            pars = self.create_pars(min_ws, max_ws, self.nsfa, random_sampling=False, is_sfa=True)            
            for p in pars:
                self.config.append(
                        {'method': 'sfa', 'window': p[0], 'word': p[1], 'alphabet': p[2] , 'normSFA': False, 'normTS': True
                        })        

        
        for cfg in self.config:
            for i in range(ts_x.shape[1]):
                tssr = []

                if cfg['method'] == 'sax':  # convert time series to SAX                    
                    ps = PySAX(cfg['window'], cfg['word'], cfg['alphabet'], cfg['dilation'])
                    for ts in ts_x.iloc[:,i]:
                        sr = ps.timeseries2SAXseq(ts)
                        tssr.append(sr)
                elif  cfg['method'] == 'sfa':
                    if 'signature' not in cfg:
                        cfg['signature'] = PySFA(cfg['window'], cfg['word'], cfg['alphabet'], cfg['normSFA'], cfg['normTS']).fit(ts_x_array)
                    
                    tssr = cfg['signature'].transform(ts_x_array)
                multi_tssr.append(tssr)        

        return multi_tssr
  
    def sample_random_sequences(self, seqs, min_length, max_length, max_n_seq):  
                
        output = set()
        splitted_seqs = [s.split(b' ') for s in seqs]
        n_input = len(seqs)       

        # while len(output) < n_seq: #infinity loop if sequences are too alike
        for i in range(0, max_n_seq):
            did = randint(0,n_input)
            wid = randint(0,len(splitted_seqs[did]))
            word = splitted_seqs[did][wid]
            
            s_length = randint(min_length, min(len(word) + 1, max_length + 1))
            start = randint(0,len(word) - s_length + 1)        
            sampled = word[start:(start + s_length)]
            output.add(sampled)
        
        return list(output)

    def feature_selection_on_train(self, mr_seqs, y):
        debug_logging("Compute train data in subsequence space.")
        full_fm = []
        self.filters = []

        for rep, seq_features in zip(mr_seqs, self.sequences):            
            fm = np.zeros((len(rep), len(seq_features)),dtype = np.int32)
            ft = PyFeatureTrie(seq_features)
            for i,s in enumerate(rep):
                fm[i,:] = ft.search(s)            
            fm = fm > 0 # binary only

            fs = SelectKBest(chi2, k=min(self.fpr, fm.shape[1]))
            if self.strat == 'RS':
                debug_logging("Filter subsequences of this representation with chi2 (only with RS).")
                fm = fs.fit_transform(fm, y)

            self.filters.append(fs)
            full_fm.append(fm)


        full_fm = np.hstack(full_fm)

        #self.final_vt = VarianceThreshold()
        #return self.final_vt.fit_transform(full_fm)
        return full_fm

    def feature_selection_on_test(self, mr_seqs):
        debug_logging("Compute test data in subsequence space.")
        full_fm = []
        

        for rep, seq_features, fs in zip(mr_seqs, self.sequences, self.filters):            
            fm = np.zeros((len(rep), len(seq_features)),dtype = np.int32)
            ft = PyFeatureTrie(seq_features)
            for i,s in enumerate(rep):
                fm[i,:] = ft.search(s)
            fm = fm > 0 # binary only

            if self.strat == 'RS':
                fm = fs.transform(fm)        
            full_fm.append(fm)


        full_fm = np.hstack(full_fm)
        #return self.final_vt.transform(full_fm)
        return full_fm

    def read_reps_from_file(self, inputf):
        last_cfg = None
        mr_seqs = []
        rep = []
        i = 0
        for l in open(inputf,"r"):
            i += 1
            l_splitted = bytes(l,'utf-8').split(b" ")
            cfg = l_splitted[0]
            seq = b" ".join(l_splitted[2:])
            if cfg == last_cfg:
                rep.append(seq)
            else:
                last_cfg = cfg
                if rep:
                    mr_seqs.append(rep)
                rep = [seq]
        if rep:
            mr_seqs.append(rep)    
        return mr_seqs

    def mine(self,rep, int_y):        
        mined_subs = []
        if self.strat == 'S':
            debug_logging("Select " + str(self.fpr) + " discrimative subsequences with SQM.")
            miner = PySQM(self.fpr,0.0)
            mined_subs = miner.mine(rep, int_y)

        elif self.strat == 'SR':
            debug_logging("Select " + str(self.spr) + " discrimative subsequences with SQM.")
            miner = PySQM(self.spr,0.0)
            mined_subs = miner.mine(rep, int_y)
            debug_logging("Randomly pick " + str(self.fpr) + " subsequences from the list.")
            mined_subs = np.random.permutation(mined_subs)[:self.fpr].tolist()

        elif self.strat == 'R':
            debug_logging("Random sampling " + str(self.fpr) + " subsequences from this symbolic representation.")
            mined_subs = self.sample_random_sequences(rep,3,16,self.fpr)
        elif self.strat == 'RS':
            debug_logging("Random sampling " + str(self.spr) + " subsequences from this symbolic representation.")
            mined_subs = self.sample_random_sequences(rep,3,16,self.spr)

        debug_logging("Found " + str(len(mined_subs)) + " unique subsequences.")
        return mined_subs



    def fit(self, X, y):
        debug_logging("Fit training data.")
        self.classes_ = np.unique(y) #because sklearn also uses np.unique

        int_y = [np.where(self.classes_ == c)[0][0] for c in y]

        self.sequences = []

        debug_logging("Search for subsequences.")        
        mr_seqs = self.transform_time_series(X)
        
        
        
        for rep in mr_seqs:
            mined = self.mine(rep,int_y)
            self.sequences.append(mined)


    
        # first computing the feature vectors
        # then fit the new data to a logistic regression model
        
        debug_logging("Compute feature vectors.")
        train_x = self.feature_selection_on_train(mr_seqs, int_y)
        
        debug_logging("Fit logistic regression model.")
        self.clf = LogisticRegression(solver='newton-cg',multi_class = 'multinomial', class_weight='balanced').fit(train_x, y)        
        # self.clf = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10), normalize = True).fit(train_x, y)        
        self.classes_ = self.clf.classes_ # shouldn't matter  


    def predict_proba(self, X): 
        mr_seqs = self.transform_time_series(X)       
        test_x = self.feature_selection_on_test(mr_seqs)
        return self.clf.predict_proba(test_x) 

    def predict(self, X):
        mr_seqs = self.transform_time_series(X)       
        test_x = self.feature_selection_on_test(mr_seqs)
        return self.clf.predict(test_x)





 






