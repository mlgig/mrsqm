from libcpp.string cimport string
from libcpp cimport bool
from libcpp.vector cimport vector

import numpy as np
import pandas as pd
from numpy.random import randint
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV

from sklearn.feature_selection import SelectKBest, chi2, VarianceThreshold, f_classif

from sklearn.feature_selection import chi2, f_classif
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.tree import DecisionTreeClassifier

from sfa import _dilation, _binning_dft, _mft
from sktime.utils.validation.panel import check_X

import math
import sys
import logging

def debug_logging(message):
    logging.info(message)

def from_nested_to_2d_array(X, return_numpy=False):
    """Convert nested Panel to 2D numpy Panel.

    Convert nested pandas DataFrame or Series with NumPy arrays or
    pandas Series in cells into tabular
    pandas DataFrame with primitives in cells, i.e. a data frame with the
    same number of rows as the input data and
    as many columns as there are observations in the nested series. Requires
    series to be have the same index.

    Parameters
    ----------
    X : nested pd.DataFrame or nested pd.Series
    return_numpy : bool, default = False
        - If True, returns a NumPy array of the tabular data.
        - If False, returns a pandas DataFrame with row and column names.

    Returns
    -------
     Xt : pandas DataFrame
        Transformed DataFrame in tabular format
    """
    # TODO does not handle dataframes with nested series columns *and*
    #  standard columns containing only primitives

    # convert nested data into tabular data
    if isinstance(X, pd.Series):
        Xt = np.array(X.tolist())

    elif isinstance(X, pd.DataFrame):
        try:
            Xt = np.hstack([X.iloc[:, i].tolist() for i in range(X.shape[1])])

        # except strange key error for specific case
        except KeyError:
            if (X.shape == (1, 1)) and (X.iloc[0, 0].shape == (1,)):
                # in fact only breaks when an additional condition is met,
                # namely that the index of the time series of a single value
                # does not start with 0, e.g. pd.RangeIndex(9, 10) as is the
                # case in forecasting
                Xt = X.iloc[0, 0].values
            else:
                raise

        if Xt.ndim != 2:
            raise ValueError(
                "Tabularization failed, it's possible that not "
                "all series were of equal length"
            )

    else:
        raise ValueError(
            f"Expected input is pandas Series or pandas DataFrame, "
            f"but found: {type(X)}"
        )

    if return_numpy:
        return Xt

    Xt = pd.DataFrame(Xt)

    # create column names from time index
    if X.ndim == 1:
        time_index = (
            X.iloc[0].index
            if hasattr(X.iloc[0], "index")
            else np.arange(X.iloc[0].shape[0])
        )
        columns = [f"{X.name}__{i}" for i in time_index]

    else:
        columns = []
        for colname, col in X.items():
            time_index = (
                col.iloc[0].index
                if hasattr(col.iloc[0], "index")
                else np.arange(col.iloc[0].shape[0])
            )
            columns.extend([f"{colname}__{i}" for i in time_index])

    Xt.index = X.index
    Xt.columns = columns
    return Xt



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




class AdaptedSFA:
    """Adapted from SFADilation."""

    def __init__(
        self,
        word_length=8,
        alphabet_size=4,
        window_size=12,
        norm=False,
        binning_method="equi-depth",
        anova=False,
        variance=False,
        bigrams=False,
        skip_grams=False,
        remove_repeat_words=False,
        lower_bounding=True,
        save_words=False,
        dilation=0,
        first_difference=False,
        feature_selection="none",
        sections=1,
        max_feature_count=256,
        p_threshold=0.05,
        random_state=None,
        return_sparse=True,
        return_pandas_data_series=False,
        n_jobs=1,
    ):
        self.words = []
        self.breakpoints = []

        # we cannot select more than window_size many letters in a word
        self.word_length = word_length

        self.alphabet_size = alphabet_size
        self.window_size = window_size

        self.norm = norm
        self.lower_bounding = lower_bounding
        self.inverse_sqrt_win_size = (
            1.0 / math.sqrt(window_size) if not lower_bounding else 1.0
        )

        self.remove_repeat_words = remove_repeat_words

        self.save_words = save_words

        self.binning_method = binning_method
        self.anova = anova
        self.variance = variance

        self.bigrams = bigrams
        self.skip_grams = skip_grams
        self.n_jobs = n_jobs
        self.sections = sections

        self.n_instances = 0
        self.series_length = 0
        self.letter_bits = 0

        self.dilation = dilation
        self.first_difference = first_difference

        # Feature selection part
        self.feature_selection = feature_selection
        self.max_feature_count = max_feature_count
        self.feature_count = 0
        self.relevant_features = None

        # feature selection is applied based on the chi-squared test.
        self.p_threshold = p_threshold

        self.return_sparse = return_sparse
        self.return_pandas_data_series = return_pandas_data_series

        self.random_state = random_state  


    def fit(self, X, y=None):
        offset = 2 if self.norm else 0
        self.word_length_actual = min(self.window_size - offset, self.word_length)
        self.dft_length = (
            self.window_size - offset
            if (self.anova or self.variance) is True
            else self.word_length_actual
        )
        # make dft_length an even number (same number of reals and imags)
        self.dft_length = self.dft_length + self.dft_length % 2
        self.word_length_actual = self.word_length_actual + self.word_length_actual % 2

        self.support = np.arange(self.word_length_actual)
        self.letter_bits = np.uint32(math.ceil(math.log2(self.alphabet_size)))
        # self.word_bits = self.word_length_actual * self.letter_bits

        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
        X = X.squeeze(1)

        if self.dilation >= 1 or self.first_difference:
            X2, self.X_index = _dilation(X, self.dilation, self.first_difference)
        else:
            X2, self.X_index = X, np.arange(X.shape[-1])

        self.n_instances, self.series_length = X2.shape
        self.breakpoints = self._binning(X2, y)
        
        return self

    def transform(self,X):
        X = check_X(X, enforce_univariate=True, coerce_to_numpy=True)
        X = X.squeeze(1)

        if self.dilation >= 1 or self.first_difference:
            X2, self.X_index = _dilation(X, self.dilation, self.first_difference)
        else:
            X2, self.X_index = X, np.arange(X.shape[-1])

        dfts = _mft(
            X2,
            self.window_size,
            self.dft_length,
            self.norm,
            self.support,
            self.anova,
            self.variance,
            self.inverse_sqrt_win_size,
            self.lower_bounding,
            )
        all_seqs = []
        
        for seq in dfts:
            sfa_str = b''
            for dft in seq:
                if sfa_str:
                    sfa_str += b' '
                
                first_char = ord(b'A')
                for i in range(self.word_length):
                    for bp in range(self.alphabet_size):
                        if dft[i] <= self.breakpoints[i][bp]:
                            sfa_str += bytes([first_char + bp])
                            break
                    first_char += self.alphabet_size
            all_seqs.append(sfa_str)
        return all_seqs

    def _binning(self, X, y=None):
        dft = _binning_dft(
            X,
            self.window_size,
            self.series_length,
            self.dft_length,
            self.norm,
            self.inverse_sqrt_win_size,
            self.lower_bounding,
        )

        if y is not None:
            y = np.repeat(y, dft.shape[0] / len(y))

        if self.variance and y is not None:
            # determine variance
            dft_variance = np.var(dft, axis=0)

            # select word-length-many indices with the largest variance
            self.support = np.argsort(-dft_variance)[: self.word_length_actual]

            # sort remaining indices
            self.support = np.sort(self.support)

            # select the Fourier coefficients with highest f-score
            dft = dft[:, self.support]
            self.dft_length = np.max(self.support) + 1
            self.dft_length = self.dft_length + self.dft_length % 2  # even

        if self.anova and y is not None:
            non_constant = np.where(
                ~np.isclose(dft.var(axis=0), np.zeros_like(dft.shape[1]))
            )[0]

            # select word-length many indices with best f-score
            if self.word_length_actual <= non_constant.size:
                f, _ = f_classif(dft[:, non_constant], y)
                self.support = non_constant[np.argsort(-f)][: self.word_length_actual]

            # sort remaining indices
            self.support = np.sort(self.support)

            # select the Fourier coefficients with highest f-score
            dft = dft[:, self.support]
            self.dft_length = np.max(self.support) + 1
            self.dft_length = self.dft_length + self.dft_length % 2  # even

        if self.binning_method == "information-gain":
            return self._igb(dft, y)
        elif self.binning_method == "kmeans" or self.binning_method == "quantile":
            return self._k_bins_discretizer(dft)
        else:
            return self._mcb(dft)

    def _k_bins_discretizer(self, dft):
        encoder = KBinsDiscretizer(
            n_bins=self.alphabet_size, strategy=self.binning_method
        )
        encoder.fit(dft)
        if encoder.bin_edges_.ndim == 1:
            breaks = encoder.bin_edges_.reshape((-1, 1))
        else:
            breaks = encoder.bin_edges_
        breakpoints = np.zeros((self.word_length_actual, self.alphabet_size))

        for letter in range(self.word_length_actual):
            for bp in range(1, len(breaks[letter]) - 1):
                breakpoints[letter, bp - 1] = breaks[letter, bp]

        breakpoints[:, self.alphabet_size - 1] = sys.float_info.max
        return breakpoints

    def _mcb(self, dft):
        breakpoints = np.zeros((self.word_length_actual, self.alphabet_size))

        dft = np.round(dft, 2)
        for letter in range(self.word_length_actual):
            column = np.sort(dft[:, letter])
            bin_index = 0

            # use equi-depth binning
            if self.binning_method == "equi-depth":
                target_bin_depth = len(dft) / self.alphabet_size

                for bp in range(self.alphabet_size - 1):
                    bin_index += target_bin_depth
                    breakpoints[letter, bp] = column[int(bin_index)]

            # use equi-width binning aka equi-frequency binning
            elif self.binning_method == "equi-width":
                target_bin_width = (column[-1] - column[0]) / self.alphabet_size

                for bp in range(self.alphabet_size - 1):
                    breakpoints[letter, bp] = (bp + 1) * target_bin_width + column[0]

        breakpoints[:, self.alphabet_size - 1] = sys.float_info.max
        return breakpoints

    def _igb(self, dft, y):
        breakpoints = np.zeros((self.word_length_actual, self.alphabet_size))
        clf = DecisionTreeClassifier(
            criterion="entropy",
            max_depth=np.uint32(np.log2(self.alphabet_size)),
            max_leaf_nodes=self.alphabet_size,
            random_state=1,
        )

        for i in range(self.word_length_actual):
            clf.fit(dft[:, i][:, None], y)
            threshold = clf.tree_.threshold[clf.tree_.children_left != -1]
            for bp in range(len(threshold)):
                breakpoints[i, bp] = threshold[bp]
            for bp in range(len(threshold), self.alphabet_size):
                breakpoints[i, bp] = np.inf

        return np.sort(breakpoints, axis=1)
        
    def timeseries2SFAseq(self, ts):
        """Convert time series to SFA sequence."""
        dfts = self.sfa._mft(ts)
        sfa_str = b''
        for window in range(dfts.shape[0]):
            if sfa_str:
                sfa_str += b' '
            dft = dfts[window]
            first_char = ord(b'A')
            for i in range(self.sfa.word_length):
                for bp in range(self.sfa.alphabet_size):
                    if dft[i] <= self.sfa.breakpoints[i][bp]:
                        sfa_str += bytes([first_char + bp])
                        break
                first_char += self.sfa.alphabet_size
        return sfa_str


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
    ts_norm             : time series normalisation (standardisation). By default set to 'True'.

    '''

    def __init__(self, 
        strat = 'RS', 
        features_per_rep = 500, 
        selection_per_rep = 2000, 
        nsax = 1, 
        nsfa = 0, 
        custom_config=None, 
        random_state = None, 
        sfa_norm = True, 
        use_dilation = False, 
        use_first_diff = False
        ):
        

        self.nsax = nsax
        self.nsfa = nsfa

        self.sfa_norm = sfa_norm
        if random_state is not None:
            np.random.seed(random_state)
        # self.random_state = (
        #     np.int32(random_state) if isinstance(random_state, int) else None
        # )

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
        
        self.use_dilation = use_dilation
        self.use_first_diff = use_first_diff

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
                diff_choices = [True, False]

                nrep = xrep*int(np.log2(max_ws))                                
                
                for w in range(nrep):
                    window_size = np.random.choice(ws_choices)
                    dilation = np.maximum(
                        1,
                        np.int32(2 ** np.random.uniform(0, np.log2((max_ws - 1) / (window_size - 1)))), # max_ws == series_length
                    )
                    word_length = np.random.choice(wl_choices)
                    if is_sfa:
                        word_length = min(word_length,window_size)
                    pars.append([window_size , word_length , np.random.choice(alphabet_choices), dilation, np.random.choice(diff_choices)])
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

        # ts_x_array = from_nested_to_2d_array(ts_x).values
        
        # X = check_X(ts_x, enforce_univariate=True, coerce_to_numpy=True).squeeze(1)
        
        if not self.config:
            self.config = []
            
            min_ws = 16
            min_len = max_len = len(ts_x.iloc[0, 0])
            for a in ts_x.iloc[:, 0]:
                min_len = min(min_len, len(a)) 
                max_len = max(max_len, len(a))
            max_ws = (min_len + max_len)//2            
            
            
            pars = self.create_pars(min_ws, max_ws, self.nsax, random_sampling=True, is_sfa=False)            
            for p in pars:
                self.config.append(
                        {'method': 'sax', 'window': p[0], 'word': p[1], 'alphabet': p[2], 
                        # 'dilation': np.int32(2 ** np.random.uniform(0, np.log2((min_len - 1) / (p[0] - 1))))})
                        'dilation': 1})
            
            pars = self.create_pars(min_ws, max_ws, self.nsfa, random_sampling=True, is_sfa=True)            
            for p in pars:
                self.config.append(
                        {'method': 'sfa', 'window': p[0], 'word': p[1], 'alphabet': p[2] , 'normSFA': False, 'normTS': self.sfa_norm, 'dilation': p[3], 'first_diff': p[4]
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
                    # print(f"Config:{cfg['window']}-{cfg['word']}-{cfg['alphabet']}")
                    dilation = cfg['dilation']
                    if not self.use_dilation:
                        dilation = 0
                    
                    first_diff = cfg['first_diff']
                    if not self.use_first_diff:
                        first_diff = False
                    # dilated_ts_x = _dilation(X,cfg['dilation'],False)[0]
                    if 'signature' not in cfg:                        
                        # cfg['signature'] = PySFA(cfg['window'], cfg['word'], cfg['alphabet'], cfg['normSFA'], cfg['normTS']).fit(dilated_ts_x)
                        cfg['signature'] = AdaptedSFA(
                            window_size=cfg['window'], 
                            word_length=cfg['word'], #
                            alphabet_size=cfg['alphabet'],
                            dilation = dilation,
                            first_difference = first_diff
                            ).fit(ts_x)
                    
                    tssr = cfg['signature'].transform(ts_x)
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

        for i in range(0,len(mr_seqs)):
        #for rep, seq_features in zip(mr_seqs, self.sequences):            
            rep = mr_seqs[i]
            seq_features = self.sequences[i]
            fm = np.zeros((len(rep), len(seq_features)),dtype = np.int32)
            ft = PyFeatureTrie(seq_features)
            for ii,s in enumerate(rep):
                fm[ii,:] = ft.search(s)            
            fm = fm > 0 # binary only

            fs = SelectKBest(chi2, k=min(self.fpr, fm.shape[1]))
            if self.strat == 'RS':
                debug_logging("Filter subsequences of this representation with chi2 (only with RS).")
                fm = fs.fit_transform(fm, y)
                self.sequences[i] = [seq_features[ii] for ii in fs.get_support(indices=True)]               


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

            # if self.strat == 'RS':
            #     fm = fs.transform(fm)        
            full_fm.append(fm)


        full_fm = np.hstack(full_fm)
        #return self.final_vt.transform(full_fm)
        return full_fm
    

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

        return self


    def predict_proba(self, X): 
        mr_seqs = self.transform_time_series(X)       
        test_x = self.feature_selection_on_test(mr_seqs)
        return self.clf.predict_proba(test_x) 

    def predict(self, X):
        mr_seqs = self.transform_time_series(X)       
        test_x = self.feature_selection_on_test(mr_seqs)
        return self.clf.predict(test_x)

    def decision_function(self, X):
        mr_seqs = self.transform_time_series(X)       
        test_x = self.feature_selection_on_test(mr_seqs)
        return self.clf.decision_function(test_x)

    def get_saliency_map(self, ts):        

        is_multiclass = len(self.classes_) > 2
        weighted_ts = np.zeros((len(self.classes_), len(ts)))

        fi = 0
        for cfg, features in zip(self.config, self.sequences):
            if cfg['method'] == 'sax':
                ps = PySAX(cfg['window'], cfg['word'], cfg['alphabet'])
                if is_multiclass:
                    for ci, cl in enumerate(self.classes_):
                        weighted_ts[ci, :] += ps.map_weighted_patterns(
                            ts, features, self.clf.coef_[ci, fi:(fi+len(features))])
                else:
                    # because classes_[1] is the positive class
                    weighted_ts[1, :] += ps.map_weighted_patterns(
                        ts, features, self.clf.coef_[0, fi:(fi+len(features))])

            fi += len(features)
        if not is_multiclass:
            weighted_ts[0, :] = -weighted_ts[1, :]
        return weighted_ts
        





 






