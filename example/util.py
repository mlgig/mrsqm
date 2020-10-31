import numpy as np
#from scipy.io import arff
import pandas as pd


uea_dir = '/home/thachln/myworkdir/data/UEA_2018_univariate_arff/'

# code taken from https://github.com/alan-turing-institute/sktime/blob/master/sktime/utils/load_data.py
def load_from_arff_to_dataframe(
    full_file_path_and_name,
    has_class_labels=True,
    return_separate_X_and_y=True,
    replace_missing_vals_with="NaN",
):
    instance_list = []
    class_val_list = []

    data_started = False
    is_multi_variate = False
    is_first_case = True

    with open(full_file_path_and_name, "r") as f:
        for line in f:

            if line.strip():
                if (
                    is_multi_variate is False
                    and "@attribute" in line.lower()
                    and "relational" in line.lower()
                ):
                    is_multi_variate = True

                if "@data" in line.lower():
                    data_started = True
                    continue

                # if the 'data tag has been found, the header information
                # has been cleared and now data can be loaded
                if data_started:
                    line = line.replace("?", replace_missing_vals_with)

                    if is_multi_variate:
                        if has_class_labels:
                            line, class_val = line.split("',")
                            class_val_list.append(class_val.strip())
                        dimensions = line.split("\\n")
                        dimensions[0] = dimensions[0].replace("'", "")

                        if is_first_case:
                            for _d in range(len(dimensions)):
                                instance_list.append([])
                            is_first_case = False

                        for dim in range(len(dimensions)):
                            instance_list[dim].append(
                                pd.Series(
                                    [float(i) for i in dimensions[dim].split(",")]
                                )
                            )

                    else:
                        if is_first_case:
                            instance_list.append([])
                            is_first_case = False

                        line_parts = line.split(",")
                        if has_class_labels:
                            instance_list[0].append(
                                pd.Series(
                                    [
                                        float(i)
                                        for i in line_parts[: len(line_parts) - 1]
                                    ]
                                )
                            )
                            class_val_list.append(line_parts[-1].strip())
                        else:
                            instance_list[0].append(
                                pd.Series(
                                    [float(i) for i in line_parts[: len(line_parts)]]
                                )
                            )

    x_data = pd.DataFrame(dtype=np.float32)
    for dim in range(len(instance_list)):
        x_data["dim_" + str(dim)] = instance_list[dim]

    if has_class_labels:
        if return_separate_X_and_y:
            return x_data, np.asarray(class_val_list)
        else:
            x_data["class_vals"] = pd.Series(class_val_list)

    return x_data

def read_data(input_file):    
    if input_file.lower().endswith(".csv"):
        train_data = np.genfromtxt(input_file,delimiter=',')
        X = train_data[:,1:]
        y = train_data[:,0]
    elif input_file.lower().endswith(".arff"):
        #from sktime.utils.load_data import load_from_arff_to_dataframe
        X,y = load_from_arff_to_dataframe(input_file)

    return X, y


# def read_arff(arff_file):
#     data = arff.loadarff(arff_file)
#     X = []
#     y = []
#     for ts in data[0]:
#         ts_as_list = ts.tolist()
#         X.append(list(ts_as_list[:-1]))
#         y.append(ts_as_list[-1])
    
#     return X,y



def get_uea_path(ds):
    train_file = uea_dir + ds + '/' + ds + '_TRAIN.arff' 
    test_file = uea_dir + ds + '/' + ds + '_TEST.arff'
    return train_file,test_file


def get_ucr_data(ds):
    train_file,test_file = get_ucr_path(ds)
    train_x, train_y = read_data(train_file)
    test_x, test_y = read_data(test_file)

    return train_x, train_y, test_x, test_y



def np_array_to_df(nparray):
    X = pd.DataFrame()
    X['dim_0'] = [pd.Series(x) for x in nparray]
    return X

def load_uea_arff_data(ds):    
    train_file, test_file = get_uea_path(ds)
    train_x, train_y = read_data(train_file)
    test_x, test_y = read_data(test_file)

    return train_x, train_y, test_x, test_y



def read_reps_from_file(inputf):
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

def load_uea_sfa_reps(ds):
    sfa_dir = "/home/thachln/myworkdir/experiments/sfa_x2_uea/"
    return read_reps_from_file(sfa_dir + ds + '/sfa.train'), read_reps_from_file(sfa_dir + ds + '/sfa.test')

