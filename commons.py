import numpy as np
import pandas as pd
import random as rd
import logging as log
from sklearn import preprocessing

# Constants
NEIGHBORS = 7
FEATURE = "gender"
TARGET = "loan"
CATEGORICAL_COLS = ["gender", "loan"]
BALANCED_MODEL_NAME = "Balanced_Model.pkl"
IMBALANCED_MODEL_NAME = "Imbalanced_Model.pkl"
NUMERICAL_COLS = ["age", "education-num", "capital-gain", "capital-loss", "hours-per-week"]


def encode_data(df_x: pd.DataFrame, ds_y: pd.Series, le: preprocessing.LabelEncoder) -> (np.array, np.array):
    """
    Encode the data of a pandas.DataFrame and a pandas.Series by a given
    sklearn.preprocessing.LabelEncoder.
    For example: 'gender' : ['Female', 'Male'] -> 'gender' : [0, 1]
    :param df_x: DataFrame containing the features to be encoded.
    :param ds_y: Series containing the target to be encoded.
    :param le: LabelEncoder for the transformation.
    :return: A tuple containing the encoded values.
    """
    labels_x = df_x.columns.values
    num_x = []
    le_mapping = []
    # Transform features' values to numeric
    for label in labels_x:
        if not np.issubdtype(df_x[label].dtype, np.number):
            num_val = le.fit_transform(list(df_x[label]))
            num_x.append(num_val)
            le_mapping.append(dict(zip(le.classes_, le.transform(le.classes_))))
            log.debug("The value of {} was transformed numeric.".format(label))
        else:
            log.debug("The value of {} is already numeric.".format(label))
            num_x.append(df_x[label])
    # Transform target's values to numeric
    num_y = le.fit_transform(ds_y)

    le_mapping.append(dict(zip(le.classes_, le.transform(le.classes_))))
    log.debug(le_mapping)

    return np.swapaxes(np.array(num_x), 0, 1), np.array(num_y)


def get_unique_data(df_x: pd.DataFrame) -> dict:
    """
    Get all unique values for all labels from a pandas.DataFrame.
    :param df_x: DataFrame from which the values should be extracted.
    :return: Dictionary containing all unique values for each label.
    """
    unique_data = {}

    for c in df_x.columns:
        unique_data[c] = df_x[c].unique()

    return unique_data


def generate_random_data(mapping: dict, sample_size: int) -> dict:
    """
    Generates random data from a given dictionary consisting
    of labels as keys and all possible values for that label
    as a value.
    For example: 'gender' : ['Male', 'Female'].
    :param mapping: Dictionary containing feature labels and label values.
    :param sample_size: Size of the data to be generated.
    :return: A dictionary containing random examples of size {sample_size}.
    """
    random_data = {}

    for i in range(sample_size):
        for k in mapping:
            rand_v = rd.choice(mapping[k])
            if k in random_data.keys():
                random_data[k].append(rand_v)
            else:
                temp_l = [rand_v]
                random_data[k] = temp_l

    return random_data


def generate_output(data: dict,
                    sample_size: int,
                    balanced_prediction: list,
                    imbalanced_prediction: list,
                    target_values: list) -> (str, str):
    """
    Generates string comparisons between two models' predictions.
    :param data: The random data.
    :param sample_size: The size of the random data.
    :param balanced_prediction: Predictions from the balanced model.
    :param imbalanced_prediction: Predictions from the imbalanced model.
    :param target_values: All possible unique target values.
    :return: Tuple, where the first element is a filtered string containing only the differences and the second
    string contains all the comparisons.
    """
    str_fil = ""
    str_full = ""
    str3 = "\n******************************************************************"

    for i in range(sample_size):
        ans_differs = balanced_prediction[i] != imbalanced_prediction[i]
        str1 = "\nPerson {}: ".format(i + 1)
        str_fil = str_fil + str1 if ans_differs else str_fil
        str_full = str_full + str1

        for k in data:
            str2 = "\n\t{} -> {}".format(k, (data[k])[i])
            str_fil = str_fil + str2 if ans_differs else str_fil
            str_full = str_full + str2

        str_fil = str_fil + str3 if ans_differs else str_fil
        str_full = str_full + str3
        str4 = "\n\t\tBalanced prediction for Person {} is {}.\n\t\tImbalanced prediction for Person {} is {}."\
            .format(i+1, target_values[imbalanced_prediction[i]], i+1, target_values[balanced_prediction[i]])
        str_fil = str_fil + str4 if ans_differs else str_fil
        str_full = str_full + str4
        str_fil = str_fil + str3 if ans_differs else str_fil
        str_full = str_full + str3

    return str_fil, str_full
