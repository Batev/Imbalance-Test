import commons
import pandas as pd
import logging as log
import xai
import pickle
from xai import data
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

# Configure logger
log.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
log.getLogger().setLevel(log.INFO)

# Remove DataFrame display limitation
pd.set_option('display.max_columns', None)

# Prepare the data
df = xai.data.load_census()
df_x = df.drop(columns=[commons.TARGET], axis=1)
ds_y = df[commons.TARGET]

xai.imbalance_plot(df,
                   commons.FEATURE,
                   commons.TARGET,
                   threshold=0.50,
                   categorical_cols=commons.CATEGORICAL_COLS)

''' Balanced train-test split exactly 300 examples of the target y (loan) and 
    the column gender from the input data x.  '''
x_train_imbalanced, x_valid_imbalanced, y_train_imbalanced, y_valid_imbalanced = \
    train_test_split(df_x, ds_y, test_size=0.2, random_state=33)

# Visualise the imbalances of gender and the target
plot_valid_balanced = x_valid_imbalanced.copy()
plot_valid_balanced[commons.TARGET] = y_valid_imbalanced
xai.imbalance_plot(plot_valid_balanced,
                   commons.CATEGORICAL_COLS[0],
                   commons.CATEGORICAL_COLS[1],
                   categorical_cols=commons.CATEGORICAL_COLS)

# Encode data
le_train = preprocessing.LabelEncoder()
x_train, y_train = commons.encode_data(x_train_imbalanced, y_train_imbalanced, le_train)
le_valid = preprocessing.LabelEncoder()
x_valid, y_valid = commons.encode_data(x_valid_imbalanced, y_valid_imbalanced, le_valid)

# Create a KNN Classifier and train it.
model = KNeighborsClassifier(n_neighbors=commons.NEIGHBORS)
model.fit(x_train, y_train)
probabilities = model.predict(x_valid)
acc = model.score(x_valid, y_valid)
log.info("Accuracy for the model is {}".format(acc))

# Save the model
pickle.dump(model, open(commons.IMBALANCED_MODEL_NAME, 'wb'))

# Plot ROC and Confusion Matrix for this model
xai.confusion_matrix_plot(y_valid, probabilities)
xai.roc_plot(y_valid, probabilities)
xai.roc_plot(
    y_valid,
    probabilities,
    df=pd.DataFrame(data=x_valid, columns=df_x.columns.values),
    cross_cols=["gender"],
    categorical_cols=["gender"])
