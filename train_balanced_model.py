import commons
import pandas as pd
import logging as log
import xai
import pickle
from xai import data
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

# Constants
MIN_PER_GROUP = 600
MAX_PER_GROUP = 600

# Configure logger
log.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
log.getLogger().setLevel(log.INFO)

# Remove DataFrame display limitation
pd.set_option('display.max_columns', None)

# Prepare the data
df = data.load_census()
df_x = df.drop(commons.TARGET, axis=1)
ds_y = df[commons.TARGET]

xai.imbalance_plot(df,
                   commons.FEATURE,
                   commons.TARGET,
                   threshold=0.50,
                   categorical_cols=commons.CATEGORICAL_COLS)

''' Balanced train-test split exactly 600 examples of the target y (loan) and 
    the column gender from the input data x.  '''
x_train_balanced, y_train_balanced, x_valid_balanced, y_valid_balanced, train_idx, test_idx = \
    xai.balanced_train_test_split(
                                df_x, ds_y, commons.FEATURE,
                                min_per_group=MIN_PER_GROUP,
                                max_per_group=MAX_PER_GROUP,
                                categorical_cols=commons.CATEGORICAL_COLS)

log.debug("Data was split successfully in test and train data with length respectively: {} and {}."
          .format(len(x_train_balanced), len(x_valid_balanced)))

# Visualise the imbalances of gender and the target
plot_valid_balanced = x_valid_balanced.copy()
plot_valid_balanced[commons.TARGET] = y_valid_balanced

xai.imbalance_plot(plot_valid_balanced,
                   commons.CATEGORICAL_COLS[0],
                   commons.CATEGORICAL_COLS[1],
                   categorical_cols=commons.CATEGORICAL_COLS)

# Encode data
le_train = preprocessing.LabelEncoder()
x_train, y_train = commons.encode_data(x_train_balanced, y_train_balanced, le_train)
le_valid = preprocessing.LabelEncoder()
x_valid, y_valid = commons.encode_data(x_valid_balanced, y_valid_balanced, le_valid)

# Create a KNN Classifier and train it.
model = KNeighborsClassifier(n_neighbors=commons.NEIGHBORS)
model.fit(x_train, y_train)
probabilities = model.predict(x_valid)
acc = model.score(x_valid, y_valid)
log.info("Accuracy for the model is {}".format(acc))

# Save the model
pickle.dump(model, open(commons.BALANCED_MODEL_NAME, 'wb'))

# Plot ROC and Confusion Matrix for this model
xai.confusion_matrix_plot(y_valid, probabilities)
xai.roc_plot(y_valid, probabilities)
xai.roc_plot(
    y_valid,
    probabilities,
    df=pd.DataFrame(data=x_valid, columns=df_x.columns.values),
    cross_cols=["gender"],
    categorical_cols=["gender"])
