import pickle
import commons
import pandas as pd
import logging as log
from xai import data
from sklearn import preprocessing

# Constants
SAMPLE_SIZE = 500

# Configure logger
log.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
log.getLogger().setLevel(log.INFO)

# Prepare data
df = data.load_census()

# Load models trained with balanced and imbalanced data.
balanced_model = pickle.load(open(commons.BALANCED_MODEL_NAME, 'rb'))
imbalanced_model = pickle.load(open(commons.IMBALANCED_MODEL_NAME, 'rb'))

# Generate random test data
sample_data = commons.generate_random_data(commons.get_unique_data(df.drop(commons.TARGET, axis=1)),
                                           SAMPLE_SIZE)
random_df = pd.DataFrame(data=sample_data)

# Encode the data
le_train = preprocessing.LabelEncoder()
test_x, _ = commons.encode_data(random_df,
                                df[commons.TARGET],
                                le_train)

# Predict the random generated data.
balanced_prediction = balanced_model.predict(test_x)
imbalanced_prediction = imbalanced_model.predict(test_x)

# Compare the models' results.
filtered, full = commons.generate_output(sample_data,
                                         SAMPLE_SIZE,
                                         balanced_prediction=balanced_prediction,
                                         imbalanced_prediction=imbalanced_prediction,
                                         target_values=df[commons.TARGET].unique())

log.debug(full)
log.info(filtered)
