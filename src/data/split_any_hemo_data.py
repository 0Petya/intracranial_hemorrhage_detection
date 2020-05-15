# This will isolate the labels to just the Any column (whether any hemorrhage is detected), and split the data into a 80/20 train/test split.
# The test split will be used for final evaluation in this project.

import numpy as np
import os

from sklearn.model_selection import train_test_split

np.random.seed(5102020)

if not os.path.exists("../../data/processed/any_hemo_split/"):
    os.makedirs("../../data/processed/any_hemo_split/")

ids = np.load("../../data/processed/ids.npy", allow_pickle=True)
y = np.load("../../data/processed/Y.npy", allow_pickle=True)[:, 0]

ids_train, ids_test, y_train, y_test = train_test_split(ids, y, test_size=0.2, stratify=y)

np.save("../../data/processed/any_hemo_split/X_any_hemo_ids_train.npy", ids_train)
np.save("../../data/processed/any_hemo_split/X_any_hemo_ids_test.npy", ids_test)
np.save("../../data/processed/any_hemo_split/y_any_hemo_train.npy", y_train)
np.save("../../data/processed/any_hemo_split/y_any_hemo_test.npy", y_test)
