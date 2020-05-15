# This is used to train a singular model once and get its results.
# It takes a model name (function names under ./candidates.py) as an argument.
# It will perform a random stratified sampling of observations (size defined under the model) from the training data.
# Used to quickly test different models and identify candidates for further evaluation.
# Results and loss curves are under ../../models/quick_evaluation/

import sys
sys.path.insert(0, "../data/")

import candidates
import multiprocessing
import matplotlib.pyplot as plt
import numpy as np
import os

from data_generator import DataGenerator
from sklearn.metrics import accuracy_score, classification_report, fbeta_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

model_name = sys.argv[1]

cores = multiprocessing.cpu_count()

if not os.path.exists("../../models/quick_evaluation/quick_evaluation.csv"):
    file = open("../../models/quick_evaluation/quick_evaluation.csv", 'w')
    file.write("Candidate,F2,Accuracy,Sensitivity,Specificity\n")
    file.close()

ids = np.load("../../data/processed/any_hemo_split/X_any_hemo_ids_train.npy", allow_pickle=True)
y = np.load("../../data/processed/any_hemo_split/y_any_hemo_train.npy", allow_pickle=True)

model, epochs, batch_size, callbacks, sample_size, threshold = getattr(candidates, model_name)()
print(model.summary())

# This performs random stratified sampling of observations.
_, ids, _, y = train_test_split(ids, y, test_size=sample_size/len(ids), stratify=y)

ids, ids_test, y, y_test = train_test_split(ids, y, test_size=0.2, stratify=y)
ids_train, ids_val, y_train, y_val = train_test_split(ids, y, test_size=0.2, stratify=y)
class_weights = dict(enumerate(class_weight.compute_class_weight("balanced", np.unique(y_train), y_train)))

train_gen = DataGenerator(ids_train, y_train, path="../../data/processed/images", batch_size=batch_size)
val_gen = DataGenerator(ids_val, y_val, path="../../data/processed/images", batch_size=batch_size)
test_gen = DataGenerator(ids_test, y_test, path="../../data/processed/images", batch_size=batch_size)

history = model.fit(train_gen, epochs=epochs, callbacks=callbacks, validation_data=val_gen, class_weight=class_weights, workers=cores)

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title(f"{model_name} Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"], loc="upper left")
plt.savefig(f"../../models/quick_evaluation/{model_name}_loss.png")
plt.close()

y_pred = model.predict(test_gen) > threshold

f2 = fbeta_score(y_test, y_pred, 2)
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)

print("F2 score: ", f2)
print(classification_report(y_test, y_pred))

file = open("../../models/quick_evaluation/quick_evaluation.csv", 'a')
file.write(f"{model_name},{f2},{accuracy},{recall},{precision}\n")
file.close()
