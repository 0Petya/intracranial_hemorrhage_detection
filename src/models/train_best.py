# This is used to train the final model with a very large sample size.
# This will be analyzed and evaluated on the hold-out test set in ../../notebooks/evaluation.ipynb

import sys
sys.path.insert(0, "../data/")

import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import os

from data_generator import DataGenerator
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

cores = multiprocessing.cpu_count()

if not os.path.exists("../../models/"):
    os.makedirs("../../models/")

ids = np.load("../../data/processed/any_hemo_split/X_any_hemo_ids_train.npy", allow_pickle=True)
y = np.load("../../data/processed/any_hemo_split/y_any_hemo_train.npy", allow_pickle=True)

# This performs random stratified sampling of observations.
_, ids, _, y = train_test_split(ids, y, test_size=300000/len(ids), stratify=y)

ids_train, ids_val, y_train, y_val = train_test_split(ids, y, test_size=0.2, stratify=y)
class_weights = dict(enumerate(class_weight.compute_class_weight("balanced", np.unique(y_train), y_train)))

train_gen = DataGenerator(ids_train, y_train, path="../../data/processed/images", batch_size=16)
val_gen = DataGenerator(ids_val, y_val, path="../../data/processed/images", batch_size=16)

model = Sequential()
model.add(BatchNormalization(input_shape=(512, 512, 1)))
model.add(Conv2D(64, (6, 6), activation="relu"))
model.add(MaxPooling2D(pool_size=(3, 3)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(1, activation="sigmoid"))
model.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])

callbacks = [EarlyStopping(monitor="val_loss", patience=5, mode="min", restore_best_weights=True)]
history = model.fit(train_gen, epochs=10, callbacks=callbacks, validation_data=val_gen, class_weight=class_weights, workers=cores)
model.save("../../models/best_model")

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Best Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"], loc="upper left")
plt.savefig("../../models/best_model_loss.png")
plt.close()
