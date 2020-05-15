# Here is where we define all the models we are evaluating.
# Notes on each model's performance are above the function definition.
# Each model returns: the model itself, how many epochs to train for, the batch size, a list of callbacks, the number of samples to use, precision/recall threshold.

from tensorflow.keras.layers import BatchNormalization, Conv2D, Dense, Dropout, Flatten, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

# For such a simple model, it works suprisingly well. It's certainly not great though, as expected. The loss curve plateaus too quickly.
def m1():
    m = Sequential()
    m.add(BatchNormalization(input_shape=(512, 512, 1)))
    m.add(Flatten())
    m.add(Dense(32, activation="relu"))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m, 10, 32, [], 10000, 0.5

# This is much better. Though it looks like the model overfits very quickly. Maybe some dropout layers will work well.
# It also looks like it can be further improved with additional training.
def m2():
    m = Sequential()
    m.add(BatchNormalization(input_shape=(512, 512, 1)))
    m.add(Conv2D(64, (3, 3), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(64, (3, 3), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Flatten())
    m.add(Dense(64, activation="relu"))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m, 10, 16, [], 10000, 0.5

# That didn't work well. It looks like we might be loosing too much information and isn't really able to learn.
# Two things we can try: simplifying the model and increasing the sample size.
def m3():
    m = Sequential()
    m.add(BatchNormalization(input_shape=(512, 512, 1)))
    m.add(Conv2D(64, (3, 3), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.5))
    m.add(Conv2D(64, (3, 3), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Dropout(0.5))
    m.add(Flatten())
    m.add(Dense(64, activation="relu"))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m, 20, 16, [], 10000, 0.5

# Looks like this model isn't able to learn, probably because it is too simple.
# Let's take a step back and try increasing the sample size.
def m4():
    m = Sequential()
    m.add(BatchNormalization(input_shape=(512, 512, 1)))
    m.add(Conv2D(64, (3, 3), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Flatten())
    m.add(Dense(64, activation="relu"))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m, 10, 12, [], 10000, 0.5

# This works really well! However it does still overfit the data, but later than m2 does.
# This is promising since even with this larger sample size, it's stil just a fraction of the whole dataset.
# It also looks like it would benefit from continued training.
# So the general architecture works well; let's investigate changing the Conv2D hyperparameters a bit, to see if we can't push it further.
# Our images are very large (512 x 512), so maybe using 3 x 3 is too small.
def m5():
    m = Sequential()
    m.add(BatchNormalization(input_shape=(512, 512, 1)))
    m.add(Conv2D(64, (3, 3), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(64, (3, 3), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Flatten())
    m.add(Dense(64, activation="relu"))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m, 10, 16, [], 30000, 0.5

# This really doesn't perform at all. Maybe we can try going from a large window size to a smaller one. Maybe two smaller ones sequentially.
# The idea is maybe the model can capture some large regions of interest, then drill down to a lower level to capture specific details.
def m6():
    m = Sequential()
    m.add(BatchNormalization(input_shape=(512, 512, 1)))
    m.add(Conv2D(64, (6, 6), activation="relu"))
    m.add(MaxPooling2D(pool_size=(3, 3)))
    m.add(Conv2D(64, (6, 6), activation="relu"))
    m.add(MaxPooling2D(pool_size=(3, 3)))
    m.add(Flatten())
    m.add(Dense(64, activation="relu"))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m, 5, 16, [], 30000, 0.5

# This performs really well, but it's hard to say if it performs better or worse than m5.
# It looks like it may be less prone to overfitting, but the data may be unrepresentative. Perhaps we just need a larger sample size.
# We can compare with cross-validation.
def m7():
    m = Sequential()
    m.add(BatchNormalization(input_shape=(512, 512, 1)))
    m.add(Conv2D(64, (6, 6), activation="relu"))
    m.add(MaxPooling2D(pool_size=(3, 3)))
    m.add(Conv2D(64, (3, 3), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(64, (3, 3), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Flatten())
    m.add(Dense(64, activation="relu"))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m, 10, 16, [], 30000, 0.5

# Looks like adding a second dense layer works really well on the training data, but not so much on validation.
# We might need to add some normalization and increase the sample size.
def m8():
    m = Sequential()
    m.add(BatchNormalization(input_shape=(512, 512, 1)))
    m.add(Conv2D(64, (6, 6), activation="relu"))
    m.add(MaxPooling2D(pool_size=(3, 3)))
    m.add(Conv2D(64, (3, 3), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(64, (3, 3), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Flatten())
    m.add(Dense(64, activation="relu"))
    m.add(Dense(64, activation="relu"))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m, 10, 16, [], 30000, 0.5

# It doesn't seem to work that well. It trains really slow on the training data, which is fine, but validation loss doesn't really improve at all.
# Maybe we'll try without dropout, and only with an increased sample size.
def m9():
    m = Sequential()
    m.add(BatchNormalization(input_shape=(512, 512, 1)))
    m.add(Conv2D(64, (6, 6), activation="relu"))
    m.add(Dropout(0.1))
    m.add(MaxPooling2D(pool_size=(3, 3)))
    m.add(Conv2D(64, (3, 3), activation="relu"))
    m.add(Dropout(0.1))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(64, (3, 3), activation="relu"))
    m.add(Dropout(0.1))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Flatten())
    m.add(Dense(64, activation="relu"))
    m.add(Dropout(0.5))
    m.add(Dense(64, activation="relu"))
    m.add(Dropout(0.5))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m, 15, 16, [], 90000, 0.5

# This performs great! It looks like all we needed was to increase the sample size.
# Hopefully that means when we do the final training it will perform great.
def m10():
    m = Sequential()
    m.add(BatchNormalization(input_shape=(512, 512, 1)))
    m.add(Conv2D(64, (6, 6), activation="relu"))
    m.add(MaxPooling2D(pool_size=(3, 3)))
    m.add(Conv2D(64, (3, 3), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Conv2D(64, (3, 3), activation="relu"))
    m.add(MaxPooling2D(pool_size=(2, 2)))
    m.add(Flatten())
    m.add(Dense(64, activation="relu"))
    m.add(Dense(64, activation="relu"))
    m.add(Dense(1, activation="sigmoid"))
    m.compile(optimizer=Adam(learning_rate=1e-3), loss="binary_crossentropy", metrics=["accuracy"])
    return m, 5, 16, [], 90000, 0.5
