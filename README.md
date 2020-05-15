# Intracranial Hemorrhage Detection

This is a project for detecting intracranial hemorrhages from head CT scans. We are using a dataset from [Kaggle](https://www.kaggle.com/c/rsna-intracranial-hemorrhage-detection).

## Setup

This project was developed with python 3.7.7. The project requirements are under `requirements.txt`.

To download the data, you'll need to accept the competition terms, then you'll need to have your Kaggle API token stored under `~/.kaggle/kaggle.json`. Then you can run the download script `src/data/download`.

## Description

A report of the project can be found under `./reports/report.pdf`.

`./notebooks/eda.ipynb` is used to perform an exploratory analysis of the data.

`./src/data/remove_duplicates_from_manifest.py` is used to remove duplicate IDs found in the data. `./src/features/extract_pixel_arrays_and_labels.py` is used to open the DICOM files and preprocess them and save them (as well as respective labels) as numpy arrays for quick loading. `./split_any_hemo_data.py` is then used to split the data into an 80/20 train/test set, for a holdout test set.

`./src/data/data_generator.py` is used to define a generator to allow data to be loaded as needed by models, since it is too large for holding it all in memory. `./src/models/candidates.py` is where all models are stored and defined. `./src/models/quick_evaluation.py` is used to quickly evaluate individual models to determine which are worthy candidates and how to develop new models. `./notebooks/cross_validation.ipynb` is used to cross-validate the best model under statistical scrutiny. `./src/models/train_best.py` will then train the best model with a large sample size for final evaluation. `./notebooks/evaluation.ipynb` is then used to determine the best specificity-sensitivity threshold to maximize the $F_2$ score and evaluate the model on the holdout test set.
