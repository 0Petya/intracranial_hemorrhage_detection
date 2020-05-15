# The provided manifest file has some duplicate IDs. This will remove them.

import os
import pandas as pd

if not os.path.exists("../../data/interim/"):
    os.makedirs("../../data/interim/")

manifest = pd.read_csv("../../data/raw/rsna-intracranial-hemorrhage-detection/stage_2_train.csv")
manifest = manifest.drop_duplicates(subset="ID")
manifest.to_csv("../../data/interim/manifest.csv", index=False)
