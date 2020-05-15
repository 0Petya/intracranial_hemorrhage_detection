# This is used to extract the pixel arrays from the dicom files, as well as saving the labels in an array.

import os
import multiprocessing
import numpy as np
import pandas as pd
import pydicom

from p_tqdm import p_imap

cores = multiprocessing.cpu_count()

if not os.path.exists("../../data/processed/images/"):
    os.makedirs("../../data/processed/images/")

def process_image(row):
    try:
        dcm = pydicom.dcmread(f"../../data/raw/rsna-intracranial-hemorrhage-detection/stage_2_train/{row[1]}.dcm")
        return row[1], dcm.pixel_array, row[2:]
    except:
        return None, None, None

# Transforming the manifest such that each row is one image and each column is a label.
manifest = pd.read_csv("../../data/interim/manifest.csv")
split_id = manifest["ID"].str.rsplit('_', n=1, expand=True)
manifest["ID"] = split_id[0]
manifest["Subtype"] = split_id[1]
manifest = manifest.pivot(index="ID", columns="Subtype", values="Label").reset_index().rename_axis(None, axis=1)

processed_images = p_imap(process_image, list(manifest.itertuples()), num_cpus=cores)
ids = []
labels = []
for id, pixels, labels_ in processed_images:
    # Rarely pydicom fails to read an image, discard it if that's the case.
    if id is None and pixels is None and labels_ is None: continue
    # Rarely an image is of a different size than the overwhelming majority, discard it if that's the case.
    if pixels.shape != (512, 512): continue

    np.save(f"../../data/processed/images/{id}.npy", pixels)
    ids.append(id)
    labels.append(labels_)

print(f"Processed {len(ids)} images successfully.")
np.save("../../data/processed/ids.npy", np.array(ids))
np.save("../../data/processed/Y.npy", np.array(labels))
