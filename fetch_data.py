import os
import pandas as pd
from pathlib import Path
import numpy as np

DATASET_SIZE = 100
DATA_OUT = Path("/data/temporary/ivan/DeepDerma/BCC_SCLAM/cobra-streamingclam/data_source")

def symlink_force(src, dest):
# https://stackoverflow.com/questions/8299386/modifying-a-symlink-in-python
    try:
        os.symlink(src, dest)
    except FileExistsError as e:
        os.remove(dest)
        os.symlink(src, dest)


def create_symlinks(label_csv,split="train"):
    label_csv["slide"] = list(map(lambda x: f'{split}_' + x, map(str,np.arange(len(label_csv)))))
    label_csv["data_source_folder"] = label_csv["tumor"].map( lambda x: source_folder[x])

    for (slide,uuid,data_source_folder) in label_csv[["slide","uuid","data_source_folder"]].itertuples(index=False):
        image_source_path = data_source_folder / f"images/{uuid}.tiff"
        mask_source_path = data_source_folder / f"tissue_masks/{uuid}.tiff"
        symlink_force(image_source_path,DATA_OUT / f"images/{slide}.tiff")
        symlink_force(mask_source_path,DATA_OUT / f"tissue_masks/{slide}.tiff")
        
    label_csv = label_csv[["slide","tumor"]]
    label_csv.to_csv(DATA_OUT / f"{split}.csv",index=False)


if __name__ == "__main__":
    if not os.path.exists(DATA_OUT / "images"):
        os.makedirs(DATA_OUT / "images")

    if not os.path.exists(DATA_OUT / "tissue_masks"):
        os.makedirs(DATA_OUT / "tissue_masks")

    image_folder_bcc = Path("/data/pa_cpgarchive/archives/skin/cobra/data/bcc_risk/data/bcc")
    image_folder_nonmalignant = Path("/data/pa_cpgarchive/archives/skin/cobra/data/bcc_risk/data/non-malignant")
    label_folder = Path("/data/pa_cpgarchive/archives/skin/cobra/folds/bcc_risk")
    source_folder = {0: image_folder_nonmalignant,
                        1: image_folder_bcc
                    }

    for split in ("train","test","val"):
        label_path = label_folder / "train.csv"
    # test_label_path = label_folder / "test.csv"
    # val_label_path = label_folder / "test.csv"
        labels = pd.read_csv(label_path)[:DATASET_SIZE]
    # train_labels = pd.read_csv(train_label_path)[:DATASET_SIZE]
    # test_labels = pd.read_csv(test_label_path)[:DATASET_SIZE]

        create_symlinks(labels,split=split)
        # create_symlinks(test_labels,split="test")

