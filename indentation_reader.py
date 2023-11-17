from chardet.universaldetector import UniversalDetector
import numpy as np
import imageio
from tqdm.auto import tqdm
import pandas as pd
import main


def read_data(data_root,out_folder):
    """
    The method reads data in the following structure

    dir1
    img.jpg
    img_*.jpg for example img_after.jpg
    dir2
    ...

    returns:
    img_before - (img.jpg)
    img_after - (img_after.jpg)
    """
    data = [
        _read_record(dir_path,out_folder)
        for dir_path in tqdm(list(data_root.glob('*')))
    ]
    cols = ['img_before', 'img_after']
    return pd.DataFrame(data, columns=cols)


def _maybe_img(img_path,out_folder):
    
    if not img_path.exists():
        error_message = "Image is not in good format."
        folder_name = "Image"
        main.log_error(folder_name, error_message, out_folder)
    img = imageio.imread(img_path)
    if len(img.shape) == 2:
        return img
    elif len(img.shape) == 3:
        return img[:, :, 0]
    else:
        error_message = "Unexpected image shape"
        folder_name = "Image"
        main.log_error(folder_name, error_message, out_folder)
        return None



def _read_record(dir_path,out_folder):
    img_before_path = dir_path / f"{dir_path.stem}.jpg"

    img_after_candidates = list(dir_path.glob(f"{dir_path.stem}_*.jpg"))

    if not img_after_candidates:
        error_message = "No matching img."
        folder_name = "Image"
        main.log_error(folder_name, error_message, out_folder)
      
    img_after_path = img_after_candidates[0]

    img_before = _maybe_img(img_before_path, out_folder)
    img_after = _maybe_img(img_after_path, out_folder)

    return img_before, img_after
