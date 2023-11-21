import imageio
from tqdm.auto import tqdm
import pandas as pd
from shared import log_error


def read_data(data_root):
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
        _read_record(dir_path)
        for dir_path in tqdm(list(data_root.glob('*')))
    ]
    cols = ['img_before', 'img_after']
    return pd.DataFrame(data, columns=cols)

def _maybe_img(img_path):
    try:
        img = imageio.imread(img_path)
        if len(img.shape) == 2:
            return img
        elif len(img.shape) == 3:
            return img[:, :, 0]
    except Exception as e:
        error_message = f"Unexpected image shape: {str(e)}"
        folder_name = "Image"
        log_error(folder_name, error_message)
        raise Exception("Unexpected image shape")


def _read_record(dir_path):
    img_before_path = dir_path / f"{dir_path.stem}.jpg"

    img_after_candidates = list(dir_path.glob(f"{dir_path.stem}_*.jpg"))

    if not img_before_path.exists() or not img_after_candidates:
        error_message = "One or both images are missing ."
        folder_name = "Image"
        log_error(folder_name, error_message)
        raise Exception(f"One or both images are missing in folder {dir_path.name}.")
      
    img_after_path = img_after_candidates[0]

    img_before = _maybe_img(img_before_path)
    img_after = _maybe_img(img_after_path)

    return img_before, img_after