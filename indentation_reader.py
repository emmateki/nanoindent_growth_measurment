import imageio
from tqdm.auto import tqdm
import pandas as pd
import pathlib

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
    folder_name - (name of the folder)
    """
    data = []
    for dir_path in tqdm(list(data_root.glob('*'))):
        img_before, img_after = _read_record(dir_path)
        folder_name = dir_path.name 
        data.append([img_before, img_after, folder_name])

    cols = ['img_before', 'img_after', 'folder_name']
    return pd.DataFrame(data, columns=cols)


def _maybe_img(img_path):
    try:
        img = imageio.imread(img_path)
        if len(img.shape) == 2:
            return img
        elif len(img.shape) == 3:
            return img[:, :, 0]
    except Exception as e:
        raise Exception("Unexpected image shape in folder {dir_path.name}.")


def _read_record(dir_path):
    img_before_path = dir_path / f"{dir_path.stem}.jpg"

    img_after_candidates = list(dir_path.glob(f"{dir_path.stem}_*.jpg"))

    if not img_before_path.exists() or not img_after_candidates:
        raise Exception(f"One or both images are missing in folder {dir_path.name}.")
      
    img_after_path = img_after_candidates[0]

    img_before = _maybe_img(img_before_path)
    img_after = _maybe_img(img_after_path)

    return img_before, img_after