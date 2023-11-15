from chardet.universaldetector import UniversalDetector
import numpy as np
import imageio
from tqdm.auto import tqdm
import pandas as pd
import main

"""
Credit to Jaroslav Knotek for the following functions:
- read_data
- _maybe_img
- _read_record
- _read_indent_centers
"""

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


def _read_indent_centers(filename):
    # HACK: Input data have different text encoding. The goal is to reliably turn decimal numbers from Czech notation "1,2 " to English notation 1.2 for further parsing.
    detector = UniversalDetector()
    detector.reset()
    for line in open(filename, 'rb'):
        detector.feed(line)
        if detector.done:
            break
    detector.close()

    with open(filename, encoding=detector.result['encoding']) as f:
        lines_raw = f.readlines()
        lines = [line.replace(',', '.').strip().split('\t')
                 for line in lines_raw]

    cnts = [len(x) for x in lines]

    d = {}
    for c in cnts:
        d[c] = d.get(c, 0) + 1

    idxs = [i for i, x in enumerate(lines) if len(x) == 1]
    indent_centers = np.array(lines[idxs[2]+1:idxs[3]]).astype(float)

    expected = 282
    # {np.array(lines_raw)[idxs]}
    assert len(
        indent_centers) == expected, f"Invalid number of centers. {len(indent_centers)}!= {expected}. {d}, {filename}"
    return indent_centers
