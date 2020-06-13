import numpy as np
import nibabel as nib

import cv2
from typing import Tuple, List
from pathlib import Path


def load_raw_volume(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data: nib.Nifti1Image = nib.load(str(path))
    data = nib.as_closest_canonical(data)
    raw_data = data.get_fdata(caching='unchanged', dtype=np.float32)
    return raw_data, data.affine


def maximum_shape() -> int:
    first_dataset_path = Path('./FirstDataset/train')
    second_dataset_path = Path('./SecondDataset/train')

    max_size = 0
    for scan_path in first_dataset_path.iterdir():
        if not scan_path.name.endswith('mask.nii.gz'):
            data, aff = load_raw_volume(str(scan_path))
            max_this_data = max(data.shape)
            max_size = max(max_size, max_this_data)

    for scan_path in second_dataset_path.iterdir():
        data, aff = load_raw_volume(str(scan_path/'T1w.nii.gz'))
        max_this_data = max(data.shape)
        max_size = max(max_size, max_this_data)

    return max_size

if __name__ == '__main__':
    maximum_shape()
