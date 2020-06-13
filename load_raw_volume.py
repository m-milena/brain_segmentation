import numpy as np
import nibabel as nib

from pathlib import Path
from typing import Tuple, List


def load_raw_volume(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ''' Loading raw volume from *.nii medical files '''
    data: nib.Nifti1Image = nib.load(str(path))
    data = nib.as_closest_canonical(data)
    try:
        raw_data = data.get_fdata(caching='unchanged', dtype=np.float32)
    except Exception as E:
        raw_data = None
        with open('issues.txt', 'a') as f:
            f.write(str(path) + '\n')
            f.write(str(E) + '\n')
    return raw_data, data.affine
