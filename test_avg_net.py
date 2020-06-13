import os
from pathlib import Path
from typing import Tuple, List

import nibabel as nib


def load_raw_volume(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ''' Load raw medical *.nii files '''
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

def save_labels(data: np.ndarray, affine: np.ndarray, path: Path):
    ''' Save predictions '''
    nib.save(nib.Nifti1Image(data, affine), str(path))
    

def main():
    path1 = './predictions_ax1/'
    path2 = './predictions_ax2/'
    path3 = './predictions_ax3/'
    
    predictions = [f for f in os.listdir(path1) if f[-3:] == '.gz']
    for pr in predictions:
        result1, aff1 = load_raw_volume(Path(path1+pr))
        result2, aff2 = load_raw_volume(Path(path2+pr))
        result3, aff3 = load_raw_volume(Path(path3+pr))
        
        result = (result1 + result2 + result3)/3
        result = np.array(np.where(result < 0.5, 0, 1), np.float32)
        
        save_labels(result, aff1, Path('./predictions_allax/'+pr))
        

    

if __name__ == '__main__':
    main()
