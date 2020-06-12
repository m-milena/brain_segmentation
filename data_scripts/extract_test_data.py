import numpy as np
import nibabel as nib
import os
import cv2
from typing import Tuple, List
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

max_size = 288

def show_img(img):
    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(img, cmap="gray", origin="lower")
    plt.show()

def load_raw_volume(path: Path) -> Tuple[np.ndarray, np.ndarray]:
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

def padd_img(img):
    out_size = (288,288)
    out_img = np.zeros(out_size)
    x_pad = int((288-img.shape[0])/2)
    y_pad = int((288-img.shape[1])/2)

    out_img[x_pad:x_pad+img.shape[0], y_pad:y_pad+img.shape[1]] = img

    return out_img
        
def normalize_data(data):
    return np.array(255*data/max(data.flatten()), np.uint8)

def save_img(img, path, is_mask = False):
    if is_mask:
        img = np.array(img*255, np.uint8)
    img_flipped = np.flip(img, axis=0)
    img_to_save = padd_img(img_flipped)
    cv2.imwrite(path, img_to_save)
    return np.array(img_to_save, np.uint8)


def main():
    first_dataset_path = Path('./FirstDataset/test')
    second_dataset_path = Path('./SecondDataset/test')

    for scan_path in first_dataset_path.iterdir():
        if not scan_path.name.endswith('mask.nii.gz'):
            print('Started %s'%str(scan_path)[19:-7])
            # Load data
            data, aff = load_raw_volume(str(scan_path))

            if type(data) == type(None):
                continue
            data_normalized = normalize_data(data)
            # Create data folder
            folder_path = './test/'+str(scan_path)[19:-7]
            os.mkdir(folder_path)
            print(data.shape)
            # Save axis1 data
            sample = 0
            os.mkdir(folder_path + '/ax1')
            os.mkdir(folder_path + '/ax1/images')

            for i in range(0, data.shape[0]):
                filename = '/sample_%05d.png'%sample
                if not np.all(data_normalized[i] == 0):
                    img = save_img(data_normalized[i], folder_path + '/ax1/images' + filename)
                sample += 1

            # Save axis2 data
            sample = 0
            os.mkdir(folder_path + '/ax2')
            os.mkdir(folder_path + '/ax2/images')
            for i in range(0, data.shape[1]):
                filename = '/sample_%05d.png'%sample
                if not np.all(data_normalized[:,i,:] == 0):
                    img = save_img(data_normalized[:,i,:], folder_path + '/ax2/images' + filename)
                sample += 1

            # Save axis3 data
            sample = 0
            os.mkdir(folder_path + '/ax3')
            os.mkdir(folder_path + '/ax3/images')
            for i in range(0, data.shape[1]):
                filename = '/sample_%05d.png'%sample
                if not np.all(data_normalized[:,i,:] == 0):
                    img = save_img(data_normalized[:,:,i], folder_path + '/ax3/images' + filename)
                sample += 1
            
            with open(folder_path +'/info.txt', 'w+') as f:
                f.write('Original shape: %d, %d, %d \n'%data.shape)

            print('Extracted %s' %str(scan_path)[19:-7])


    for scan_path in second_dataset_path.iterdir():
        # Load data
        print('Started %s'%str(scan_path)[20:])
        data, aff = load_raw_volume(str(scan_path/'T1w.nii.gz'))
        if type(data) == type(None):
            continue
        data_normalized = normalize_data(data)

        folder_path = './test/'+str(scan_path)[20:]
        os.mkdir(folder_path)

        # Save axis1 data
        sample = 0
        os.mkdir(folder_path + '/ax1')
        os.mkdir(folder_path + '/ax1/images')

        for i in range(0, data.shape[0]):
            filename = '/sample_%05d.png'%sample
            if not np.all(data_normalized[i] == 0):
                img = save_img(data_normalized[i], folder_path + '/ax1/images' + filename)
            sample += 1

        # Save axis2 data
        sample = 0
        os.mkdir(folder_path + '/ax2')
        os.mkdir(folder_path + '/ax2/images')
        for i in range(0, data.shape[1]):
            filename = '/sample_%05d.png'%sample
            if not np.all(data_normalized[:,i,:] == 0):
                img = save_img(data_normalized[:,i,:], folder_path + '/ax2/images' + filename)
            sample += 1

        # Save axis3 data
        sample = 0
        os.mkdir(folder_path + '/ax3')
        os.mkdir(folder_path + '/ax3/images')
        for i in range(0, data.shape[1]):
            filename = '/sample_%05d.png'%sample
            if not np.all(data_normalized[:,i,:] == 0):
                img = save_img(data_normalized[:,:,i], folder_path + '/ax3/images' + filename)
            sample += 1
            
        with open(folder_path +'/info.txt', 'w+') as f:
                f.write('Original shape: %d, %d, %d \n'%data.shape)

        print('Extracted %s' %str(scan_path)[20:])

if __name__ == '__main__':
    main()