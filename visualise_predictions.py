import cv2
import numpy as np

from pathlib import Path

from load_raw_volume import load_raw_volume


def normalize_data(data: np.array) -> np.array:
    ''' Normalize data to values 0-255'''
    return np.array(255*data/max(data.flatten()), np.uint8)

def get_img_axis1(nb: int, matrix: np.array) -> np.array:
    '''Returns img from axis 1'''
    return np.array(matrix[nb], np.uint8)

def get_img_axis1(nb: int, matrix: np.array) -> np.array:
    '''Returns img from axis 2'''
    return np.array(matrix[:,nb,:], np.uint8)

def get_img_axis3(nb: int, matrix: np.array) -> np.array:
    '''Returns img from axis 3'''
    return np.array(matrix[:,:,nb], np.uint8)


def main():
    data_path = Path('./dataset/SecondDataset/test/00e3519eae7d40246b89c4dec1f3c5f0/T1w.nii.gz')
    mask_path = Path('./predictions_allax/00e3519eae7d40246b89c4dec1f3c5f0.nii.gz')
    data, _ = load_raw_volume(data_path)
    data = normalize_data(data)
    mask, _ = load_raw_volume(mask_path)

    sample_nb = [70, 120, 150]
    for i in sample_nb:
        img1 = get_img_axis1(i, data)
        img2 = get_img_axis1(i, mask*255.0)

        z = np.zeros((img2.shape[0], img2.shape[1]), np.uint8)
        img1 = cv2.merge([img1, img1, img1])
        img2 = cv2.merge([z, img2, z])
        result = cv2.addWeighted(img1, 0.7, img2, 0.3, 0.5)

        if i == sample_nb[0]:
            out = result
        else:
            out = np.hstack([out, result])

    # Show img
    key = ord('a')
    while key != ord('q'):
        cv2.imshow('d', out)
        key = cv2.waitKey(3)


if __name__ == '__main__':
    main()
