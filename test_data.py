import torch
import torch.nn as nn
from dataset import Dataset, Preprocessing
from torchvision import transforms, utils
import numpy as np
import cv2
import os
import nibabel as nib
from typing import Tuple, List
from pathlib import Path


def load_raw_volume(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    ''' Load medical *.nii files '''
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
    
    
def crop(img, size):
    ''' Crop img to original volume size '''
    x_padding = int((288-size[0])/2)
    y_padding = int((288-size[1])/2)
    new_img = img[x_padding:x_padding+size[0], y_padding:y_padding+size[1]]
    print(new_img.shape)
    print(size)
    return new_img
    

def save_labels(data: np.ndarray, affine: np.ndarray, path: Path):
    ''' Save masks '''
  nib.save(nib.Nifti1Image(data, affine), str(path))
    
    
def return_affine(kod):
    ''' Read affine from originl data '''
    path_1 = './dataset/FirstDataset/test/'
    affine = None
    for f in os.listdir(path_1):
        if f == kod+'.nii.gz':
            path = Path(path_1 + f)
            data, affine = load_raw_volume(path)
            break
    if type(affine) == type(None):
        path_2 = './dataset/SecondDataset/test/'
        for f in os.listdir(path_2):
            if f == kod:
                path = Path(path_2 + f + '/T1w.nii.gz')
                data, affine = load_raw_volume(path)
                break
    return affine
    
def train_device():
    ''' Function for using cuda if is available
    returns used device: cuda or cpu
    '''
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    return device
            

def main():
    device = train_device()
    
    # Test dataset
    test_params = {'batch_size': 5,
              'shuffle': False,
              'num_workers': 4}
              
    data_path = './dataset/test/'
    kod = [f for f in os.listdir(data_path)]
    samples = [data_path+f+'/ax1/' for f in os.listdir(data_path)]
    info_file = [data_path+f+'/info.txt' for f in os.listdir(data_path)]
    
    # Load model
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)
    model.encoder1.enc1conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=(1,1), padding= (1,1), bias=False)
    model = torch.load('./model_ax1_titan/best_model.pt', map_location=device)
    model.to(device)
    model.eval()
    j = 0
    # Testing
    for sample in samples:
        with open(info_file[j], 'r') as f:
            info = f.read()
            info = info.replace(",", "")
            img_shape = list(map(int,(info.split()[-3:])))  
        test_dataset = Dataset(sample, is_test=True,
                transform=transforms.Compose([
                                    Preprocessing()]))
        test_loader = torch.utils.data.DataLoader(test_dataset, **test_params)
        result = np.zeros(tuple(img_shape))
        affine = return_affine(kod[j])
        k = 0
        for i, (x_test) in enumerate(test_loader):
            torch.cuda.empty_cache()
            x_test = x_test.to(device)
            outputs = model(x_test)  
            l = 0    
            for out in outputs:     
                key = ord('a')
                img = np.array(np.squeeze((out.cpu().detach().numpy())), np.uint8)
                img2 = np.array(np.squeeze(255*(x_test[l].cpu().detach().numpy())), np.uint8)
                img = crop(img, (img_shape[0], img_shape[1]))
                img2 = crop(img2, (img_shape[0], img_shape[1]))
                print(max(img.flatten()))
                img = np.flip(img, axis=0)
                img = np.array(np.where(img < 0.5, 0, 1), np.float32)
                print(max(img.flatten()))
                result[:,:, k] = img
                
                l += 1
                k += 1
        save_labels(result, affine, Path('./predictions2_ax1/'+kod[j]+'.nii.gz'))
        
        j+= 1   
    
    
if __name__ == '__main__':
    main()
