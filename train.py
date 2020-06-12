import torch
import torch.nn as nn
from torchvision import transforms, utils

import numpy as np

from dataset import Dataset, Preprocessing

import loggs
from loss import DiceLoss
from average_meter import AverageMeter

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
    # CUDA for PyTorch
    device = train_device()

    # Training dataset
    train_params = {'batch_size': 10,
              'shuffle': True,
              'num_workers': 4}

    data_path = './dataset/dataset_ax1/train/'
    train_dataset = Dataset(data_path,
                transform=transforms.Compose([
                                    Preprocessing()]))
    lenght = int(len(train_dataset))
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_params)
    
    # Validation dataset
    data_path = './dataset/dataset_ax1/valid/'
    valid_dataset = Dataset(data_path,
                transform=transforms.Compose([
                                    Preprocessing()]))
    valid_params = {'batch_size': 10,
              'shuffle': True,
              'num_workers': 4}
    val_loader = torch.utils.data.DataLoader(valid_dataset, **valid_params)
    
    # Training params
    learning_rate = 1e-4
    max_epochs = 1000

    # Used pretrained model and modify channels from 3 to 1
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)
    model.encoder1.enc1conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=(1,1), padding= (1,1), bias=False)
    model.to(device)
    
    # Optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    dsc_loss = DiceLoss()
    
    # Metrics
    train_loss = AverageMeter('Training loss', ':.6f')
    val_loss = AverageMeter('Validation loss', ':.6f')
    best_loss = float('inf')
    nb_of_batches = lenght//train_params['batch_size']
    
    for epoch in range(max_epochs):
        val_loss.avg = 0
        train_loss.avg = 0
        if not epoch:
            logg_file = loggs.Loggs(['epoch', 'train_loss', 'val_loss'])
            model.train()
        for i, (image, label) in enumerate(train_loader):
            torch.cuda.empty_cache()
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            y_pred = model(image)
            loss = dsc_loss(y_pred, label)
            del y_pred
            train_loss.update(loss.item(), image.size(0))
            loss.backward()
            optimizer.step()
            loggs.training_bar(i, nb_of_batches, prefix='Epoch: %d/%d'%(epoch,max_epochs), suffix='Loss: %.6f'%loss.item())
        print(train_loss.avg)
        
        with torch.no_grad():
            for i, (x_val, y_val) in enumerate(val_loader):
                x_val, y_val = x_val.to(device), y_val.to(device)
                model.eval()
                yhat = model(x_val)
                loss = dsc_loss(yhat, y_val)
                val_loss.update(loss.item(), x_val.size(0))
            print(val_loss)
            logg_file.save([epoch, train_loss.avg, val_loss.avg])

            # Save the best model with minimum validation loss
            if best_loss > val_loss.avg:
                print('Updated model with validation loss %.6f ---> %.6f' %(best_loss, val_loss.avg))
                best_loss = val_loss.avg
                torch.save(model, './model_ax1/best_model.pt')
           


if __name__ == '__main__':
    main()
