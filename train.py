#import segmentation_models_pytorch as smp
import torch

def main():
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=2, out_channels=1, init_features=32, pretrained=True)

if __name__ == '__main__':
    main()
