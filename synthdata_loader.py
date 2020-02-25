import torch
import vision_utils.transforms as T

# TODO: Add a cropping step to transforms to standardize network inputs ? 
def build_transform(train):
    '''
    Build the transform to be applied to each input image.
    '''
    transforms = []
    # Converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # During training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# TODO: incomplete class
class SynthData(torch.utils.data.Dataset):
    def __init__(self, root, train):
        self.root = root
        self.train = train
        self.transforms = build_transform(self.train)

        # Load the images and the targets from the input file locations.
        self.images = 
        self.targets = 

    def __getitem__(self):

    def __len__(self):
        return len(self.images)
