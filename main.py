#########################################
# Mackenzie Clark 02/2020 
# 
# Fine tuning the torchvision model zoo faster rcnn models on a small
# synthetic dataset.
# Requires python 3.7, torchvision 0.4.2, pytorch 1.3.1
#########################################

import os
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

from vision_utils.engine import train_one_epoch, evaluate
from vision_utils import utils
from global_vars import *
from synthdata_loader import SynthData

torch.manual_seed(1)

def load_frcnn(object_specific=False):
    '''
    Loads faster rcnn with an RPN and resnet101 backbone network, pretrained on
    COCO detection dataset.

    param object_specific: bool, True if want 
    '''
    if object_specific:
        # Read the dataset object files to know how many outputs there should be
        # num_clases = read file
        with open(OBJ_LIST, 'r') as f:
            ids = f.read().splitlines()
        num_classes = len(ids)
    else:
        num_classes = 2
        
    # Load an instance segmentation model pre-trained on COCO
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # now get the number of input features for the mask classifier
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # and replace the mask predictor with a new one
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model

class FineTuner(object):

    def __init__(self):
        # Save memory for the training and testing data. 
        self.train_data = None
        self.test_data = None
        # Fill in the data & loaders with our data. 
        self.train_data_loader = None
        self.test_data_loader = None
        self.load_data()

        # Initialize model and training params.
        self.model = None
        self.train_epochs = TRAIN_EPOCHS

        # Check to see if CUDA available.
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
    
    def load_data(self):
        '''
        Loads the data that we'll fine tune with. Splitting into train
        and test datasets. 
        '''
        # Load all the of the data twice (different transforms for train and
        # test), but only use each image once, based on random indices.
        train_dataset = SynthData(SYNTHDATA_ROOT, train=True)
        test_dataset = SynthData(SYNTHDATA_ROOT, train=False)

        indices = torch.randperm(len(train_dataset)).tolist()

        self.train_data = torch.utils.data.Subset(
            train_dataset, indices[:-TESTING_PARTITION])
        self.test_data = torch.utils.data.Subset(
            test_dataset, indices[-TESTING_PARTITION:])

        # Define the training and testing data loaders.
        self.train_data_loader = torch.utils.data.DataLoader(
            self.train_data, batch_size=2, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)

        self.test_data_loader = torch.utils.data.DataLoader(
            self.test_data, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=utils.collate_fn)

    def train(self):
        '''
        Fine tune the object detector on new data.
        '''
        # Get the model using our helper function
        self.model = load_frcnn()
        # Move model to device.
        self.model.to(self.device)

        # Build optimizer.
        # TODO: try out the Adam optimizer?
        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)

        # Learning rate scheduler which decreases the learning rate by
        # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        # Perform SGD for x epochs.
        for epoch in range(self.train_epochs):
            # Train for one epoch, printing every 10 iterations
            train_one_epoch(self.model, optimizer, self.train_data_loader,
                self.device, epoch, print_freq=10)
            # Update the learning rate
            lr_scheduler.step()
            # Evaluate on the test dataset
            self.evaluate()
    
    def evaluate(self):
        '''
        Evaluate the model on the testing portion of the data.
        '''
        # Pick one image from the test set
        img, _ = self.test_data[0]
        # Put the model in evaluation mode
        self.model.eval()
        with torch.no_grad():
            prediction = self.model([img.to(self.device)])

        print(prediction)
        # TODO: save prediction or display it?


if __name__ == "__main__":
    fine_tuner = FineTuner()
    finer_tuner.train()