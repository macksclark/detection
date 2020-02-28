import os
import torch
import numpy as np
import vision_utils.transforms as T
from PIL import Image
import xml.etree.ElementTree as ET

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

def parse_annotation(annotation_filename):
    '''
    Parse the XML outputted from synthetic data generation to get the bounding
    box coordinates and object labels.
    :param annotation_filename: str, absolute path to the annotation file for
        a specific image
    :return: tuple of lists, ([bbox pixel coords], [labels])
    '''
    # Parse annotation's XML file
    tree = ET.parse(annotation_filename)
    root = tree.getroot().find('annotation')
    labels = []
    boxes = []
    for obj in root.findall('object'):
        bndbox = obj.find('bndbox')
        # Match the label with what ones we are expecting
        labels.append(root.find('name').text)

        # To match the coco dataset, we put bounding box annotations in
        # [xmin, ymin, width, height] format
        xmin = int(bndbox.find('xmin').text)
        ymin = int(bndbox.find('ymin').text)
        width = int(bndbox.find('width').text)
        height = int(bndbox.find('height').text)
        boxes.append([xmin, ymin, width, height])
    return boxes, labels

class SynthData(torch.utils.data.Dataset):
    def __init__(self, root, train):
        self.root = root
        self.train = train
        self.transforms = build_transform(self.train)

        # Load the images and the targets from the input file locations.
        self.images = list(sorted(os.listdir(os.path.join(
            root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(
            root, "annotations", "masks"))))
        self.bbs = list(sorted(os.listdir(os.path.join(
            root, "annotations", "bboxes"))))

    def __getitem__(self, idx):
        '''
        Load the image and the target.
        '''
        img_path = os.path.join(self.root, "images", self.images[idx])
        mask_path = os.path.join(
            self.root, "annotations", "masks", self.masks[idx])
        anno_path = os.path.join(
            self.root, "annotations", "bboxes", self.bbs[idx])
        
        img = Image.open(img_path).convert("RGB")
        mask = np.array(Image.open(mask_path))
        target = {}
        
        # Parse annotation's XML file, second argument would be label numbers
        # for when that is implemented.
        boxes, _ = parse_annotation(anno_path)

        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.ones((len(boxes),), dtype=torch.int64)
        target['area'] = boxes[:, 2] * boxes[:, 3]      # width * height
        target['mask'] = torch.as_tensor(mask, dtype=torch.uint8)

        # Apply the transforms to the image and target
        img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.images)
