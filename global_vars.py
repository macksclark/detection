import os

# Global constants for file paths. Set to match the local computer setup.
IMAGE_EXT = '.png'
MASK_EXT = '_mask.pbm'

# These paths are incorrect right now. Will fix in a hot second. 
SYNTHDATA_ROOT = '/mnt/c/Users/Mackenzie/Documents/github/synth-data-gen/'
OBJ_LIST = os.path.join(SYNTHDATA_ROOT, 'images', 'selected.txt')

# Model training/fine tuning params
TESTING_PARTITION = 50         # number of images in testing set
TRAIN_EPOCHS = 10