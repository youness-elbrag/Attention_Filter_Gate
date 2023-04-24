
from tqdm.notebook import tqdm
import numpy as np 
import nibabel as nib
from pathlib import Path
import os
from utils import *
import argparse

# create argument parser to accept command line arguments
parser = argparse.ArgumentParser(description='Process image data')
parser.add_argument('--root_img', required=True, help='Root directory of input images')
parser.add_argument('--root_lab', required=True, help='Root directory of input labels')
args = parser.parse_args()

## Create the function to loop over the sample data
length_sample = 19
root_saved = Path("./Processed")
root_img = Path(args.root_img)
root_lab = Path(args.root_lab)

for counter, samples in enumerate(tqdm(range(length_sample))):
    ## return the path (image and label)
    image_path, label_path = image_to_label(samples, root_img, root_lab)
    ## loading sample 
    image = nib.load(image_path)
    label = nib.load(label_path)
    ## assert the Voxel Corrd 
    assert nib.aff2axcodes(image.affine) == ('R', 'A', 'S')
    ## get the array values
    image_array = image.get_fdata()
    label_array = label.get_fdata().astype(np.uint8)
    ## Crop region through 2D shape 
    crop_2d_img = image_array[32:-32, 32:-32]
    crop_2d_lab = label_array[32:-32, 32:-32]
    ## normalize and Standardize Image only 
    normalized = Normalization(crop_2d_img)
    std_mri = Standardization(normalized)
    ## split data into Train/Val
    if counter < 17:
        current = root_saved / "train" / str(counter)
    else:
        current = root_saved / "val" / str(counter)
    ## extract the 2D slices from image MRI
    for Slice in range(std_mri.shape[-1]):
        slices_image = std_mri[:, :, Slice]
        slices_label = crop_2d_lab[:, :, Slice]
        slice_path = current / "data"
        slice_label = current / "mask"
        ## check Folder exist 
        slice_path.mkdir(parents=True, exist_ok=True)
        slice_label.mkdir(parents=True, exist_ok=True)
        ## save the Slices 
        np.save(slice_path / str(Slice), slices_image)
        np.save(slice_label / str(Slice), slices_label)
