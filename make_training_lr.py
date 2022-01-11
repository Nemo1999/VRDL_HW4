from PIL import Image
import os 
import numpy as np
import shutil
import tqdm

hr_path = 'data/training_hr'
lr_path = 'data/training_lr'

# remove the old folder if exists
if os.path.exists(lr_path):
    shutil.rmtree(lr_path)
    os.mkdir(lr_path)

# read each image in the folder and downsize it with 3x scale factor
# save the results in another folder
print("resizing training images...")
for filename in tqdm.tqdm(os.listdir(hr_path)):
    if filename.endswith('.png'):
        im_path = os.path.join(hr_path, filename)
        im = Image.open(im_path)
        im_resized = im.resize(size=(int(im.size[0]/3), int(im.size[1]/3)), resample=Image.BICUBIC)
        im_resized.save(os.path.join(lr_path, filename))