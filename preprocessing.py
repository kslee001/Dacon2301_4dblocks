# import torch
from PIL import Image
import pandas as pd
import numpy as np
import os
# from dm.auto import dm as 
import cv2
import torch
from rembg import remove, new_session

import warnings

train = pd.read_csv('/home/gyuseonglee/workspace/play/data/train.csv')
train['img_path'] = train['img_path'].str.replace(pat='./train', repl='/home/gyuseonglee/workspace/play/data/train', regex=None)
test = pd.read_csv('/home/gyuseonglee/workspace/play/data/test.csv')
test['img_path'] = test['img_path'].str.replace('./test', '/home/gyuseonglee/workspace/play/data/test')

train_imgs = train.img_path.tolist()
test_imgs = test.img_path.tolist()



batch_size = 96


from carvekit.api.high import HiInterface

# Check doc strings for more information
interface = HiInterface(object_type="object",  # Can be "object" or "hairs-like".
                        batch_size_seg=batch_size,
                        device = 'cuda',
                        seg_mask_size=640,  # Use 640 for Tracer B7 and 320 for U2Net
                        fp16=True)
interface.preprocessing_pipeline = None
interface.postprocessing_pipeline = None



def make_dir(directory:str):
    if not os.path.exists(directory):
        os.makedirs(directory)
        
def remove_transparrent(img):
    pixel_data = np.array(img)
    pixel_data[pixel_data==130] = 0
    return Image.fromarray(pixel_data).convert("RGB")


train_ids = train.id.tolist()
train_num_iters = len(train)//batch_size + 1
output_folder = "/home/gyuseonglee/workspace/play/data/train_bg"
make_dir(output_folder)
num_img = 0

for it in (range(train_num_iters)):
    train_samples = train_imgs[(it*batch_size):((it+1)*batch_size)]
    interface.segmentation_pipeline.to('cuda')
    converted = interface(train_samples)
    interface.segmentation_pipeline.to('cpu')
    torch.cuda.empty_cache()

    
    for idx in (range(len(converted))):
        file_name = f"{output_folder}/BG_{train_ids[num_img]}.jpg"
        cur_img = remove_transparrent(converted[idx])
        cur_img.save(file_name)
        num_img += 1
        
    del converted
    
    

test_ids = test.id.tolist()
test_num_iters = len(test)//batch_size + 1
output_folder = "/home/gyuseonglee/workspace/play/data/test_bg"
make_dir(output_folder)
num_img = 0

for it in (range(test_num_iters)):
    test_samples = test_imgs[(it*batch_size):((it+1)*batch_size)]
    interface.segmentation_pipeline.to('cuda')
    converted = interface(test_samples)
    interface.segmentation_pipeline.to('cpu')
    torch.cuda.empty_cache()

    
    for idx in (range(len(converted))):
        file_name = f"{output_folder}/BG_{test_ids[num_img]}.jpg"
        cur_img = remove_transparrent(converted[idx])
        cur_img.save(file_name)
        num_img += 1
        
    del converted
    
    
