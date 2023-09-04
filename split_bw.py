import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

data_count = 28559


indexes = list(range(data_count+1))
train_idx, val_test_idx = train_test_split(indexes, train_size=0.70)

val_idx, test_idx = train_test_split(val_test_idx, train_size=0.50)


print(len(indexes))
print(len(train_idx))
print(len(val_idx))
print(len(test_idx))

def copy_splitted(data_type, index, new_index):
    source = f"C:\\Users\\İTÜ\\Desktop\\ieee_whisper_segmentation\\Patching\\data\\C2Seg_BW_patched\\msi\\{index}.tiff"
    dest = f"C:\\Users\\İTÜ\\Desktop\\ieee_whisper_segmentation\\Patching\\data\\C2Seg_BW_splitted\\{data_type}\\msi\\{new_index}.tiff"
    shutil.copyfile(source, dest)
    
    source = f"C:\\Users\\İTÜ\\Desktop\\ieee_whisper_segmentation\\Patching\\data\\C2Seg_BW_patched\\sar\\{index}.tiff"
    dest = f"C:\\Users\\İTÜ\\Desktop\\ieee_whisper_segmentation\\Patching\\data\\C2Seg_BW_splitted\\{data_type}\\sar\\{new_index}.tiff"
    shutil.copyfile(source, dest)
    
    source = f"C:\\Users\\İTÜ\\Desktop\\ieee_whisper_segmentation\\Patching\\data\\C2Seg_BW_patched\\label\\{index}.tiff"
    dest = f"C:\\Users\\İTÜ\\Desktop\\ieee_whisper_segmentation\\Patching\\data\\C2Seg_BW_splitted\\{data_type}\\label\\{new_index}.tiff"
    shutil.copyfile(source, dest)

new_index = 0
for index in train_idx:
    copy_splitted("train", index, new_index)    
    print(f"Training: Copied {index} to {new_index} ({(((new_index+1)/len(train_idx))*100):.2f}%)")
    new_index += 1
    
new_index = 0
for index in val_idx:
    copy_splitted("val", index, new_index)    
    print(f"Validation: Copied {index} to {new_index} ({(((new_index+1)/len(val_idx))*100):.2f}%)")
    new_index += 1
    
new_index = 0
for index in test_idx:
    copy_splitted("test", index, new_index)    
    print(f"Test: Copied {index} to {new_index} ({(((new_index+1)/len(test_idx))*100):.2f}%)")
    new_index += 1
    

