import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import requests
from pathlib import Path

def download_py(file_name:str, path:str=None):
  if path == None:
    path = "https://raw.githubusercontent.com/canmike/lulc_semantic_segmentation/main/" + file_name

  if Path(file_name).is_file():
    print(f"{file_name} already exists, skipping download")
  else:
    # Note: you need the "raw" GitHub URL for this to work
    request = requests.get(path)
    with open(file_name, "wb") as f:
      f.write(request.content)
    print(f"Downloaded {file_name}.")

download_py("utils.py")
download_py("visualization.py")
download_py("training.py")
download_py("dataset.py")

data_count_ab = 273
data_count_bw = 7139

def get_parts(arr):
  part1 = arr[:, :128, :128]
  part2 = arr[:, :128, 128:]
  part3 = arr[:, 128:, :128]
  part4 = arr[:, 128:, 128:]

  return [part1, part2, part3, part4]

def get_parts_mask(mask):
  mpart1 = mask[:128, :128]
  mpart2 = mask[:128, 128:]
  mpart3 = mask[128:, :128]
  mpart4 = mask[128:, 128:]

  return [mpart1, mpart2, mpart3, mpart4]

from osgeo import gdal, gdal_array

def save_tiff(tiff, file_name):
  output_tiff_filename = f"{file_name}.tiff"

  # Specify the data type and format of the output TIFF file
  data_type = gdal_array.NumericTypeCodeToGDALTypeCode(tiff.dtype)
  driver = gdal.GetDriverByName("GTiff")

  # Create a new GDAL dataset
  num_bands, height, width = tiff.shape
  dataset = driver.Create(output_tiff_filename, width, height, num_bands, data_type)

  # Loop through each band and write the data
  for band_idx in range(num_bands):
      band = dataset.GetRasterBand(band_idx + 1)  # GDAL bands are 1-based
      band.WriteArray(tiff[band_idx])

  dataset = None


def save_tiff_mask(tiff, file_name):
  output_tiff_filename = f"{file_name}.tiff"

  # Specify the data type and format of the output TIFF file
  data_type = gdal_array.NumericTypeCodeToGDALTypeCode(tiff.dtype)
  driver = gdal.GetDriverByName("GTiff")

  # Create a new GDAL dataset
  height, width = tiff.shape
  dataset = driver.Create(output_tiff_filename, width, height, 1, data_type)

  # Loop through each band and write the data
  band = dataset.GetRasterBand(1)  # GDAL bands are 1-based
  band.WriteArray(tiff)

  dataset = None



if False:
  msi_path = "C:/Users/İTÜ/Desktop/ieee_whisper_segmentation/Patching/data/parts/msi"
  index = 0
  for i in range(data_count_bw+1):
    msi = gdal.Open(f"C:/Users/İTÜ/Desktop/ieee_whisper_segmentation/Patching/data/C2Seg_BW_train_msisar/msi/{i}.tiff")
    array_msi = msi.ReadAsArray()
    parts = get_parts(array_msi)
    for part in parts:
      save_tiff(part, f"{msi_path}/{index}")
      index += 1
      print(f"Divided MSI {i}/{data_count_bw} ({((i / data_count_bw) * 100):.2f}%) into 128x128.")
  msi_path = "C:/Users/İTÜ/Desktop/ieee_whisper_segmentation/Patching/data/parts/msi"

if False:
  sar_path = "C:/Users/İTÜ/Desktop/ieee_whisper_segmentation/Patching/data/parts/sar"
  index = 0
  for i in range(data_count_bw+1):
    sar = gdal.Open(f"C:/Users/İTÜ/Desktop/ieee_whisper_segmentation/Patching/data/C2Seg_BW_train_msisar/sar/{i}.tiff")
    array_sar = sar.ReadAsArray()
    parts = get_parts(array_sar)
    for part in parts:
      save_tiff(part, f"{sar_path}/{index}")
      index += 1
      print(f"Divided SAR {i}/{data_count_bw} ({((i / data_count_bw) * 100):.2f}%) into 128x128.")

if True:
  mask_path = "C:/Users/İTÜ/Desktop/ieee_whisper_segmentation/Patching/data/parts/label"
  index = 0
  for i in range(data_count_bw+1):
    mask = gdal.Open(f"C:/Users/İTÜ/Desktop/ieee_whisper_segmentation/Patching/data/C2Seg_BW_train_msisar/label/{i}.tiff")
    array_mask = mask.ReadAsArray()
    parts = get_parts_mask(array_mask)
    for part in parts:
      save_tiff_mask(part, f"{mask_path}/{index}")
      index += 1
      print(f"Divided Mask {i}/{data_count_bw} ({((i / data_count_bw) * 100):.2f}%) into 128x128.")
