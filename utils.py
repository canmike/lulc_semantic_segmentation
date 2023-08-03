import numpy as np
import matplotlib.pyplot as plt
from osgeo import gdal
import torch

def get_mask(index: int, file_path='/content/data/C2Seg_AB/train'):  
  file_path_mask = file_path+'/label/'+str(index)+'.tiff'

  image_mask  = gdal.Open(file_path_mask)
  array_mask = image_mask.ReadAsArray()

  array_mask = np.asarray(array_mask)
  return array_mask

def get_hsi(index: int, file_path='/content/data/C2Seg_AB/train'):  
  file_path_hsi = file_path+'/hsi/'+str(index)+'.tiff'

  image_hsi = gdal.Open(file_path_hsi)
  array_hsi = image_hsi.ReadAsArray()

  array_hsi = np.asarray(array_hsi)
  return array_hsi


def reduce_band(index, n=20, show_reduced=False):
  """
  Reduces given multi-banded image to n-band image using Principal Component Analysis (PCA). 

    Args:
      index: An integer indicating the file name.
      n: An integer indicating number of bands of the output image.
      show_reduced: True/False indicating whether show output bands.
  """
  hsi = get_hsi(index)

  # Step 1: Flatten the hyperspectral image to have shape (242, 128*128)
  hsi_flat = hsi.reshape(242, -1)

  # Step 2: Calculate the mean of each band and center the data
  mean_vector = np.mean(hsi_flat, axis=1)
  hsi_centered = hsi_flat - mean_vector[:, np.newaxis]

  # Step 3: Calculate the covariance matrix
  covariance_matrix = np.cov(hsi_centered)

  # Step 4: Calculate the eigenvectors and eigenvalues of the covariance matrix
  eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)

  # Step 5: Sort the eigenvectors based on the descending order of eigenvalues
  sorted_indices = np.argsort(eigenvalues)[::-1]
  eigenvalues = eigenvalues[sorted_indices]
  eigenvectors = eigenvectors[:, sorted_indices]

  # Step 6: Choose the number of principal components to retain (e.g., retain top N components)
  N = 20
  num_components_to_retain = N  # Set the desired number of components

  # Step 7: Select the top N eigenvectors to form the transformation matrix
  transformation_matrix = eigenvectors[:, :num_components_to_retain]

  # Step 8: Project the data onto the lower-dimensional space
  reduced_hsi = np.dot(transformation_matrix.T, hsi_centered)

  # Step 9: Reshape the reduced_hsi back to its original shape
  reduced_hsi = reduced_hsi.T.reshape(128, 128, num_components_to_retain)

  # Now, 'reduced_hsi' contains the hyperspectral image with reduced dimensions.
  # It has shape (128, 128, num_components_to_retain), where num_components_to_retain is the number of components you chose to retain.

  # Your PCA code here (without the visualization part)

  if show_reduced:
    plt.figure(figsize=(15, 5))
    band_to_visualize = 0  # You can choose any band to visualize
    plt.subplot(1, 2, 1)
    plt.title(f'Original Band {band_to_visualize}')
    plt.axis("off")
    plt.imshow(hsi[band_to_visualize], cmap='gray')
    # Assuming you chose num_components_to_retain = 20 for visualization
    num_components_to_retain = 20

    # Plot the twenty reduced bands from PCA
    plt.figure(figsize=(12, 10))
    for i in range(num_components_to_retain):
        plt.subplot(4, 5, i + 1)
        plt.imshow(reduced_hsi[:, :, i], cmap='gray')
        plt.title(f'Reduced Band {i + 1}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

  return reduced_hsi

def get_img(index, file_path='/content/data/C2Seg_AB/train'):
  
  file_path_msi = file_path + '/msi/'+str(index)+'.tiff'
  file_path_sar = file_path + '/sar/'+str(index)+'.tiff'

  image_msi = gdal.Open(file_path_msi)
  array_msi = image_msi.ReadAsArray()

  image_sar = gdal.Open(file_path_sar)
  array_sar = image_sar.ReadAsArray()

  array_hsi = reduce_band(index, file_path)
  array_hsi = torch.tensor(array_hsi)
  array_hsi = torch.permute(array_hsi, (2, 0, 1))
  array_hsi = array_hsi.numpy()

  all_channels = []
  for msi in array_msi:
    all_channels.append(msi)

  for sar in array_sar:
    all_channels.append(sar)

  for hsi in array_hsi:
    all_channels.append(hsi)

  all_channels = np.asarray(all_channels)
  all_channels = torch.from_numpy(all_channels)
  all_channels = torch.unsqueeze(all_channels, 0)
  all_channels = all_channels.float()
  
  mask = get_mask(index, file_path)
  
  return (all_channels, mask)
