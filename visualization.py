

import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random

device = "cuda" if torch.cuda.is_available() else "cpu"

import numpy as np
label_map = np.array([
      (0, 0, 0),          # 0 - Background (Black)
      (0, 0, 255),        # 1 - Surface water (Blue)
      (135, 206, 250),    # 2 - Street (Light Sky Blue)
      (255, 255, 0),      # 3 - Urban Fabric (Yellow)
      (128, 0, 0),        # 4 - Industrial, commercial and transport (Maroon)
      (139, 37, 0),       # 5 - Mine, dump, and construction sites (Reddish Brown)
      (0, 128, 0),        # 6 - Artificial, vegetated areas (Green)
      (255, 165, 0),      # 7 - Arable Land (Orange)
      (0, 255, 0),        # 8 - Permanent Crops (Lime Green)
      (154, 205, 50),     # 9 - Pastures (Yellow Green)
      (34, 139, 34),      # 10 - Forests (Forest Green)
      (139, 69, 19),      # 11 - Shrub (Saddle Brown)
      (245, 245, 220),    # 12 - Open spaces with no vegetation (Beige)
      (0, 255, 255),      # 13 - Inland wetlands (Cyan)
  ])


labels = [
    "Background", "Surface water", "Street", "Urban Fabric", "Industrial, commercial and transport",
    "Mine, dump, and construction sites", "Artificial, vegetated areas", "Arable Land",
    "Permanent Crops", "Pastures", "Forests", "Shrub", "Open spaces with no vegetation", "Inland wetlands"
]

def predict_and_show(model, dataset, index):
  X, mask = dataset[index]
  X = X.to(device)
  model.eval()
  with torch.inference_mode():
    output = model(X.unsqueeze(dim=0)).to(device)

  W, H = output.shape[2], output.shape[3]

  # Get RGB segmentation map
  segmented_image = draw_segmentation_map(output)

  # Resize to original image size
  segmented_image = cv2.resize(segmented_image, (W, H), cv2.INTER_LINEAR)

  # Plot
  plt.figure(figsize=(20, 20))

  plt.subplot(1, 3, 1)
  # Create a custom colormap using the colors defined above
  cmap = ListedColormap(label_map / 255.0)

  # Display the mask using the custom colormap
  plt.imshow(mask, cmap=cmap, vmin=0, vmax=13)

  plt.title("Ground Truth")
  plt.axis("off")

  plt.subplot(1, 3, 2)
  plt.title("Segmentation")
  plt.axis("off")
  plt.imshow(segmented_image)

  plt.show()
  plt.close()
  # Save Segmented and overlayed images
  if False:
      cv2.imwrite(seg_map_save_dir, segmented_image[:, :, ::-1])
      cv2.imwrite(overlayed_save_dir, overlayed_image)



def predict_random_and_show(model, dataset):
    indices = random.sample(range(len(dataset)), 6)  # Change max_index to the maximum index available in your dataset

    plt.figure(figsize=(20, 20))

    for idx, index in enumerate(indices, 1):
        X, mask = dataset[index]
        X = X.to(device)
        model.eval()
        with torch.inference_mode():
            output = model(X.unsqueeze(dim=0)).to(device)

        W, H = output.shape[2], output.shape[3]

        # Get RGB segmentation map
        segmented_image = draw_segmentation_map(output)

        # Resize to original image size
        segmented_image = cv2.resize(segmented_image, (W, H), cv2.INTER_LINEAR)

        plt.subplot(6, 3, 3 * (idx - 1) + 1)
        cmap = ListedColormap(label_map / 255.0)
        plt.imshow(mask, cmap=cmap, vmin=0, vmax=13)
        plt.title(f"Ground Truth - Image {index}")
        plt.axis("off")

        plt.subplot(6, 3, 3 * (idx - 1) + 2)
        plt.title(f"Segmentation - Image {index}")
        plt.axis("off")
        plt.imshow(segmented_image)

    plt.tight_layout()
    plt.show()
    plt.close()



def show_mask(mask):
  cmap = ListedColormap(label_map / 255.0)

  plt.imshow(mask, cmap=cmap, vmin=0, vmax=13)

  num_labels = len(label_map)
  ticks = np.arange(num_labels)

  cbar = plt.colorbar(ticks=ticks)
  cbar.ax.set_yticklabels(labels, fontsize=8)
  plt.axis("off")
  plt.show()


def draw_segmentation_map(outputs):
    labels = torch.argmax(outputs.squeeze(), dim=0).cpu().numpy()

    red_map   = np.zeros_like(labels).astype(np.uint8)
    green_map = np.zeros_like(labels).astype(np.uint8)
    blue_map  = np.zeros_like(labels).astype(np.uint8)

    for label_num in range(0, len(label_map)):
        index = labels == label_num

        R, G, B = label_map[label_num]

        red_map[index]   = R
        green_map[index] = G
        blue_map[index]  = B

    segmentation_map = np.stack([red_map, green_map, blue_map], axis=2)
    return segmentation_map
