import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
import numpy as np

def calculate_accuracy(pred, target):
  _, predicted = torch.max(pred, 1)
  correct_pixels = (predicted == target).sum().item()
  total_pixels = target.numel()
  accuracy = correct_pixels / total_pixels * 100
  return accuracy

def calculate_f1(pred, target, num_classes):
  f1_list = []
  for class_id in range(num_classes):
    pred_mask = (pred == class_id)
    target_mask = (target == class_id)
    true_positive = (pred_mask & target_mask).sum().item()
    false_positive = (pred_mask & ~target_mask).sum().item()
    false_negative = (~pred_mask & target_mask).sum().item()
    precision = true_positive / (true_positive + false_positive + 1e-8)
    recall = true_positive / (true_positive + false_negative + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)  # Add a small epsilon to avoid division by zero
    f1_list.append(f1)
  return sum(f1_list) / len(f1_list)

def calculate_iou(pred, target, num_classes):
  iou_list = []
  for class_id in range(num_classes):
    pred_mask = (pred == class_id)
    target_mask = (target == class_id)
    intersection = (pred_mask & target_mask).sum().item()
    union = (pred_mask | target_mask).sum().item()
    iou = intersection / (union + 1e-8)  # Add a small epsilon to avoid division by zero
    iou_list.append(iou)
  return sum(iou_list) / len(iou_list)


def validate(model, val_dataloader, loss_fn):
  model.eval()
  device = next(model.parameters()).device

  val_loss = 0.0
  val_accuracy = 0.0
  val_iou = 0.0
  val_f1 = 0.0
  val_total = 0

  model.eval()
  with torch.inference_mode():
    for batch_idx, (inputs, targets) in enumerate(val_dataloader):
      inputs, targets = inputs.to(device), targets.to(device)

      outputs = model(inputs)
      loss = loss_fn(outputs, targets.long())

      val_loss += loss.item()

      accuracy = calculate_accuracy(outputs, targets)
      iou = calculate_iou(torch.argmax(outputs, dim=1), targets, num_classes=14)
      f1 = calculate_f1(torch.argmax(outputs, dim=1), targets, num_classes=14)

      val_accuracy += accuracy
      val_iou += iou
      val_f1 += f1

      val_total += targets.size(0)

  mean_val_loss = val_loss / len(val_dataloader)
  mean_val_accuracy = val_accuracy / len(val_dataloader)
  mean_val_iou = val_iou / len(val_dataloader)
  mean_val_f1 = val_f1 / len(val_dataloader)

  return mean_val_loss, mean_val_accuracy, mean_val_iou, mean_val_f1



def train(model, train_data_loader, val_data_loader, loss_fn, optimizer, num_epochs):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    running_loss = 0.0
    running_acc = 0.0
    running_iou = 0.0
    running_f1 = 0.0

    loss_epoch = []
    acc_epoch = []
    iou_epoch = []
    f1_epoch = []

    for batch_idx, (inputs, targets) in enumerate(train_data_loader):
      inputs, targets = inputs.to(device), targets.to(device)
      # Zero the parameter gradients
      optimizer.zero_grad()

      # Forward pass
      outputs = model(inputs)

      accuracy = calculate_accuracy(outputs, targets)
      iou = calculate_iou(torch.argmax(outputs, dim=1), targets, 14)
      f1 = calculate_f1(torch.argmax(outputs, dim=1), targets, num_classes=14)

      # Calculate the loss
      loss = loss_fn(outputs, targets.long())

      # Backward pass and optimize
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      running_acc += accuracy
      running_iou += iou
      running_f1 += f1

      if batch_idx % 10 == 9:
        mean_loss = running_loss / 10
        mean_acc = running_acc / 10
        mean_iou = running_iou / 10
        mean_f1 = running_f1 / 10

        running_loss = 0.0
        running_acc = 0.0
        running_iou = 0.0
        running_f1 = 0.0

        loss_epoch.append(mean_loss)
        acc_epoch.append(mean_acc)
        iou_epoch.append(mean_iou)
        f1_epoch.append(mean_f1)

    print(f"Epoch {epoch + 1} | Train Loss: {(sum(loss_epoch) / len(loss_epoch)):.4f} | Train Accuracy: {(sum(acc_epoch) / len(acc_epoch)):.4f}% | Train mIOU: {(sum(iou_epoch) / len(iou_epoch)):.4f} | Train mF1: {(sum(f1_epoch) / len(f1_epoch)):.4f}")

    if val_data_loader != None:
      val_loss, val_accuracy, val_iou, val_f1 = validate(model, val_data_loader, loss_fn)
      print(f"Epoch {epoch + 1} | Val Loss: {val_loss:.4f}   | Val Accuracy: {val_accuracy:.4f}%   | Val mIOU: {val_iou:.4f}   | Val mF1: {val_f1:.4f}")
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"----- Epoch Time: {epoch_time:.2f}s -----")

