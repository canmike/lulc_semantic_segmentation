import torch
import torch.nn as nn
import torch.optim as optim
import time
from tqdm import tqdm
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def calculate_accuracy(pred, target):
  _, predicted = torch.max(pred, 1)
  correct_pixels = (predicted == target).sum().item()
  total_pixels = target.numel()
  accuracy = correct_pixels / total_pixels * 100
  return accuracy

from torchmetrics import F1Score
def calculate_f1(pred, target, num_classes):
  f1 = F1Score(task="multiclass", num_classes=num_classes).to(device)
  return f1(pred.to(device), target.to(device))
  
  
from torchmetrics import JaccardIndex
def calculate_iou(output, mask, num_classes):
  output = output.to(device)
  mask = mask.to(device)

  jaccard = JaccardIndex(task="multiclass", num_classes=num_classes, average="weighted").to(device)
  return jaccard(output, mask)

from torchmetrics import Precision
def calculate_precision(output, mask, num_classes):
  precision = Precision(task="multiclass", num_classes=num_classes, average="weighted").to(device)
  return precision(output.to(device), mask.to(device))

from torchmetrics import Recall
def calculate_recall(output, mask, num_classes):
  recall = Recall(task="multiclass", num_classes=num_classes, average="weighted").to(device)
  return recall(output.to(device), mask.to(device))

def validate(model, val_dataloader, loss_fn):
  device = next(model.parameters()).device

  val_loss = 0.0
  val_accuracy = 0.0
  val_iou = 0.0
  val_precision = 0.0
  val_recall = 0.0
  val_total = 0

  model.eval()
  with torch.inference_mode():
    for batch_idx, (inputs, targets) in enumerate(val_dataloader):
      inputs, targets = inputs.to(device), targets.to(device)

      outputs = model(inputs)
      loss = loss_fn(outputs, targets.long())

      val_loss += loss.item()

      accuracy = calculate_accuracy(outputs, targets)
      iou = calculate_iou(outputs, targets, num_classes=14)
      precision = calculate_precision(outputs, targets, 14)
      recall = calculate_recall(outputs, targets, 14)

      val_accuracy += accuracy
      val_iou += iou
      val_precision += precision
      val_recall += recall

      val_total += targets.size(0)

  mean_val_loss = val_loss / len(val_dataloader)
  mean_val_accuracy = val_accuracy / len(val_dataloader)
  mean_val_iou = val_iou / len(val_dataloader)
  mean_val_precision = val_precision / len(val_dataloader)
  mean_val_recall = val_recall / len(val_dataloader)

  mean_val_f1 = 2*(mean_val_precision * mean_val_recall) / (mean_val_precision + mean_val_recall)

  return mean_val_loss, mean_val_accuracy, mean_val_iou, mean_val_f1, mean_val_precision, mean_val_recall


def train(model, train_data_loader, val_data_loader, loss_fn, optimizer, num_epochs, validate_train=False):
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  model.to(device)

  for epoch in range(num_epochs):
    start_time = time.time()
    model.train()
    for batch_idx, (inputs, targets) in enumerate(train_data_loader):
      inputs, targets = inputs.to(device), targets.to(device)
      # Zero the parameter gradients
      optimizer.zero_grad()

      # Forward pass
      outputs = model(inputs)

      # Calculate the loss
      loss = loss_fn(outputs, targets.long())

      # Backward pass and optimize
      loss.backward()
      optimizer.step()

    if validate_train:
      train_loss, train_accuracy, train_iou, train_f1, train_precision, train_recall = validate(model, train_data_loader, loss_fn)
      print(f"Epoch {epoch + 1} | Train Loss:   {train_loss:.4f} | Train Accuracy:   {train_accuracy:.4f}% | Train mIOU:   {train_iou:.4f} | Train mF1:   {train_f1:.4f} | Train Precision:   {train_precision:.4f} | Train Recall:   {train_recall:.4f}")
    
    if val_data_loader != None:
      val_loss, val_accuracy, val_iou, val_f1, val_precision, val_recall = validate(model, val_data_loader, loss_fn)
      print(f"Epoch {epoch + 1} | Val Loss:   {val_loss:.4f} | Val Accuracy:   {val_accuracy:.4f}% | Val mIOU:   {val_iou:.4f} | Val mF1:   {val_f1:.4f} | Val Precision:   {val_precision:.4f} | Val Recall:   {val_recall:.4f}")
    end_time = time.time()
    epoch_time = end_time - start_time
    print(f"----- Epoch Time: {epoch_time:.2f}s -----")
