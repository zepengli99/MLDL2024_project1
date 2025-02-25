# TODO: Define here your training loops.
import torch
import numpy as np
from utils import total_hist
from utils import per_class_iou

def train(epoch, model, dataloader_train, criterion, optimizer):
  model.train()
  running_loss = 0.0
  hist = 0
  for i, (inputs, labels) in enumerate(dataloader_train, 0):
      inputs, labels = inputs.cuda(), labels.cuda()
      outputs, _, _ = model(inputs)
      loss = criterion(outputs, labels)

      optimizer.zero_grad()
      loss.backward()
      optimizer.step()

      running_loss += loss.item()
      outputs = torch.argmax(outputs, dim=1)
      hist += total_hist(outputs, labels, 19)
  avg_loss = running_loss / len(dataloader_train)
  miou_per_class = per_class_iou(hist)
  miou = np.mean(miou_per_class)
  print(f"Epoch{epoch+1} Avg. Training Loss: {avg_loss}, mIoU: {miou}")
  return avg_loss, miou, miou_per_class