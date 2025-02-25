# TODO: Define here your validation loops.
import torch
import numpy as np
from utils import total_hist
from utils import per_class_iou

def validation(model, dataloader_val, criterion):
  model.eval()
  val_loss = 0.0
  hist = 0

  with torch.no_grad():
      for i, (inputs, labels) in enumerate(dataloader_val, 0):
          inputs, labels = inputs.cuda(), labels.cuda()
          outputs = model(inputs)
          loss = criterion(outputs, labels)
          val_loss += loss.item()
          outputs = torch.argmax(outputs, dim=1)
          hist += total_hist(outputs, labels, 19)

  avg_val_loss = val_loss / len(dataloader_val)
  miou_per_class = per_class_iou(hist)
  miou = np.mean(miou_per_class)
  print(f"Avg. Validation Loss: {avg_val_loss}, mIoU: {miou}")
  return avg_val_loss, miou, miou_per_class