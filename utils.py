import numpy as np
import torch
import time

def poly_lr_scheduler(optimizer, init_lr, iter, lr_decay_iter=1,
                      max_iter=300, power=0.9):
    """Polynomial decay of learning rate
            :param init_lr is base learning rate
            :param iter is a current iteration
            :param lr_decay_iter how frequently decay occurs, default is 1
            :param max_iter is number of maximum iterations
            :param power is a polymomial power

    """
    # if iter % lr_decay_iter or iter > max_iter:
    # 	return optimizer

    lr = init_lr*(1 - iter/max_iter)**power
    optimizer.param_groups[0]['lr'] = lr
    return lr
    # return lr


def fast_hist(a, b, n):
    '''
    a and b are predict and mask respectively
    n is the number of classes
    '''
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iou(hist):
    epsilon = 1e-5
    return (np.diag(hist)) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + epsilon)


def total_hist(outputs, labels, num_classes):
    hist = 0
    for i in range(len(outputs)):
        output, label = outputs[i].cpu().detach().numpy().reshape(-1,), labels[i].cpu().detach().numpy().reshape(-1,)
        hist += fast_hist(label, output, num_classes)
    return hist
    
def latency_fps(model, train_dataset, transform, device = 'cuda', iterations = 1000):
    height = train_dataset[0][0].shape[0]
    width = train_dataset[0][0].shape[1]
    image = np.random.randint(0,256,(height, width, 3)) / 255.
    image = transform(image)
    image = torch.unsqueeze(image, dim=0).float().to(device)
    
    latency = np.zeros(iterations)
    fps = np.zeros(iterations)
    for i in range(iterations):
      start = time.time()
      output = model(image)
      end = time.time()
      time_diff_seconds = end - start
      latency[i] = time_diff_seconds
      fps[i] = 1/time_diff_seconds
    
    meanLatency = np.mean(latency)*1000
    stdLatency = np.std(latency)*1000
    meanFPS = np.mean(fps)
    stdFPS = np.std(fps)
    
    print(f"Mean Latency: {meanLatency} ms")
    print(f"Std Latency: {stdLatency} ms")
    print(f"Mean FPS: {meanFPS}")
    print(f"Std FPS: {stdFPS}")