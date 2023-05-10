import torch
import matplotlib.pyplot as plt

def get_default_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')
    
    
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

def show_img(img1, img2):
    axs = plt.figure(figsize=(9, 9)).subplots(1, 2)
    axs[0].imshow(img1)
    axs[1].imshow(img2)


def accuracy(preds, labels):
    preds = torch.flatten(preds)
    preds[preds > 0] = 1
    preds[preds < 0] = 0
    return torch.sum(preds == labels).item() / len(labels)


class DeviceDataLoader():
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device
    
    def __iter__(self):
        for batch in self.dl:
            yield to_device(batch, self.device)
    
    def __len__(self):
        return len(self.dl)