

from PIL import Image
import cv2

import torch

from model import SiameseNetwork
from utils import get_default_device, show_img, to_device
from transforms import Transforms



def predict_image(target, img, transform=None):
    xb, xb2 = transform(target).unsqueeze(0), transform(img).unsqueeze(0) # Convert to batch of 1
    model.eval()
    yb = model(xb.to(device), xb2.to(device))
    return yb


device = get_default_device()
model = SiameseNetwork()
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cuda')))
to_device(model, device)

t = Transforms()

target = ""
img = ""

target = cv2.imread(target)
img = cv2.imread(img)

print(float(predict_image(target, img, transform=t))) # > 0 means match, < 0 means no match
show_img(target, img)


# -3.3546

