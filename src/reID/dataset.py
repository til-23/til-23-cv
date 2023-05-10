import os
import matplotlib.pyplot as plt

from PIL import Image
import cv2

from torch.utils.data import Dataset
from transforms import Transforms




class PlushieTrainDataset(Dataset):
    
    def __init__(self, filepath, img_dir, transform=None):
        self.samples = []
        self.img_dir = img_dir
        self.transform = transform

        with open(filepath, 'r') as f:
            self.samples = [line.strip() for line in f]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        line = self.samples[i].split()
        if len(line) == 3:
            anchor_name, anchor_num, img_num = line
            img_name = anchor_name
            is_same = 1
        elif len(line) == 4:
            anchor_name, anchor_num, img_name, img_num = line
            is_same = 0
        else:
            print(len(line), line)
            raise Exception("Shouldn't be here")
        
        anchor = cv2.imread(os.path.join(self.img_dir, str(anchor_name), f"{anchor_name}_{anchor_num}.png"))
        img = cv2.imread(os.path.join(self.img_dir, img_name, f"{img_name}_{img_num}.png"))
        
        if self.transform:
            anchor = self.transform(anchor)
            img = self.transform(img)

        return anchor, img, is_same



def main():
    t = Transforms()
    filepath = ""
    img_dir = ""
    d = PlushieTrainDataset(filepath=filepath, img_dir=img_dir, transform=t)
    
    e = d[0]
    axs = plt.figure(figsize=(9, 9)).subplots(1, 2)
    plt.title(e[2])
    axs[0].imshow(e[0].permute(1,2,0))
    axs[1].imshow(e[1].permute(1,2,0))

if __name__ == "__main__":
    main()
