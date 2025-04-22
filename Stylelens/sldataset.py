import torch
import os
import pandas as pd
from PIL import Image
from torchvision import transforms

import config as c

class SLDataset(torch.utils.data.Dataset):
    def __init__(self, image_filename, caption_filename, img_dir, transform=None):
        
        self.img_dir = img_dir

        self.image_paths = pd.read_csv(image_filename, sep='\t')        

        self.text_labels = pd.read_csv(caption_filename, sep='\t')

        self.labels = c.get_label()

        self.transform = transform

        self.preprocess = transforms.Compose([
            transforms.Resize(224),
            #transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
    def __getitem__(self, idx):
        item = {}

        #img_path = os.path.join(self.img_dir, self.image_paths.iloc[idx, 0])
        img_path = self.image_paths.iloc[idx, 0]
        item['img_path'] = img_path

        image = Image.open(img_path)
        if self.transform:
            image = self.preprocess(image)
        item['image'] = image        
        
        item['caption'] = self.text_labels.iloc[idx, 0]
        
        words = self.image_paths.iloc[idx, 0].split('/')
        item['target_label'] = torch.tensor(self.labels.get(words[2]))

        return item
    
    def __len__(self):
        return len(self.image_paths)

if __name__ == '__main__':    
        
    train_dataset = SLDataset(image_filename=c.gal_image_filename, caption_filename=c.gal_caption_filename, img_dir=c.img_dir, transform='transform')
    
    c=0
    for batch in train_dataset:
        if c > 5:
            break
        print(batch['caption'])
        c +=1 