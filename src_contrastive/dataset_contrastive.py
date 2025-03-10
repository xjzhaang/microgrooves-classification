import os 
import torch
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import skimage

class SimCLRDataset(Dataset):
    def __init__(self, cell_type, exp_ids, mode, transform=None):
        self.cell_type = cell_type
        self.exp_ids = exp_ids
        self.mode = mode
        self.transform = transform
        if self.mode == "train" or self.mode == "val":
            self.data_csv = pd.read_csv("data/" + self.cell_type + "/data_cropped.csv")
            self.data = self.data_csv[self.data_csv["Exp ID"].isin(self.exp_ids) & ~self.data_csv["original ID"].str.contains("5-7-5.4", na=False) & ~self.data_csv["original ID"].str.contains("5-3-5.4", na=False)]
            unique_ids = self.data['original ID'].unique()
            
            id_class_df = pd.DataFrame({'original ID': unique_ids})
            id_class_df = id_class_df.merge(self.data[['original ID', 'class']], on='original ID', how='left').drop_duplicates()
            
            id_class_df['Exp ID'] = id_class_df['original ID'].apply(lambda x: x.split('_')[0])
            id_class_df = id_class_df.astype(str)
            id_class_df['strat'] = id_class_df['Exp ID'] + id_class_df['class']
            train_ids, val_ids = train_test_split(id_class_df, test_size=0.3, random_state=42, stratify=id_class_df["strat"])
            
            if self.mode == "train":
                self.data = self.data[self.data['original ID'].isin(train_ids['original ID'])]
            else:
                self.data = self.data[self.data['original ID'].isin(val_ids['original ID'])]
        elif self.mode == "cropped":
            self.data_csv = pd.read_csv("data/" + self.cell_type + "/data_cropped.csv")
            self.data = self.data_csv[self.data_csv["Exp ID"].isin(self.exp_ids)]
        elif self.mode == "test":
            self.data_csv = pd.read_csv("data/" + self.cell_type + "/data.csv")
            self.data = self.data_csv[self.data_csv["Exp ID"].isin(self.exp_ids)]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data.iloc[idx]['path']
        label = self.data.iloc[idx]['class'].astype(int)
        image = skimage.io.imread(os.path.join(path))
        if self.transform:
            image = self.transform(Image.fromarray(image))
        data = {'image': image, 'label': torch.tensor(label), "meta": path}
        return data