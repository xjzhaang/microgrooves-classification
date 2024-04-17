import os
import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import skimage


class MyoblastDataset(Dataset):
    def __init__(self, cell_type, exp_ids, mode, transform=None):
        self.cell_type = cell_type
        self.exp_ids = exp_ids
        self.mode = mode
        self.transform = transform
        if self.mode == "train" or self.mode == "val":
            self.data_csv = pd.read_csv("data/" + self.cell_type + "/data_cropped.csv")
            self.data = self.data_csv[self.data_csv["Exp ID"].isin(self.exp_ids)]
            # # self.data = self.data[self.data["class"].isin([0, 1, 2, 4])]
            # # self.data = self.data.replace({"class": {4: 3}})
            unique_ids = self.data['original ID'].unique()
            
            id_class_df = pd.DataFrame({'original ID': unique_ids})
            id_class_df = id_class_df.merge(self.data[['original ID', 'class']], on='original ID', how='left').drop_duplicates()
            
            id_class_df['Exp ID'] = id_class_df['original ID'].apply(lambda x: x.split('_')[0])
            id_class_df = id_class_df.astype(str)
            id_class_df['strat'] = id_class_df['Exp ID'] + id_class_df['class']
            train_ids, val_ids = train_test_split(id_class_df, test_size=0.25, random_state=42, stratify=id_class_df["strat"])
            
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
            # self.data = self.data[self.data["class"].isin([0, 1, 2, 4])]
            # self.data = self.data.replace({"class": {4: 3}})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        path = self.data.iloc[idx]['path']
        label = self.data.iloc[idx]['class'].astype(int)
        image = skimage.io.imread(os.path.join(path))
        if self.mode == "test":
            if len(image.shape) > 2:
                new_image = np.zeros((2, max(image.shape), max(image.shape)))
                for channel in range(image.shape[0]):
                    padd = np.max(image[channel].shape) - np.min(image[channel].shape)
                    pad_width = ((0, padd), (0, 0)) 
                    new_image[channel] = np.pad(image[channel], pad_width, mode='constant', constant_values=0)
                image = new_image    
            else:
                padd = np.max(image.shape) - np.min(image.shape)
                pad_width = ((0, padd), (0, 0)) 
                image = np.pad(image, pad_width, mode='constant', constant_values=0)
        if len(image.shape) == 2: 
            image = np.expand_dims(image, axis=0)
        data = {'image': torch.tensor(image), 'label': torch.tensor(label), "meta": path}
        if self.transform:
            data = self.transform(data)
        
        return data


class TransformDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        if self.transform:
            sample = self.transform(sample)
        return sample
