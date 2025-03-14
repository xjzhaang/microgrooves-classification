import os
import torch
import pandas as pd
import numpy as np
import cv2
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import skimage
from scipy.ndimage import distance_transform_edt
from sklearn.utils import resample
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

class MyoblastDataset(Dataset):
    def __init__(self, cell_type, exp_ids, mode, include_mask=True, transform=None):
        self.cell_type = cell_type
        self.exp_ids = exp_ids
        self.mode = mode
        self.include_mask = include_mask
        self.transform = transform

        self.data_csv = pd.read_csv("data/" + self.cell_type + "/data_cropped.csv")
        self.data = self.data_csv[self.data_csv["Exp ID"].isin(self.exp_ids) & ~self.data_csv["original ID"].str.contains("5-7-5.4", na=False) & ~self.data_csv["original ID"].str.contains("5-3-5.4", na=False)]
        # self.data = self.data[self.data["class"] != 0]
        # self.data["class"] = self.data["class"] - 1
        #self.data = self.data[self.data["class"] != 3]
        #self.data.loc[self.data["class"].isin([1, 2, 3, 4]), "class"] = 1

        class_counts = self.data['class'].value_counts()
        mean_samples = class_counts.mean()
        std_samples = class_counts.std()
        
        resampled_classes = []
        
        for class_label in class_counts.index:
            class_data = self.data[self.data['class'] == class_label]
            n_samples = len(class_data)
            
            # Calculate target samples proportionally
            if n_samples < mean_samples - std_samples:
                # Small classes: oversample to mean - std/2
                target = int(mean_samples - std_samples/2)
                replace = True
            elif n_samples > mean_samples + std_samples:
                # Large classes: undersample to mean + std/2
                target = int(mean_samples + std_samples/2)
                replace = False
            else:
                # Medium classes: keep as is
                resampled_classes.append(class_data)
                continue
                
            resampled = resample(class_data,
                                replace=replace,
                                n_samples=target,
                                random_state=42)
            resampled_classes.append(resampled)
        
        self.data = pd.concat(resampled_classes)

        unique_ids = self.data['original ID']
        
        id_class_df = pd.DataFrame({'original ID': unique_ids})
        id_class_df = id_class_df.merge(self.data[['original ID', 'class']], on='original ID', how='left').drop_duplicates()
        
        id_class_df['Exp ID'] = id_class_df['original ID'].apply(lambda x: x.split('_')[0])
        id_class_df = id_class_df.astype(str)
        id_class_df['strat'] = id_class_df['Exp ID'] + id_class_df['class']
        value_counts = id_class_df['strat'].value_counts()
        print("Class distribution:")
        print(value_counts)


        if self.mode == "train" or self.mode == "val_exp" or self.mode == "val_original" or self.mode == "test_original":
            train_ids, test_val_ids = train_test_split(id_class_df, test_size=0.30, random_state=42, stratify=id_class_df["strat"])
        #test_ids, val_ids = train_test_split(test_val_ids, test_size=0.5, random_state=42, stratify=test_val_ids["strat"])
        
        if self.mode == "train":
            self.data = self.data[self.data['original ID'].isin(train_ids['original ID'])]
            
        if self.mode == "val_exp":
            self.data = self.data[self.data['original ID'].isin(test_val_ids['original ID'])]

        if self.mode == "val_original":
            self.data_csv = pd.read_csv("data/" + self.cell_type + "/data.csv")
            self.data = self.data_csv[self.data_csv["Exp ID"].isin(self.exp_ids)]
            self.data = self.data[self.data['original ID'].isin(test_val_ids['original ID'])]               
        if self.mode == "cropped":
            self.data_csv = pd.read_csv("data/" + self.cell_type + "/data_cropped.csv")
            self.data = self.data_csv[self.data_csv["Exp ID"].isin(self.exp_ids)]
        if self.mode == "test_exp":
            self.data_csv = pd.read_csv("data/" + self.cell_type + "/data.csv")
            self.data = self.data_csv[self.data_csv["Exp ID"].isin(self.exp_ids) & ~self.data_csv["original ID"].str.contains("5-7-5.4", na=False) & ~self.data_csv["original ID"].str.contains("5-3-5.4", na=False)]
            # self.data = self.data[self.data["class"] != 0]
            # self.data["class"] = self.data["class"] - 1

            #self.data = self.data[self.data["class"] != 3]
            #self.data.loc[self.data["class"].isin([1, 2, 3, 4]), "class"] = 1

        if self.mode == "test_original":
            self.data_csv = pd.read_csv("data/" + self.cell_type + "/data.csv")
            self.data = self.data_csv[self.data_csv["Exp ID"].isin(self.exp_ids)]
            self.data = self.data[self.data['original ID'].isin(test_ids['original ID'])]
            # self.data = self.data[self.data["class"].isin([0, 4])]
            # self.data = self.data.replace({"class": {0:0, 4:1}})
            combined_ids = pd.concat([train_ids['original ID'], val_ids['original ID'], test_ids['original ID']])
            has_duplicates = combined_ids.duplicated().any()
            
            if has_duplicates:
                print("There are duplicate 'original ID' values across the DataFrames.")
            else:
                print("All 'original ID' values are unique across the DataFrames.")

    def compute_distance_transform(self, mask):
        # Compute the distance transform for the given mask
        distance_transform = distance_transform_edt(1 - mask.numpy())
        #max_distance = distance_transform.max()
        # if max_distance > 0:
        #     distance_transform = distance_transform / max_distance
        return distance_transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        mask_condition = (self.mode == "train" or self.mode == "cropped")
        path = self.data.iloc[idx]['path']
        
        label = self.data.iloc[idx]['class'].astype(int)
        image = skimage.io.imread(os.path.join(path))
        if self.include_mask and self.mode != "test_exp":
            path_mask = self.data.iloc[idx]['path'].replace("cropped", "cropped_mask")
            try:
                image_mask = skimage.io.imread(os.path.join(path_mask)).astype(int)
            except:
                image_mask = 0
        if self.mode == "test_exp":
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
            if self.include_mask and mask_condition:
                image_mask = np.expand_dims(image_mask, axis=0)
        if self.include_mask and mask_condition:
            data = {'image': torch.tensor(image), 'label': torch.tensor(label), "meta": path, 'mask': torch.tensor(image_mask)}
        elif self.include_mask and self.mode == "val_exp":
            #image[image_mask == 0] = 0
            data = {'image': torch.tensor(image), 'label': torch.tensor(label), "meta": path}
        else:
            data = {'image': torch.tensor(image), 'label': torch.tensor(label), "meta": path}
        if self.transform:
            data = self.transform(data)
            if self.include_mask and mask_condition:
                data["mask"][data["mask"] > 0] = 1
                data["mask"] = torch.tensor(self.compute_distance_transform(data["mask"]))
            # skimage.io.imsave("mask.tif", data["mask"].numpy())
            # skimage.io.imsave("image.tif", data["image"].numpy())
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
