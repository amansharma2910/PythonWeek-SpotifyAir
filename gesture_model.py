# importing project dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2 as cv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


# method to preprocess the dataset

def preprocess_dataset(csv_path, selected_gestures):
    """Preprocesses the CSV dataset to extract required data + apply scaling
    Args: 
        csv_path::string- Path to the csv file 
        selected_gestures::list- List of integers specifying selected labels
    Returns:
        images::np.ndarray- Array of images
        labels::np.ndarray- Array of labels
    """
    # reading dataframe
    df = pd.read_csv(csv_path)

    # selecting 4 gestures for pause, play, next and previous
    mask = df['label'].isin(selected_gestures)
    df = df[mask]

    # separating labels and image data
    images = df.drop(columns=['label']).to_numpy()     
    labels = df.label.to_numpy()

    # renaming labels
    for g, l in zip(selected_gestures, range(4)):
        labels = np.where(labels==g, l, labels)

    # reshaping image data
    images = images.reshape(images.shape[0], 28, 28, 1) # m, n_W, n_H, n_C

    # normalizing images
    images = images / 255

    return images, labels


selected_gestures = [0, 1, 21, 22] # indexes for selected gestures
train_path = 'sign_mnist_train/sign_mnist_train.csv'
test_path = 'sign_mnist_test/sign_mnist_test.csv'

train_x, train_y = preprocess_dataset(train_path, selected_gestures)
test_x, test_y = preprocess_dataset(test_path, selected_gestures)


# creating dataset

class GestureDataset(Dataset):
    def __init__(self, images, labels, transforms):
        self.images = images
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        if self.transforms:
            image = self.transforms(self.images[idx])
        else:
            image = self.images[idx]
        label = self.labels[idx]
        return {'image': image, 'label': label}


# training dataset
transform =  transforms.ToTensor()
train_dataset = GestureDataset(train_x, train_y, transform)

# validation dataset
test_dataset = GestureDataset(test_x, test_y, transform)