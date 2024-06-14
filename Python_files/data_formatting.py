import pandas as pd
import torch
import numpy as np
from PIL import Image
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler, Dataset, Subset
import torch.nn.functional as F
import os
import random
from collections import defaultdict
import ast



def format_data_distancing(df, interpolate=True):
    pixel = df.columns[0:64]
    sessions_unique = df['session'].unique()
    num_sessions = len(sessions_unique)

    # Preallocate a list of lists
    session_arrays = [None] * num_sessions
    
    for i, session_label in enumerate(sessions_unique):
        df_temp = df[df['session'] == session_label]
        # Preallocate list for tuples (image, label) for each image
        pics_tuples = []

        for idx, row in df_temp.iterrows():
            # Extract pixel values and check if it can be reshaped to 8x8
            pic = pd.to_numeric(row[pixel], errors='coerce').values
            label = row['violations']
            min_value = pic.min()
            if pic.size == 64:
                if interpolate:
                    try:
                        pic_reshaped = pic.reshape(1, 1, 8, 8)
                    except Exception:
                        continue
                    pic_reshaped = torch.Tensor(pic_reshaped)
                    pic_reshaped = F.interpolate(pic_reshaped, size=(16, 16), 
                                                 mode='bilinear', align_corners=None, 
                                                 recompute_scale_factor=None, antialias=False)
                    pic_reshaped = pic_reshaped.cpu().detach().numpy().reshape(1, 16, 16)
                else:
                    pic_reshaped = pic.reshape((1, 8, 8))
                pic_reshaped = pic_reshaped - min_value
                pics_tuples.append((pic_reshaped, int(label)))
            else:
                raise ValueError("The image does not have exactly 64 pixels to reshape into 8x8")

        session_arrays[i] = pics_tuples
        
    return session_arrays


def prepare_data_distancing(df, sequence=False, seq_len=8, interpolate = True):
    data = format_data_distancing(df, interpolate = interpolate)
    data_collection = []

    if sequence:
        for seq in data:
            num_frames = len(seq)
            for i in range(num_frames - seq_len + 1):
                frames = np.concatenate([seq[j][0] for j in range(i, i + seq_len)])
                label = seq[i + seq_len - 1][1]
                data_collection.append((frames, label))
    else:
        for seq in data:
            data_collection.extend(seq)
    
    return data_collection


def format_data_localization(df):

    pixel = df.columns[0:64]
    sessions = np.unique(df["session"])
    # Assuming 'df' is a pandas DataFrame
    sessions_unique = df['session'].unique()
    num_sessions = len(sessions_unique)

    # Preallocate a list of NumPy arrays with known dimensions
    session_arrays = [None] * num_sessions
    
    for i, session_label in enumerate(sessions_unique):
        df_temp = df[df['session'] == session_label]
        # Preallocate object array for tuples (image, empty array) for each image
        pics_tuples = np.empty((len(df_temp),), dtype=object)

        for idx, row in df_temp.iterrows():
            # Extract pixel values and check if it can be reshaped to 8x8
            pic = pd.to_numeric(row[pixel], errors='coerce').values
            min_value = pic.min()
            if pic.size == 64:
                pic_reshaped = pic.reshape((8, 8))
                empty_8x8 = np.zeros((8, 8))  # Create an empty 8x8 array

                targets = ast.literal_eval(row["target_coordinates"])

                for p in targets:

                    #a = p[0]
                    #b = p[1]
                    a = p[1]
                    b = p[0]

                    empty_8x8[a][b] = 1
                pic_reshaped = pic_reshaped - min_value
                pics_tuples[idx - df_temp.index[0]] = (pic_reshaped.reshape(1, 8, 8), empty_8x8)
            else:
                raise ValueError("The image does not have exactly 64 pixels to reshape into 8x8")

        session_arrays[i] = pics_tuples 
        
    return session_arrays


def prepare_data_localization(df, sequence=True, seq_len=8):
    data = format_data_localization(df)
    data_collection = []

    if sequence:
        for seq in data:
            num_frames = len(seq)
            for i in range(num_frames - seq_len + 1):
                frames = np.concatenate([seq[j][0] for j in range(i, i + seq_len)])
                detection_map = seq[i + seq_len - 1][1]
                frames_expanded = np.expand_dims(frames, axis=1)
                data_collection.append((frames_expanded, detection_map))
    else:
        for seq in data:
            data_collection.extend(seq)
    
    return data_collection

class SimpleTorchDataset(Dataset):
    def __init__(self, data_collection):
        self.data_collection = data_collection

    def __len__(self):
        return len(self.data_collection)

    def __getitem__(self, idx):
        frames, label = self.data_collection[idx]
        
        # Ensure frames and label are numpy arrays of appropriate types
        frames = np.array(frames, dtype=np.float32)
        label = np.array(label, dtype=np.float32)
        
        # Convert to PyTorch tensors
        frames = torch.tensor(frames, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.float32)
        
        return frames, label


def create_datasets(data_collection_train, data_collection_test):
    # Creating PyTorch datasets
    train_set = SimpleTorchDataset(data_collection_train)
    test_set = SimpleTorchDataset(data_collection_test)
    
    return train_set, test_set



def dataset_to_loader(torch_dataset, batch_size, balancing = True, shuffle = True):
    if balancing == True:
        loader = class_balancing(torch_dataset, batch_size)   
    else:
        loader = DataLoader(torch_dataset, batch_size = batch_size, shuffle=shuffle)
    return loader
    


def class_balancing(torch_dataset, batch_size, random_seed=42):
    if random_seed is not None:
        torch.manual_seed(random_seed)  # Fix the random seed for reproducibility

    # Get the labels from the dataset
    labels = np.array([int(torch_dataset[i][1].item()) for i in range(len(torch_dataset))])

    # Compute class frequencies
    class_counts = np.bincount(labels)

    # Calculate class weights
    class_weights = 1. / class_counts

    # Assign weights to each sample based on its class
    sample_weights = class_weights[labels]

    # Create a WeightedRandomSampler
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights))

    # Return a DataLoader with the WeightedRandomSampler
    return DataLoader(torch_dataset, batch_size=batch_size, sampler=sampler)

#In case if you decide to normalize data. in our experience it spoiled the performance and simple min subtraction is already enough
def get_normalization_param(dataloader):
    mean = 0.0
    for images, _ in dataloader:
        batch_samples = images.size(0) 
        images = images.view(batch_samples, images.size(1), -1)
        mean += images.mean(2).sum(0)
    mean = mean / len(dataloader.dataset)
    print(f'Calculated mean: {mean}')
    var = 0.0
    print(images.shape)
    for images, _ in dataloader:
        batch_samples = images.size(0)
        images = images.view(batch_samples, images.size(1), -1)
        var += ((images - mean.unsqueeze(1))**2).sum([0,2])
    std = torch.sqrt(var / (len(dataloader.dataset)*8*8))
    print('Calculated standard deviation: {std}')
    return mean, std


