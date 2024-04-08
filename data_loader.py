import re
import os
import fnmatch
from PIL import Image
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from tqdm.contrib.concurrent import process_map  # Use tqdm's process_map

import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

def list_files_by_type(folder_path, file_type):
    filtered_files = []
    for file in os.listdir(folder_path):
        if fnmatch.fnmatch(file, f"*.{file_type}"):
            filtered_files.append(os.path.join(folder_path, file))
    return filtered_files

def process_image(args):
    file_path, label, transform = args  # Unpack the tuple
    image = Image.open(file_path)
    default_transform = transforms.ToTensor()
    tensor = default_transform(image)
    if transform:
        tensor = transform(tensor)
    return tensor, torch.tensor(label, dtype=torch.long)

class CIFAKEDataset(Dataset):
    @staticmethod
    def extract_index_and_category(file_path):
        filename = os.path.basename(file_path)
        pattern = r"(\d+)(?: \((\d+)\))?\..+"
        match = re.match(pattern, filename)
        if match:
            index = int(match.group(1))
            category = int(match.group(2)) if match.group(2) else 0
            return index, category
        else:
            return None
    
    @staticmethod
    def load_folder(folder_path, label, category=None, transform=None, num_processes=1):
        print(f"Loading folder: {folder_path}")
        files = list_files_by_type(folder_path, "jpg")
        if category is not None:
            files = [file for file in files if CIFAKEDataset.extract_index_and_category(file)[1] == category]

        # Use process_map from tqdm.contrib.concurrent for better tqdm updates
        results = process_map(process_image, [(file, label, transform) for file in files], max_workers=num_processes, chunksize=1)

        x = torch.stack([result[0] for result in results])
        y = torch.stack([result[1] for result in results])
        return x, y
        
    def __init__(self, folder_path, category=None, transform=None, num_processes=1):
        label_1_folders = [
            os.path.join(folder_path, "train/REAL"),
            os.path.join(folder_path, "test/REAL"),
        ]
        label_0_folders = [
            os.path.join(folder_path, "train/FAKE"),
            os.path.join(folder_path, "test/FAKE"),
        ]
        x1, y1 = CIFAKEDataset.load_folder(label_1_folders[0], 1, category, transform, num_processes)
        x2, y2 = CIFAKEDataset.load_folder(label_0_folders[0], 0, category, transform, num_processes)
        x3, y3 = CIFAKEDataset.load_folder(label_1_folders[1], 1, category, transform, num_processes)
        x4, y4 = CIFAKEDataset.load_folder(label_0_folders[1], 0, category, transform, num_processes)
        self.x = torch.cat((x1, x2, x3, x4))
        self.y = torch.cat((y1, y2, y3, y4))

    def __len__(self):
        return self.y.size(0)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
    
    def data_dim(self):
        return self.x[0].size()
    
    def show_example(self, idx):
        x, y = self[idx]
        image_array = x.permute(1, 2, 0).numpy()
        plt.imshow(image_array)
        plt.title(f"Label: {y}")
        plt.axis('off')
        plt.show()


if __name__ == "__main__":
    folder_path = "data/CIFAKE"
    file_type = "jpg"
    print("Loading CIFAKEDataset...")
    data_set = CIFAKEDataset(folder_path, num_processes=4)
    print("Dataset loaded successfully!")
    print("Dataset length:", len(data_set))
    print("Data dimension:", data_set.data_dim())
    print("Showing example image...")
    data_set.show_example(0)
