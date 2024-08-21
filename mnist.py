import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
import torch.utils.data as data
import torch
import numpy as np
import os
from params import *
import trimesh
from PIL import Image


batch_size = 8
class TargetDomainDataset(data.Dataset):
    def __init__(self, root, npoints=1024, classes=None, transform=None):
        self.root = root
        self.npoints = npoints
        self.transform = transform

        self.categories = classes if classes else self._get_categories()
        self.files = self._get_files()

    def _get_categories(self):
        return sorted(os.listdir(self.root))

    def _get_files(self):
        files = []
        for category in self.categories:
            category_path = os.path.join(self.root, category)
            for model in os.listdir(category_path):
                if model.endswith('.obj'):
                    files.append((category, os.path.join(category_path, model)))
        return files

    def _load_model(self, path):
        mesh = trimesh.load(path)
        points = mesh.sample(self.npoints)
        return points

    def __getitem__(self, index):
        category, path = self.files[index]
        points = self._load_model(path)
        label = self.categories.index(category)

        points = torch.tensor(points, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            points = self.transform(points)

        return points, label

    def __len__(self):
        return len(self.files)


# Transforms
class RandomRotate(object):
    def __call__(self, pointcloud):
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta),  np.cos(theta), 0],
                                    [0, 0, 1]])
        return torch.matmul(pointcloud, torch.tensor(rotation_matrix, dtype=torch.float32))

class RandomScale(object):
    def __call__(self, pointcloud):
        scale = np.random.uniform(0.8, 1.25)
        return pointcloud * scale

class RandomTranslate(object):
    def __call__(self, pointcloud):
        translation = np.random.uniform(-0.2, 0.2, size=(3,))
        return pointcloud + torch.tensor(translation, dtype=torch.float32)

class Jitter(object):
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, pointcloud):
        jittered_data = pointcloud + torch.clamp(self.sigma * torch.randn_like(pointcloud), -self.clip, self.clip)
        return jittered_data

class Normalize(object):
    def __call__(self, pointcloud):
        centroid = torch.mean(pointcloud, dim=0)
        pointcloud = pointcloud - centroid
        furthest_distance = torch.max(torch.sqrt(torch.sum(pointcloud ** 2, dim=1)))
        pointcloud = pointcloud / furthest_distance
        return pointcloud

class ToTensor(object):
    def __call__(self, pointcloud):
        return torch.tensor(pointcloud, dtype=torch.float32)


# Transform pipeline
transform = transforms.Compose([
    RandomRotate(),
    RandomScale(),
    RandomTranslate(),
    Jitter(),
    Normalize(),
    ToTensor(),
])

target_root_dir = 'target'  # Target domain


target_dataset = TargetDomainDataset(root=target_root_dir, npoints=2048, transform=transform)

# Data Loader for target domain (no labels, only points)
target_loader = DataLoader(target_dataset, batch_size=batch_size, shuffle=True)

# Example: Iterate through the target domain data loader
for points, labels in target_loader:
    print("Target Domain Points:", points.shape)  # Should be (batch_size, npoints, 3)
    print("Target Domain Labels:", labels.shape)  # Should be (batch_size,)
    break
