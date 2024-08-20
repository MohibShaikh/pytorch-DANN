import torchvision.datasets as datasets
from torch.utils.data import SubsetRandomSampler, DataLoader
from torchvision import transforms
import torch.utils.data as data
import torch
import numpy as np
import os
import errno
from PIL import Image
import params


# MNIST-M
class ModelNetDataset(data.Dataset):
    """Princeton ModelNet Dataset for 3D object classification."""
    
    def __init__(self, root, npoints=1024, split='train', classes=None, transform=None):
        """
        Args:
            root (str): Root directory of the ModelNet dataset.
            npoints (int): Number of points to sample from the 3D model.
            split (str): 'train' or 'test'.
            classes (list of str): List of class names to load. Load all classes if None.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.npoints = npoints
        self.split = split
        self.transform = transform

        self.categories = classes if classes else self._get_categories()
        self.files = self._get_files()

    def _get_categories(self):
        """Returns the list of object categories."""
        return sorted(os.listdir(os.path.join(self.root, self.split)))

    def _get_files(self):
        """Returns a list of file paths for all 3D models."""
        files = []
        for category in self.categories:
            category_path = os.path.join(self.root, self.split, category)
            for model in os.listdir(category_path):
                if model.endswith('.off'):
                    files.append((category, os.path.join(category_path, model)))
        return files

    def _load_model(self, path):
        """Loads a 3D model and samples points from its surface."""
        mesh = trimesh.load(path)
        points = mesh.sample(self.npoints)  # Sample npoints from the surface
        return points

    def __getitem__(self, index):
        """Get the 3D point cloud and the corresponding target class."""
        category, path = self.files[index]
        points = self._load_model(path)
        label = self.categories.index(category)

        points = torch.tensor(points, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            points = self.transform(points)

        return points, label

    def __len__(self):
        """Return the size of the dataset."""
        return len(self.files)

class RandomRotate(object):
    """Randomly rotate the point cloud around the z-axis."""
    def __call__(self, pointcloud):
        theta = np.random.uniform(0, 2 * np.pi)
        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
                                    [np.sin(theta),  np.cos(theta), 0],
                                    [0, 0, 1]])
        return torch.matmul(pointcloud, torch.tensor(rotation_matrix, dtype=torch.float32))

class RandomScale(object):
    """Randomly scale the point cloud."""
    def __call__(self, pointcloud):
        scale = np.random.uniform(0.8, 1.25)
        return pointcloud * scale

class RandomTranslate(object):
    """Randomly translate the point cloud."""
    def __call__(self, pointcloud):
        translation = np.random.uniform(-0.2, 0.2, size=(3,))
        return pointcloud + torch.tensor(translation, dtype=torch.float32)

class Jitter(object):
    """Randomly jitter the points in the point cloud."""
    def __init__(self, sigma=0.01, clip=0.05):
        self.sigma = sigma
        self.clip = clip

    def __call__(self, pointcloud):
        jittered_data = pointcloud + torch.clamp(self.sigma * torch.randn_like(pointcloud), -self.clip, self.clip)
        return jittered_data

class Normalize(object):
    """Normalize the point cloud to fit in a unit sphere."""
    def __call__(self, pointcloud):
        centroid = torch.mean(pointcloud, dim=0)
        pointcloud = pointcloud - centroid
        furthest_distance = torch.max(torch.sqrt(torch.sum(pointcloud ** 2, dim=1)))
        pointcloud = pointcloud / furthest_distance
        return pointcloud

class ToTensor(object):
    """Convert point cloud numpy array to PyTorch tensor."""
    def __call__(self, pointcloud):
        return torch.tensor(pointcloud, dtype=torch.float32)



transform = transforms.Compose([
    RandomRotate(),
    RandomScale(),
    RandomTranslate(),
    Jitter(),
    Normalize(),
    ToTensor()


])


root_dir = '/path/to/ModelNet40'  # Change to your ModelNet root directory
dataset = ModelNetDataset(root=root_dir, npoints=2048, split='train', transform=transform)


train_loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

# Example: Iterate through the data loader
for points, labels in train_loader:
    print(points.shape)  # Should be (batch_size, npoints, 3)
    print(labels.shape)  # Should be (batch_size,)
    break

dataset_size = len(dataset)
indices = list(range(dataset_size))
validation_size = int(0.2 * dataset_size)  # 20% for validation
test_size = int(0.1 * dataset_size)        # 10% for testing

# Shuffle the dataset and split indices
np.random.shuffle(indices)
train_indices, val_indices, test_indices = indices[validation_size + test_size:], indices[:validation_size], indices[validation_size:validation_size + test_size]

# Create samplers for each dataset
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)

# Create DataLoader for each split
batch_size = 32
train_loader = DataLoader(dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4)
val_loader = DataLoader(dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4)
test_loader = DataLoader(dataset, batch_size=batch_size, sampler=test_sampler, num_workers=4)

# Example: Iterate through the train data loader
for points, labels in train_loader:
    print(points.shape)  # Should be (batch_size, npoints, 3)
    print(labels.shape)  # Should be (batch_size,)
    break

# Example: Iterate through the validation data loader
for points, labels in val_loader:
    print(points.shape)  # Should be (batch_size, npoints, 3)
    print(labels.shape)  # Should be (batch_size,)
    break

# Example: Iterate through the test data loader
for points, labels in test_loader:
    print(points.shape)  # Should be (batch_size, npoints, 3)
    print(labels.shape)  # Should be (batch_size,)
    break