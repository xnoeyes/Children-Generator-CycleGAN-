import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class KinFaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.relationships = ['father-dau', 'father-son', 'mother-dau', 'mother-son']
        self.pairs = self._load_pairs()

    def _load_pairs(self):
        pairs = []
        for relationship in self.relationships:
            relationship_dir = os.path.join(self.root_dir, relationship)
            images = sorted(os.listdir(relationship_dir))

            parent_images = [img for img in images if img.endswith('_1.jpg')]
            child_images = [img for img in images if img.endswith('_2.jpg')]

            for parent_image, child_image in zip(parent_images, child_images):
                if 'Thumbs.db' in [parent_image, child_image]:
                    continue
                
                parent_image_path = os.path.join(relationship_dir, parent_image)
                child_image_path = os.path.join(relationship_dir, child_image)

                if 'father' in relationship:
                    parent_gender = 'male'
                else:
                    parent_gender = 'female'

                if 'dau' in relationship:
                    child_gender = 'female'
                else:
                    child_gender = 'male'

                pairs.append((parent_image_path, child_image_path, parent_gender, child_gender))
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        parent_image_path, child_image_path, parent_gender, child_gender = self.pairs[idx]
        parent_image = Image.open(parent_image_path).convert('RGB')
        child_image = Image.open(child_image_path).convert('RGB')
        parent_image = np.array(parent_image)
        child_image = np.array(child_image)

        if self.transform:
            parent_image = self.transform(image=parent_image)["image"]
            child_image = self.transform(image=child_image)["image"]

        parent_gender_onehot = torch.tensor([1, 0]) if parent_gender == 'male' else torch.tensor([0, 1])
        child_gender_onehot = torch.tensor([1, 0]) if child_gender == 'male' else torch.tensor([0, 1])

        return parent_image, child_image, parent_gender_onehot, child_gender_onehot
