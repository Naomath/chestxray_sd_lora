
from PIL import Image, ImageOps
from torch.utils.data import Dataset
import torchvision.transforms as T
import os
from .utils import find_images

class ChestXrayDataset(Dataset):
    def __init__(self, root, resolution=256):
        self.files = list(find_images(root))
        if len(self.files) == 0:
            raise ValueError(f"No images found in {root}")
        self.resolution = resolution
        self.transforms = T.Compose([
            T.CenterCrop(min(resolution, resolution)),
            T.Resize((resolution, resolution), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),  # [0,1], CxHxW
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        img = Image.open(path)
        # Convert to grayscale if needed
        if img.mode != 'L':
            img = ImageOps.grayscale(img)
        # replicate to 3 channels for SD
        img = self.transforms(img)  # 1xHxW
        img = img.repeat(3, 1, 1)   # 3xHxW
        example = {
            "pixel_values": img,
            "prompt": "chest x-ray, normal",
        }
        return example
