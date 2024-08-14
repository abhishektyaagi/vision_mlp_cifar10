from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pdb

class TinyImageNetHFDataset(Dataset):
    def __init__(self, split='train', transform=None):
        # Load dataset from Hugging Face
        #self.dataset = load_dataset('Maysee/tiny-imagenet', split=split)
        self.dataset = load_dataset("zh-plus/tiny-imagenet", split=split)
        #pdb.set_trace()
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Extract image and label
        item = self.dataset[idx]
        image = item['image']

        # Convert image to RGB if it is grayscale
        if image.mode != 'RGB':
            image = image.convert('RGB')

        label = item['label']

        # Apply any transformations
        if self.transform:
            image = self.transform(image)

        return image, label

