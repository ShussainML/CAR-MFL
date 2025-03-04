import os
import pickle
import re
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

def clean_report_mimic_cxr(report):
    report_cleaner = lambda t: t.replace('\n', ' ').replace('__', '_').replace('__', '_').replace('__', '_') \
        .replace('__', '_').replace('__', '_').replace('__', '_').replace('__', '_').replace('  ', ' ').replace('\r', '') \
        .replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ').replace('  ', ' ') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.').replace('..', '.') \
        .replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '').replace('. 2. ', '. ') \
        .replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ').replace(' 2. ', '. ') \
        .replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
        .strip().lower().split('. ')
    sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '')
                                    .replace('\\', '').replace("'", '').strip().lower())
    tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
    report = ' . '.join(tokens) + ' .'
    return report

def ensure_directory_exists(directory):
    """Ensure the directory exists; if not, create it."""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def load_pickle_file(file_path):
    """Load a pickle file and handle errors."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}. Please ensure the file exists.")
    with open(file_path, 'rb') as pickle_file:
        return pickle.load(pickle_file)

class MimicMultiModal(Dataset):
    def __init__(self, data_root, ann_root, view_type="view1", split="train"):
        super().__init__()

        # Ensure the annotation directory exists
        ensure_directory_exists(ann_root)

        # Construct the path to the annotation file
        annFile = os.path.join(ann_root, f'mimic-cxr-{view_type}.pkl')

        # Load the annotation file
        self.data = load_pickle_file(annFile)[split]

        # Define transforms for training and validation
        if split == "train":
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize the image to a fixed size (e.g., 224x224)
                transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally for data augmentation
                transforms.RandomRotation(15),  # Randomly rotate the image by up to 15 degrees
                transforms.ToTensor(),  # Convert the image to a PyTorch tensor (0-1 range)
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image using precomputed values
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # Ensure the image directory exists
        self.image_root = os.path.join(data_root, "mimic-cxr-resized/files/")
        ensure_directory_exists(self.image_root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        report = data_item['report']

        relative_path = data_item["image_path"][0].replace("jpg", "png")
        img_path = os.path.join(self.image_root, relative_path)

        # Check if the image file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}. Please ensure the file exists.")

        image = Image.open(img_path).convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        label = data_item["label"]
        cleaned_report = clean_report_mimic_cxr(report)
        return image, torch.tensor(label), cleaned_report, idx

class MimicPublic(Dataset):
    def __init__(self, data_root, ann_root, view_type="view1", dst_type="train"):
        super().__init__()

        # Ensure the annotation directory exists
        ensure_directory_exists(ann_root)

        # Construct the path to the annotation file
        annFile = os.path.join(ann_root, f'mimic-cxr-{view_type}.pkl')

        # Load the annotation file
        self.data = load_pickle_file(annFile)["train"]

        # Define transforms for training and validation
        if dst_type == "train":
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),  # Resize the image to a fixed size (e.g., 224x224)
                transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally for data augmentation
                transforms.RandomRotation(15),  # Randomly rotate the image by up to 15 degrees
                transforms.ToTensor(),  # Convert the image to a PyTorch tensor (0-1 range)
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image using precomputed values
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # Ensure the image directory exists
        self.image_root = os.path.join(data_root, "mimic-cxr-resized/files/")
        ensure_directory_exists(self.image_root)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_item = self.data[idx]
        report = data_item['report']

        relative_path = data_item["image_path"][0].replace("jpg", "png")
        img_path = os.path.join(self.image_root, relative_path)

        # Check if the image file exists
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Image file not found: {img_path}. Please ensure the file exists.")

        image = Image.open(img_path).convert('RGB')
        if self.transforms:
            image = self.transforms(image)
        label = data_item["label"]
        cleaned_report = clean_report_mimic_cxr(report)
        return image, torch.tensor(label), cleaned_report, idx
