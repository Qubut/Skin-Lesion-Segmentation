import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from augmentation import RandomAug

class LesionDataset(Dataset):
    def __init__(self, root_path, img_size, augmentation=False, test=False):
        self.root_path = root_path
        self.img_size = img_size

        if test:
            self.images = sorted([os.path.join(root_path, "test", i) for i in os.listdir(os.path.join(root_path, "test"))])
            self.masks = sorted([os.path.join(root_path, "test_mask", i) for i in os.listdir(os.path.join(root_path, "test_mask"))])
        else:
            self.images = sorted([os.path.join(root_path, "train", i) for i in os.listdir(os.path.join(root_path, "train"))])
            self.masks = sorted([os.path.join(root_path, "train_mask", i) for i in os.listdir(os.path.join(root_path, "train_mask"))])

        self.augment = RandomAug() if augmentation else None

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        self.mask_transform = transforms.Compose([
            transforms.ToTensor(),  # No normalization for binary mask
        ])

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        mask = Image.open(self.masks[index]).convert("L")  # Keep it grayscale
        image = transforms.Resize((self.img_size, self.img_size))(image)
        mask = transforms.Resize((self.img_size, self.img_size))(mask)


        if self.augment:
            image, mask = self.augment((image, mask))
        else:
            image = self.transform(image)
            mask = self.mask_transform(mask)

        return image, mask

    def __len__(self):
        return len(self.images)
