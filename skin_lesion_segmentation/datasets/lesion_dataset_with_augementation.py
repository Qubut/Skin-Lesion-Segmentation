import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
import random
import torch

class RandomAug:
    """
    Custom augmentation pipeline combining multiple transformations with probability control

    Example usage:
        augmentor = RandomAug()
        augmented_image, augmented_mask = augmentor(raw_image, raw_mask)
    """
    def __init__(self):
        self.color_jitter = transforms.RandomApply([
            transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5)
        ], p=0.5)

        self.affine = transforms.RandomApply([
            transforms.RandomAffine(
                degrees=(-90, 90),
                translate=(0.1, 0.1),
                scale=(0.8, 1.5),
                shear=(-20, 20),
                interpolation=InterpolationMode.BILINEAR
            )
        ], p=0.5)

        self.flip = transforms.RandomApply([
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomHorizontalFlip(p=0.5)
        ], p=0.5)

        self.crop = transforms.RandomApply([
            transforms.RandomResizedCrop(
                size=(256, 256),  # Will be overridden by dataset's img_size
                scale=(0.4, 1.0),
                ratio=(3/4, 4/3),
                interpolation=InterpolationMode.BILINEAR
            )
        ], p=0.5)

        self.elastic = transforms.RandomApply([
            transforms.ElasticTransform(
                alpha=random.uniform(10, 450),
                sigma=11.0,
                interpolation=InterpolationMode.BILINEAR
            )
        ], p=0.5)

    def __call__(self, image, mask):
        image = self.color_jitter(image)

        # Apply geometric transforms to both image and mask
        if self.affine.p > random.random():
            params = transforms.RandomAffine.get_params(
                self.affine.transforms[0].degrees,
                self.affine.transforms[0].translate,
                self.affine.transforms[0].scale,
                self.affine.transforms[0].shear,
                image.size
            )
            image = transforms.functional.affine(image, *params, interpolation=InterpolationMode.BILINEAR)
            mask = transforms.functional.affine(mask, *params, interpolation=InterpolationMode.NEAREST)

        if self.flip.p > random.random():
            if random.random() > 0.5:
                image = transforms.functional.vflip(image)
                mask = transforms.functional.vflip(mask)
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                mask = transforms.functional.hflip(mask)

        if self.crop.p > random.random():
            i, j, h, w = transforms.RandomResizedCrop.get_params(
                image,
                scale=self.crop.transforms[0].scale,
                ratio=self.crop.transforms[0].ratio
            )
            image = transforms.functional.resized_crop(
                image, i, j, h, w,
                size=image.size,
                interpolation=InterpolationMode.BILINEAR
            )
            mask = transforms.functional.resized_crop(
                mask, i, j, h, w,
                size=mask.size,
                interpolation=InterpolationMode.NEAREST
            )

        if self.elastic.p > random.random():
            alpha = random.uniform(10, 450)
            elastic_image = transforms.ElasticTransform(alpha, 11.0, interpolation=InterpolationMode.BILINEAR)
            elastic_mask = transforms.ElasticTransform(alpha, 11.0, interpolation=InterpolationMode.NEAREST)
            image = elastic_image(image)
            mask = elastic_mask(mask)

        return image, mask

class LesionDataset(Dataset):
    """
    Dataset class for skin lesion segmentation

    Args:
        root_path (str): Path to dataset root directory containing:
            - train/ (folder with training images)
            - train_mask/ (folder with corresponding training masks)
            - test/ (folder with test images)
            - test_mask/ (folder with corresponding test masks)
        img_size (int): Target image size (resized to square)
        augmentation (bool): Whether to apply augmentations (default: False)
        test (bool): Whether to load test set instead of training set (default: False)

    Example usage:
        Training setup:
            train_dataset = LesionDataset(
                root_path='/data/lesions',
                img_size=256,
                augmentation=True
            )
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

            for images, masks in train_loader:
                # Training loop

        Test setup:
            test_dataset = LesionDataset(
                root_path='/data/lesions',
                img_size=256,
                test=True
            )
            test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

            for images, masks in test_loader:
                # Inference loop

    Expected directory structure:
        root_path/
        ├── train/
        │   ├── image1.jpg
        │   └── image2.jpg
        ├── train_mask/
        │   ├── image1_mask.png
        │   └── image2_mask.png
        ├── test/
        │   └── ...
        └── test_mask/
            └── ...
    """
    def __init__(self, root_path, img_size, augmentation=False, test=False):
        self.root_path = root_path
        self.img_size = img_size
        self.augmentation = augmentation


        data_split = "test" if test else "train"
        image_dir = os.path.join(root_path, data_split)
        mask_dir = os.path.join(root_path, f"{data_split}_mask")


        image_names = sorted(os.listdir(image_dir))
        mask_names = sorted(os.listdir(mask_dir))


        self.samples = []
        for img_name in image_names:
            img_base = os.path.splitext(img_name)[0]
            mask_name = next((n for n in mask_names if n.startswith(img_base)), None)
            if mask_name:
                self.samples.append((
                    os.path.join(image_dir, img_name),
                    os.path.join(mask_dir, mask_name)
                ))

        self.augmentor = RandomAug() if augmentation else None
        self.to_tensor = transforms.ToTensor()
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index of the sample to retrieve

        Returns:
            tuple: (image_tensor, mask_tensor)
        """

        img_path, mask_path = self.samples[index]
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Ensure mask is grayscale

        # Resize to target dimensions
        image = transforms.Resize((self.img_size, self.img_size))(image)
        mask = transforms.Resize((self.img_size, self.img_size),
                                interpolation=InterpolationMode.NEAREST)(mask)

        if self.augmentation:
            image, mask = self.augmentor(image, mask)

        image = self.normalize(self.to_tensor(image))
        mask = (self.to_tensor(mask)*255).type(torch.uint8)

        return image, mask

    def __len__(self):
        """Return the number of samples in the dataset"""
        return len(self.samples)
