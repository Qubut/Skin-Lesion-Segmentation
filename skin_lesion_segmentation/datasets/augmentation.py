from torchvision import transforms
import random


def transform_A(image):
    return transforms.ColorJitter(
        brightness=random.uniform(0, 0.5), 
        contrast=random.uniform(0, 0.5), 
        saturation=random.uniform(0, 0.5) )(image)


def transform_B(image, mask):
    angle = random.uniform(-90,90)
    shear= random.uniform(-20,20)
    scale= random.uniform(0.8,1.5) 
    image = transforms.functional.affine(image, angle=angle, translate=(0,0), shear=shear, scale=scale)
    mask = transforms.functional.affine(mask, angle=angle, translate=(0,0), shear=shear, scale=scale)

    return image, mask

def transform_C(image,mask):
    if random.random() > 0.5:
        image = transforms.functional.vflip(image)
        mask = transforms.functional.hflip(mask)

    if random.random() > 0.5:
        image = transforms.functional.hflip(image)
        mask = transforms.functional.hflip(mask)

    return image, mask

def transform_D(image, mask):
    # Get the size of the image (PIL gives (width, height))
    width, height = image.size

    # Define the scale and ratio for cropping
    scale = (0.4, 1.0)
    ratio = (3/4, 4/3)

    # Obtain the same random crop parameters for both image and mask
    i, j, h, w = transforms.RandomResizedCrop.get_params(image, scale=scale, ratio=ratio)
    
    # Apply the crop and then resize back to original dimensions if desired.
    # Here, we resize the cropped region back to the original image size.
    image = transforms.functional.resized_crop(image, i, j, h, w, size=(height, width))
    mask  = transforms.functional.resized_crop(mask, i, j, h, w, size=(height, width))
    return image, mask

def transform_E(image, mask):
    alpha_value = random.uniform(10,450)
    elastic_transform = transforms.ElasticTransform(alpha=alpha_value,sigma=11.0)

    image = elastic_transform(image)
    mask = elastic_transform(mask)

    return image, mask

class RandomAug():
    def __call__(self, data):
        image, mask = data

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        to_tensor = transforms.ToTensor()

        if random.random() > 0.5: 
            image = transform_A(image)
        if random.random() > 0.5: 
            image, mask = transform_B(image, mask)
        if random.random() > 0.5: 
            image, mask = transform_C(image,mask)
        if random.random() > 0.5: 
            image, mask = transform_D(image,mask)
        if random.random() > 0.5:
            image, mask = transform_E(image, mask)

        image = normalize(to_tensor(image))
        mask = to_tensor(mask)
        
        return image, mask