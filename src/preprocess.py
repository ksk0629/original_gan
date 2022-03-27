import torch
from torchvision import datasets, transforms


def preprocess_from_path(data_dir: str, horizontal_flip_p: float, random_apply_p: float, image_size: int, batch_size: int, shuffle: bool) -> torch.utils.data.DataLoader:
    random_transforms = [transforms.ColorJitter(), transforms.RandomRotation(degrees=20)]
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    transforms.RandomHorizontalFlip(p=horizontal_flip_p),
                                    transforms.RandomApply(random_transforms, p=random_apply_p),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_data = datasets.ImageFolder(data_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, shuffle=shuffle, batch_size=batch_size)

    return train_loader
