import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import transforms

import config


class BinaryCifar(Dataset):

    def __init__(self, data, targets, transform=None):
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, index):
        image = self.data[index]
        if self.transform is not None:
            image = self.transform(image)
        return image, self.targets[index]

    def __len__(self):
        return len(self.targets)


def build_transforms():
    # normalize value if calculated based on train set
    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    return transform_train, transform_test


def load_cifar_2(use_transform=True):
    project_root = config.project_root
    data_path = '{}/models/CNN/data'.format(project_root)
    training_file = np.load('{}/cifar_binary_training.npz'.format(data_path))
    testing_file = np.load('{}/cifar_binary_testing.npz'.format(data_path))
    if not use_transform:
        training_dataset = BinaryCifar(training_file['data'], training_file['targets'], transform=None)
        testing_dataset = BinaryCifar(testing_file['data'], testing_file['targets'], transform=None)
        return training_dataset, testing_dataset
    transform_train, transforms_test = build_transforms()
    training_dataset = BinaryCifar(training_file['data'], training_file['targets'], transform_train)
    testing_dataset = BinaryCifar(testing_file['data'], testing_file['targets'], transforms_test)
    return training_dataset, testing_dataset
