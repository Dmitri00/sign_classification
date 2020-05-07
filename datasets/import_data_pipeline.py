import torch
from torchvision import datasets, transforms

from datasets import dataset_splitters

data_transforms = {
    'train': transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ColorJitter(0.4, 0.1, 0.1),
        transforms.RandomAffine(20),
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
    ]),
}

dataset_split_pipeline = dataset_splitters.DatasetSubsetPipeline(dataset_splitters.SubsetDatasetsByTopFreq(3))


def prepare_datasets(dataset_class, root, transforms, splitters):
    datasets = {image_set: dataset_class(root, image_set, transforms[image_set])
                for image_set in ['train', 'val']}
    return datasets


def prepare_dataloaders(datasets,
                        samplers,
                        batch_size,
                        num_workers):
    dataloaders = {x: torch.utils.data.DataLoader(datasets[x], batch_size=batch_size,
                                                  sampler=samplers[x], num_workers=num_workers)
                   for x in ['train', 'val']}
    return dataloaders


def prepare_data(dataset_class, root, transforms, splitters,
                 samplers_classes, batch_size, num_workers):
    datasets = prepare_datasets(dataset_class, root, transforms, splitters)
    datasets = splitters(datasets)
    samplers = {x: samplers_classes[x](datasets[x])
                for x in ['train', 'val']}
    dataloaders = prepare_dataloaders(datasets, samplers, batch_size, num_workers)
    return datasets, dataloaders
