from torch.utils.data import DataLoader
from torchvision.datasets import FakeData

from datasets.import_data_pipeline import data_transforms
from datasets.import_data_pipeline import dataset_split_pipeline
from datasets.import_data_pipeline import prepare_data
from datasets.traffic_sign import TrafficSign


def import_data(root, batch_size, num_workers):
    default_transforms = data_transforms
    default_splitter = dataset_split_pipeline

    datasets = {}
    dataloaders = {}
    for image_set in ('train', 'val'):
        ds = FakeData(size=1000, image_size=(3, 48, 48), num_classes=10,
                      transform=default_transforms[image_set])
        datasets[image_set] = ds
        dataloaders[image_set] = DataLoader(ds, batch_size=batch_size, num_workers=num_workers)
    return datasets, dataloaders
